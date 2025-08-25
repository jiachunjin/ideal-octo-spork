import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
from omegaconf import OmegaConf
from einops import rearrange

from diffusers import DDPMScheduler
from model.dit.lumina_next.nextdit import NextDiTCrossAttn, NextDiTCrossAttnConfig
from util.misc import process_pretrained_model_path, flatten_dict
from model.internvl.modeling_internvl_chat import InternVLChatModel
from model.internvl import extract_feature_pre_shuffle_adapter
from util.my_tool_box import get_wds_dataloader, get_accelerator, get_t2i_dataloader

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def add_query(model, config):
    model.query = nn.Parameter(torch.randn(1, config.num_query, config.dim))
    model.query.requires_grad = True
    return model

def main(args):
    config = OmegaConf.load(args.config)
    config = process_pretrained_model_path(config)

    accelerator, output_dir = get_accelerator(config)

    dit_config = NextDiTCrossAttnConfig(**config.dit)
    model = NextDiTCrossAttn(dit_config)
    model = add_query(model, config.query)

    if config.train.resume_path is not None:
        ckpt = torch.load(config.train.resume_path, map_location="cpu", weights_only=True)
        m, u = model.load_state_dict(ckpt, strict=False)
        accelerator.print(f"missing keys: {m}")
        accelerator.print(f"unexpected keys: {u}")
        accelerator.print(f"DiT loaded from {config.train.resume_path}")

    internvl = InternVLChatModel.from_pretrained(config.intern_vl_8b_path)
    internvl.requires_grad_(False)

    params_to_learn = list(p for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(
        params_to_learn,
        lr           = config.train.lr,
        betas        = (0.9, 0.95),
        weight_decay = 5e-2,
        eps          = 1e-8,
    )

    global_step = config.train.global_step if config.train.global_step is not None else 0
    
    if accelerator.mixed_precision == "bf16":
        dtype = torch.bfloat16
    elif accelerator.mixed_precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    dataloader = get_t2i_dataloader(config.data, accelerator)
    model, optimizer = accelerator.prepare(model, optimizer)
    internvl = internvl.to(accelerator.device, dtype).eval()

    training_done = False
    epoch = 0
    progress_bar = tqdm(
        total   = config.train.num_iter,
        initial = global_step,
        desc    = "Steps",
        disable = not accelerator.is_local_main_process,
    )

    config.device_count = accelerator.num_processes
    if accelerator.is_main_process:
        accelerator.init_trackers(config.train.wandb_proj, config=flatten_dict(config))
        with open(os.path.join(output_dir, "config.yaml"), "w") as f:
            OmegaConf.save(config, f)

    accelerator.print(f"Learnable parameters: {sum(p.numel() for p in params_to_learn if p.requires_grad) / 1e6} M")
    accelerator.print(f"Accelerator mixed precision: {accelerator.mixed_precision}")

    imagenet_mean = torch.tensor(IMAGENET_MEAN, device=accelerator.device, dtype=dtype).view(1, 3, 1, 1)
    imagenet_std = torch.tensor(IMAGENET_STD, device=accelerator.device, dtype=dtype).view(1, 3, 1, 1)

    train_scheduler = DDPMScheduler(
        beta_schedule          = "scaled_linear",
        beta_start             = 0.00085,
        beta_end               = 0.012,
        num_train_timesteps    = 1000,
        clip_sample            = False,
        prediction_type        = "v_prediction",
        steps_offset           = 1,
        trained_betas          = None,
        timestep_spacing       = "trailing",
        rescale_betas_zero_snr = True
    )

    while not training_done:
        for batch in dataloader:
            with accelerator.accumulate([model]):
                model.train()
                x = batch["pixel_values"].to(accelerator.device, dtype)
                input_ids = batch["input_ids"].to(accelerator.device)
                attention_mask = batch["attention_mask"].to(accelerator.device)
                x = (x - imagenet_mean) / imagenet_std
                with torch.no_grad():
                    x_clip = extract_feature_pre_shuffle_adapter(internvl.vision_model, x)
                    x_clip = rearrange(x_clip, "b (h w) d -> b d h w", h=32, w=32)

                    B = x_clip.shape[0]

                    text_embedding = internvl.language_model.get_input_embeddings()(input_ids)
                    joint_embedding = torch.cat((text_embedding, model.query.repeat(B, 1, 1)), dim=1)

                    img_mask = torch.ones((B, config.query.num_query), dtype=torch.bool, device=accelerator.device)
                    attention_mask = torch.cat([attention_mask, img_mask], dim=1)

                hidden_states = internvl.language_model(
                    inputs_embeds        = joint_embedding,
                    attention_mask       = attention_mask,
                    output_hidden_states = True,
                ).hidden_states[-1][:, -config.query.num_query:, :]

                print(hidden_states.shape)

                timesteps = torch.randint(0, 1000, (B,), device=accelerator.device, dtype=torch.int64)
                noise = torch.randn_like(x_clip, device=accelerator.device, dtype=dtype)
                noisy_latents = train_scheduler.add_noise(x_clip, noise, timesteps)
                target = train_scheduler.get_velocity(x_clip, noise, timesteps)

                pred = model(noisy_latents, timesteps, hidden_states)
                loss = torch.nn.functional.mse_loss(pred, target)

                accelerator.backward(loss)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_learn, 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                    global_step += 1
                    progress_bar.update(1)

                    logs = dict(
                        clip_loss = accelerator.gather(loss.detach()).mean().item(), 
                    )
                    accelerator.log(logs, step=global_step)
                    progress_bar.set_postfix(**logs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/clip_dit/lumina_dit.yaml")
    args = parser.parse_args()
    main(args)