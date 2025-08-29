import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import math
import torch
import argparse
from tqdm import tqdm
from omegaconf import OmegaConf
from diffusers import DDPMScheduler
from einops import rearrange

from model.dit.diff_mlp import add_diffhead_to_ar_model
from util.misc import process_pretrained_model_path, flatten_dict
from model.internvl.modeling_internvl_chat import InternVLChatModel
from model.internvl import extract_feature_pre_shuffle_adapter
from util.my_tool_box import get_accelerator, get_t2i_dataloader



torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def main(args):
    config = OmegaConf.load(args.config)
    config = process_pretrained_model_path(config)
    accelerator, output_dir = get_accelerator(config)
    
    internvl = InternVLChatModel.from_pretrained(config.intern_vl_2b_path)
    internvl, train_scheduler = add_diffhead_to_ar_model(internvl, config.model)

    vision_model = InternVLChatModel.from_pretrained(config.intern_vl_8b_path).vision_model
    vision_model.requires_grad_(False)

    global_step = config.train.global_step if config.train.global_step is not None else 0
    params_to_learn = list(p for p in internvl.parameters() if p.requires_grad)

    optimizer = torch.optim.AdamW(
        params_to_learn,
        lr           = config.train.lr,
        betas        = (0.9, 0.95),
        weight_decay = 5e-2,
        eps          = 1e-8,
    )

    if accelerator.mixed_precision == "bf16":
        dtype = torch.bfloat16
    elif accelerator.mixed_precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    
    dataloader = get_t2i_dataloader(config.data, accelerator)
    internvl, optimizer = accelerator.prepare(internvl, optimizer)
    vision_model.requires_grad_(False)
    vision_model = vision_model.to(accelerator.device, dtype).eval()

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

    while not training_done:
        for batch in dataloader:
            with accelerator.accumulate([internvl]):
                internvl.train()
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype)
                input_ids = batch["input_ids"].to(accelerator.device)
                attention_mask = batch["attention_mask"].to(accelerator.device)

                pixel_values = (pixel_values - imagenet_mean) / imagenet_std

                with torch.no_grad():
                    x_clip = vision_model(
                        pixel_values         = pixel_values,
                        output_hidden_states = False,
                        return_dict          = True
                    ).last_hidden_state[:, 1:, :]
                    visual_gen_feature = x_clip # (B, 1024, 1024)

                B, L = input_ids.shape
                text_embedding = internvl.language_model.get_input_embeddings()(input_ids).clone()
                img_embedding = internvl.clip_projector(visual_gen_feature)
                joint_embedding = torch.cat((text_embedding, img_embedding), dim=1)
                img_mask = torch.ones((B, config.data.num_img_token), dtype=torch.bool, device=accelerator.device)
                attention_mask = torch.cat([attention_mask, img_mask], dim=1)

                hidden_states = internvl.language_model(
                    inputs_embeds        = joint_embedding,
                    attention_mask       = attention_mask,
                    output_hidden_states = True,
                ).hidden_states[-1]
                hidden_state = hidden_states[:, -config.data.num_img_token-1:-1, :]

                z = rearrange(hidden_state, "B L D -> (B L) D")
                gt_feature = rearrange(visual_gen_feature, "B L D -> (B L) D")
                timesteps = torch.randint(0, 1000, (z.shape[0],), dtype=torch.int64, device=z.device)
                noise = torch.randn_like(gt_feature, device=z.device, dtype=z.dtype)
                noisy_latents = train_scheduler.add_noise(gt_feature, noise, timesteps)
                target = train_scheduler.get_velocity(gt_feature, noise, timesteps)
                pred = internvl.diff_head(noisy_latents, timesteps, z)

                loss = torch.nn.functional.mse_loss(pred.to(dtype), target)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_learn, 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                    global_step += 1
                    progress_bar.update(1)

                    logs = dict(
                        loss_diff_head = accelerator.gather(loss.detach()).mean().item(),
                    )
                    accelerator.log(logs, step=global_step)
                    progress_bar.set_postfix(**logs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/clip_dit/full_tune.yaml")
    args = parser.parse_args()
    main(args)