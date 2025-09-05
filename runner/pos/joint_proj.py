import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
import argparse
import copy
from tqdm import tqdm
from omegaconf import OmegaConf
from einops import rearrange
from diffusers import DDPMScheduler

from model.internvl import extract_feature_pre_adapter
from model.internvl.modeling_internvl_chat import InternVLChatModel

from util.misc import process_pretrained_model_path, flatten_dict
from util.my_tool_box import get_accelerator, get_t2i_dataloader

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def intern_add_diffhead_projector(internvl, config):
    from model.dit.diff_mlp import SimpleMLPAdaLN
    diff_head = SimpleMLPAdaLN(
        in_channels    = config.diffhead.x_dim,
        model_channels = config.diffhead.hidden_size,
        out_channels   = config.diffhead.x_dim,
        z_channels     = config.diffhead.z_dim,
        num_res_blocks = config.diffhead.depth,
    )
    num_parameters = sum(p.numel() for p in diff_head.parameters())
    print(f"diff_head has {num_parameters / 1e6} M parameters")

    down_projector = nn.Sequential(
        nn.Linear(config.clip_feature_dim, 4 * config.clip_feature_dim),
        nn.GELU(),
        nn.Linear(4 * config.clip_feature_dim, config.diffhead.x_dim),
    ) # 4096 -> 16
    num_parameters = sum(p.numel() for p in down_projector.parameters())
    print(f"down_projector has {num_parameters / 1e6} M parameters")
    
    clip_projector = nn.Sequential(
        nn.Linear(config.diffhead.x_dim, config.diffhead.z_dim),
        nn.GELU(),
        nn.Linear(config.diffhead.z_dim, config.diffhead.z_dim),
    ) # 16 -> 3584
    num_parameters = sum(p.numel() for p in clip_projector.parameters())
    print(f"clip_projector has {num_parameters / 1e6} M parameters")

    internvl.requires_grad_(False)

    internvl.diff_head = diff_head
    internvl.diff_head.requires_grad_(True)

    internvl.down_projector = down_projector
    internvl.down_projector.requires_grad_(True)

    internvl.clip_projector = clip_projector
    internvl.clip_projector.requires_grad_(True)

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

    return internvl, train_scheduler


def main(args):
    config = OmegaConf.load(args.config)
    config = process_pretrained_model_path(config)
    accelerator, output_dir = get_accelerator(config)

    internvl = InternVLChatModel.from_pretrained(config.model.internvl_path)
    internvl, train_scheduler = intern_add_diffhead_projector(internvl, config.model)

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

                x_intern = (pixel_values - imagenet_mean) / imagenet_std
                # x_vae = pixel_values * 2 - 1

                with torch.no_grad():
                    x_clip = extract_feature_pre_adapter(internvl.vision_model, x_intern)
                    x_gen = internvl.down_projector(x_clip)
                    # visual_gen_feature = internvl.clip_projector(visual_gen_feature)

                # ----- compute AR loss -----
                B, L = input_ids.shape
                text_embedding = internvl.language_model.get_input_embeddings()(input_ids).clone()
                img_embedding = internvl.clip_projector(x_gen)
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
                gt_feature = rearrange(x_gen.detach(), "B L D -> (B L) D")
                timesteps = torch.randint(0, 1000, (z.shape[0],), dtype=torch.int64, device=z.device)
                noise = torch.randn_like(gt_feature, device=z.device, dtype=z.dtype)
                noisy_latents = train_scheduler.add_noise(gt_feature, noise, timesteps)
                target = train_scheduler.get_velocity(gt_feature, noise, timesteps)
                pred = internvl.diff_head(noisy_latents, timesteps, z)

                loss = torch.nn.functional.mse_loss(pred, target)

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_learn, 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                    global_step += 1
                    progress_bar.update(1)
                    logs = dict(
                        ar_loss = accelerator.gather(loss.detach()).mean().item(),
                    )
                    accelerator.log(logs, step=global_step)
                    progress_bar.set_postfix(**logs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/pos/joint_proj.yaml")
    args = parser.parse_args()
    main(args)