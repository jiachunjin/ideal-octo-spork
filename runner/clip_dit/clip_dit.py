import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import math
import torch
import argparse
from tqdm import tqdm
from omegaconf import OmegaConf

from diffusers import DDPMScheduler
from model.dit.standard_dit import DiT
from util.misc import process_pretrained_model_path, flatten_dict
from model.internvl.modeling_internvl_chat import InternVLChatModel
from model.internvl import extract_feature_pre_shuffle_adapter
from util.my_tool_box import get_wds_dataloader, get_accelerator

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def main(args):
    config = OmegaConf.load(args.config)
    config = process_pretrained_model_path(config)

    accelerator, output_dir = get_accelerator(config)

    dit_model = DiT(config.dit)
    if config.train.resume_path is not None:
        ckpt = torch.load(config.train.resume_path, map_location="cpu", weights_only=True)
        m, u = dit_model.load_state_dict(ckpt, strict=False)
        accelerator.print(f"missing keys: {m}")
        accelerator.print(f"unexpected keys: {u}")
        accelerator.print(f"DiT loaded from {config.train.resume_path}")

    vision_model = InternVLChatModel.from_pretrained(config.intern_vl_8b_path).vision_model
    vision_model.requires_grad_(False)

    params_to_learn = list(p for p in dit_model.parameters() if p.requires_grad)
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

    dataloader = get_wds_dataloader(config.data, accelerator)
    dit_model, optimizer = accelerator.prepare(dit_model, optimizer)
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
            with accelerator.accumulate([dit_model]):
                dit_model.train()
                x = batch["pixel_values"].to(accelerator.device, dtype)
                y = batch["labels"].to(accelerator.device)
                x = (x - imagenet_mean) / imagenet_std
                with torch.no_grad():
                    x_clip = extract_feature_pre_shuffle_adapter(vision_model, x)

                B = x_clip.shape[0]
                timesteps = torch.randint(0, 1000, (B,), device=accelerator.device, dtype=torch.int64)
                noise = torch.randn_like(x_clip, device=accelerator.device, dtype=dtype)
                noisy_latents = train_scheduler.add_noise(x_clip, noise, timesteps)
                target = train_scheduler.get_velocity(x_clip, noise, timesteps)
                if dit_model.repa:
                    pred, zs = dit_model(noisy_latents, timesteps, y)
                    diff_loss = torch.nn.functional.mse_loss(pred, target)
                    repa_loss = dit_model.get_repa_loss(x_clip, zs)
                    loss = diff_loss + repa_loss
                else:
                    pred = dit_model(noisy_latents, timesteps, y)
                    loss = torch.nn.functional.mse_loss(pred, target)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_learn, 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                    global_step += 1
                    progress_bar.update(1)

                    if dit_model.repa:
                        logs = dict(
                            clip_loss = accelerator.gather(diff_loss.detach()).mean().item(), 
                            repa_loss = accelerator.gather(repa_loss.detach()).mean().item(),
                        )
                    else:
                        logs = dict(
                            clip_loss = accelerator.gather(loss.detach()).mean().item(), 
                        )
                    accelerator.log(logs, step=global_step)
                    progress_bar.set_postfix(**logs)

                    if global_step > 0 and global_step % config.train.save_every == 0 and accelerator.is_main_process:
                        dit_model.eval()
                        state_dict = accelerator.unwrap_model(dit_model).state_dict()
                        save_path = os.path.join(output_dir, f"dit-{config.train.exp_name}-{global_step}")
                        torch.save(state_dict, save_path)
                        print(f"dit saved to {save_path}")

                    accelerator.wait_for_everyone()

        epoch += 1
        accelerator.print(f"epoch {epoch}: finished")
        accelerator.log({"epoch": epoch}, step=global_step)

    accelerator.end_training()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/clip_dit/clip_dit.yaml")
    args = parser.parse_args()
    main(args)
