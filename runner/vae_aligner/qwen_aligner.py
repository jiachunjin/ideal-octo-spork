import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import pprint
import argparse

from tqdm import tqdm
from diffusers import AutoencoderKL
from omegaconf import OmegaConf
from einops import rearrange
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration
from transformers import Qwen2_5_VLForConditionalGeneration


from model.vae_aligner import get_vae_aligner

from util.misc import process_pretrained_model_path, flatten_dict
from runner.qwen_fix.imagenet_dataloader import get_imagenet_dataloader


def get_accelerator(config):
    output_dir = os.path.join(config.root, config.exp_name, config.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    logging_dir = os.path.join(output_dir, config.logging_dir)
    project_config = ProjectConfiguration(project_dir=config.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        log_with                    = None if config.report_to == "no" else config.report_to,
        mixed_precision             = config.mixed_precision,
        project_config              = project_config,
        gradient_accumulation_steps = config.gradient_accumulation_steps,
    )

    return accelerator, output_dir

def main(args):
    config = OmegaConf.load(args.config)
    config = process_pretrained_model_path(config)
    accelerator, output_dir = get_accelerator(config.train)
    accelerator.print("Configuration:")
    accelerator.print(pprint.pformat(OmegaConf.to_container(config, resolve=True), indent=2, width=120).strip('{}'))
    AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = config.data.batch_size
    accelerator.print(AcceleratorState().deepspeed_plugin.deepspeed_config)

    qwen_clip = Qwen2_5_VLForConditionalGeneration.from_pretrained(config.qwen_vl_path).visual
    vae_aligner = get_vae_aligner(config.vae_aligner)
    vae = AutoencoderKL.from_pretrained(config.vae_path)

    qwen_clip.requires_grad_(False)
    vae.requires_grad_(False)

    if config.train.resume_path is not None:
        ckpt = torch.load(config.train.resume_path, map_location="cpu", weights_only=True)
        if config.train.skipped_keys:
            ckpt = {k: v for k, v in ckpt.items() if k not in config.train.skipped_keys}
        m, u = vae_aligner.load_state_dict(ckpt, strict=False)
        accelerator.print(f"Missing modules: {m}, unmatched modules: {u}")

    global_step = config.train.global_step if config.train.global_step is not None else 0

    params_to_learn = list(vae_aligner.parameters())
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
    
    dataloader = get_imagenet_dataloader(config.data, accelerator)

    vae_aligner, optimizer = accelerator.prepare(vae_aligner, optimizer)
    qwen_clip = qwen_clip.to(accelerator.device, dtype).eval()
    vae = vae.to(accelerator.device, dtype).eval()

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
    accelerator.print(f"vae_aligner dtype: {next(vae_aligner.parameters()).dtype}")
    accelerator.print(f"Accelerator mixed precision: {accelerator.mixed_precision}")

    while not training_done:
        for x, _ in dataloader:
            with accelerator.accumulate([vae_aligner]):
                vae_aligner.train()
                qwen_pixel_values = x["qwen_pixel_values"].to(accelerator.device, dtype)
                pixel_values = x["pixel_values"].to(accelerator.device, dtype)
                B, L, D = qwen_pixel_values.shape
                qwen_pixel_values = rearrange(qwen_pixel_values, "B L D -> (B L) D")
                grid_thw = torch.tensor([1, 16, 16]).repeat(B, 1).to(accelerator.device)

                with torch.no_grad():
                    x_clip = qwen_clip(qwen_pixel_values, grid_thw=grid_thw)
                    x_clip = rearrange(x_clip, "(B L) D -> B L D", B=B)
                    vae_latent = vae.encode(pixel_values).latent_dist.sample().to(dtype)

                rec_latent = vae_aligner(x_clip).to(dtype)

                loss_mse = torch.nn.functional.mse_loss(rec_latent, vae_latent)

                accelerator.backward(loss_mse)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_learn, 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                    global_step += 1
                    progress_bar.update(1)

                    logs = dict(
                        loss_mse  = accelerator.gather(loss_mse.detach()).mean().item(),
                    )
                    accelerator.log(logs, step=global_step)
                    progress_bar.set_postfix(**logs)

                    if global_step > config.train.num_iter:
                        training_done = True
                        break

                    if global_step > 0 and global_step % config.train.save_every == 0 and accelerator.is_main_process:
                        vae_aligner.eval()
                        state_dict = accelerator.unwrap_model(vae_aligner).state_dict()
                        torch.save(state_dict, os.path.join(output_dir, f"vae_aligner-{config.train.exp_name}-{global_step}"))
                        accelerator.print(f"vae_aligner saved to {os.path.join(output_dir, f"vae_aligner-{config.train.exp_name}-{global_step}")}")

        epoch += 1
        accelerator.print(f"epoch {epoch}: finished")
        accelerator.log({"epoch": epoch}, step=global_step)

    accelerator.end_training()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/vae_aligner/qwen_clip.yaml")
    args = parser.parse_args()
    main(args)