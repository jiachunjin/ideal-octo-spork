import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import pprint
import argparse

from tqdm import tqdm
from diffusers import AutoencoderKL
from omegaconf import OmegaConf
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
                pixel_values = x["pixel_values"].to(accelerator.device, dtype)
                # TODO do mean and std normalization for pixel_values

                with torch.no_grad():
                    x_siglip = qwen_clip(pixel_values)
                    vae_latent = vae.encode(x).latent_dist.sample().to(dtype)
                print(x_siglip.shape, vae_latent.shape)
                exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/vae_aligner/qwen_clip.yaml")
    args = parser.parse_args()
    main(args)