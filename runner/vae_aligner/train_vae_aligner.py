import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import pprint
import argparse
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from tqdm import tqdm
from diffusers import AutoencoderKL

from model.vae_aligner import get_vae_aligner
from model.janus.models import MultiModalityCausalLM
from util.dataloader import get_dataloader
from util.misc import process_path_for_different_machine


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
    config = process_path_for_different_machine(config)
    accelerator, output_dir = get_accelerator(config.train)
    accelerator.print("Configuration:")
    accelerator.print(pprint.pformat(OmegaConf.to_container(config, resolve=True), indent=2, width=120).strip('{}'))

    vae_aligner = get_vae_aligner(config.vae_aligner)
    siglip = MultiModalityCausalLM.from_pretrained(config.janus_path, trust_remote_code=True).vision_model
    # vae = AutoencoderKL.from_pretrained(config.vae_path)

    siglip.requires_grad_(False)
    # vae.requires_grad_(False)

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
    
    dataloader = get_dataloader(config.data)

    vae_aligner, dataloader, optimizer = accelerator.prepare(vae_aligner, dataloader, optimizer)

    training_done = False
    epoch = 0
    progress_bar = tqdm(
        total   = config.train.num_iter,
        initial = global_step,
        desc    = "Steps",
        disable = not accelerator.is_local_main_process,
    )

    while not training_done:
        for batch in dataloader:
            pixel_values = batch["pixel_values"]
            texts = batch["texts"]

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/vae_aligner/siglip_flux.yaml")
    args = parser.parse_args()
    main(args)