import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import copy
import torch
import pprint
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

from sd3_5.mmditx import MMDiTX
from model.janus.models import VLChatProcessor
from util.dataloader import get_dataloader
from util.misc import process_path_for_different_machine, flatten_dict

def construct_mmdit():
    pos_embed_max_size = 384
    patch_size = 2
    depth = 8
    num_patches = 147456
    adm_in_channels = 16 # y dim
    hidden_size = 512

    context_embedder_config = {
        "target": "torch.nn.Linear",
        "params": {
            "in_features": adm_in_channels, # context dim
            "out_features": hidden_size,
        },
    }

    device = torch.device("cpu")
    dtype = torch.float32

    mmdit = MMDiTX(
        input_size               = None,
        pos_embed_scaling_factor = None,
        pos_embed_offset         = None,
        pos_embed_max_size       = pos_embed_max_size,
        patch_size               = patch_size,
        in_channels              = 16,
        depth                    = depth,
        hidden_size              = hidden_size,
        num_patches              = num_patches,
        adm_in_channels          = adm_in_channels,
        context_embedder_config  = context_embedder_config,
        qk_norm                  = "ln",
        x_block_self_attn_layers = [0, 1, 2, 3, 4, 5, 6, 7],
        device                   = device,
        dtype                    = dtype,
        verbose                  = False,
    )

    return mmdit

noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    "/data1/ckpts/stabilityai/stable-diffusion-3.5-medium", subfolder="scheduler"
)
noise_scheduler_copy = copy.deepcopy(noise_scheduler)

def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler_copy.sigmas
    schedule_timesteps = noise_scheduler_copy.timesteps
    timesteps = timesteps
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma

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

def main():
    # hyperparameters
    weighting_scheme = "logit_normal"
    logit_mean = 0.0
    logit_std = 1.0
    mode_scale = 1.29

    config = OmegaConf.load("config/mmdit/basic_sd3.yaml")
    config = process_path_for_different_machine(config)
    accelerator, output_dir = get_accelerator(config.train)
    accelerator.print("Configuration:")
    accelerator.print(pprint.pformat(OmegaConf.to_container(config, resolve=True), indent=2, width=120).strip('{}'))

    mmdit = construct_mmdit()
    tokenizer = VLChatProcessor.from_pretrained(config.janus_path).tokenizer
    vae = AutoencoderKL.from_pretrained("/data1/ckpts/black-forest-labs/FLUX.1-dev/vae")
    vae.requires_grad_(False)

    dataloader = get_dataloader(config.data, accelerator, tokenizer)

    print("done")





if __name__ == "__main__":
    main()
