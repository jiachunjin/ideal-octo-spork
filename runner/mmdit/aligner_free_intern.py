import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import copy
import argparse
import pprint
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from einops import rearrange

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from accelerate.state import AcceleratorState

from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3

from model.internvl.modeling_internvl_chat import InternVLChatModel
from model.mmdit import load_mmdit
from util.misc import process_pretrained_model_path, flatten_dict
from util.intern_dataloader import get_intern_dataloader


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

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
    accelerator.print(pprint.pformat(OmegaConf.to_container(config, resolve=True), indent=2, width=120).strip('{}'))
    AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = config.data.batch_size
    accelerator.print(AcceleratorState().deepspeed_plugin.deepspeed_config)

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(config.sd3_5_path, subfolder="scheduler")
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    ar_model = InternVLChatModel.from_pretrained(config.intern_vl_1b_path)
    vae = AutoencoderKL.from_pretrained(config.sd3_5_path, subfolder="vae")
    vae.requires_grad_(False)

    mmdit = load_mmdit(config)
    if config.train.resume_path is not None:
        ckpt = torch.load(config.train.resume_path, map_location="cpu", weights_only=True)
        mmdit.load_state_dict(ckpt, strict=True)
        accelerator.print(f"mmdit loaded from {config.train.resume_path}")

    global_step = config.train.global_step if config.train.global_step is not None else 0
    params_to_learn = list(p for p in mmdit.parameters() if p.requires_grad)

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

    dataloader = get_intern_dataloader(config.data, accelerator)
    mmdit, optimizer = accelerator.prepare(mmdit, optimizer)
    ar_model.requires_grad_(False)
    ar_model = ar_model.to(accelerator.device, dtype).eval()
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
    accelerator.print(f"vae_aligner dtype: {next(mmdit.parameters()).dtype}")
    accelerator.print(f"Accelerator mixed precision: {accelerator.mixed_precision}")

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    imagenet_mean = torch.tensor(IMAGENET_MEAN, device=accelerator.device, dtype=dtype).view(1, 3, 1, 1)
    imagenet_std = torch.tensor(IMAGENET_STD, device=accelerator.device, dtype=dtype).view(1, 3, 1, 1)

    while not training_done:
        for batch in dataloader:
            with accelerator.accumulate([mmdit]):
                mmdit.train()
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype)
                pixel_values_clip = (pixel_values - imagenet_mean) / imagenet_std
                pixel_values_vae = pixel_values * 2 - 1
                with torch.no_grad():
                    x_clip = ar_model.extract_feature(pixel_values_clip)
                    x_vae = vae.encode(pixel_values_vae).latent_dist.sample()
                
                model_input = (x_vae - vae.config.shift_factor) * vae.config.scaling_factor
                noise = torch.randn_like(model_input)

                u = compute_density_for_timestep_sampling(
                    weighting_scheme = "logit_normal",
                    batch_size       = model_input.shape[0],
                    logit_mean       = 0.0,
                    logit_std        = 1.0,
                    mode_scale       = 1.29,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                model_pred = mmdit(
                    x           = noisy_model_input,
                    t           = timesteps,
                    context     = mmdit.feature_down_projector(x_clip),
                    y           = None,
                )

                model_pred = model_pred * (-sigmas) + noisy_model_input
                weighting = compute_loss_weighting_for_sd3(weighting_scheme="logit_normal", sigmas=sigmas)
                target = model_input

                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_learn, 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                    progress_bar.update(1)
                    global_step += 1

                    logs = dict(
                        sd3_loss = accelerator.gather(loss.detach()).mean().item(),
                    )
                    accelerator.log(logs, step=global_step)
                    progress_bar.set_postfix(**logs)

                    if global_step > 0 and global_step % config.train.save_every == 0 and accelerator.is_main_process:
                        mmdit.eval()
                        state_dict = accelerator.unwrap_model(mmdit).state_dict()
                        save_path = os.path.join(output_dir, f"mmdit-{config.train.exp_name}-{global_step}")
                        torch.save(state_dict, save_path)
                        print(f"mmdit saved to {save_path}")

        epoch += 1
        accelerator.print(f"epoch {epoch}: finished")
        accelerator.log({"epoch": epoch}, step=global_step)

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/mmdit/aligner_free_intern.yaml")
    args = parser.parse_args()
    main(args)
    # patch_size = 2
    # depth = 24
    # num_patches = 200704
    # pos_embed_max_size = 448
    # adm_in_channels = 2048
    # qk_norm = "rms"
    # x_block_self_attn_layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    # context_embedder_config = {
    #     "target": "torch.nn.Linear",
    #     "params": {
    #         "in_features": 8,
    #         "out_features": 1536,
    #     },
    # }

    # device = torch.device("cpu")
    # dtype = torch.bfloat16

    # transformer = MMDiTX(
    #     input_size               = None,
    #     pos_embed_scaling_factor = None,
    #     pos_embed_offset         = None,
    #     pos_embed_max_size       = pos_embed_max_size,
    #     patch_size               = patch_size,
    #     in_channels              = 16,
    #     depth                    = depth,
    #     num_patches              = num_patches,
    #     adm_in_channels          = adm_in_channels,
    #     context_embedder_config  = context_embedder_config,
    #     qk_norm                  = qk_norm,
    #     x_block_self_attn_layers = x_block_self_attn_layers,
    #     device                   = device,
    #     dtype                    = dtype,
    #     verbose                  = False,
    # )

    # B = 3
    # noisy_model_input = torch.randn(B, 16, 56, 56).to(device, dtype)
    # timesteps = torch.randint(0, 1000, (B,)).to(device)
    # context = torch.randn(B, 256, 8).to(device, dtype)

    # model_pred = transformer(
    #     x           = noisy_model_input,
    #     t           = timesteps,
    #     context     = context,
    #     y           = None,
    # )

    # print(model_pred.shape)