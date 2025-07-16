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
from safetensors.torch import load_file
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from model.janus.models import VLChatProcessor, MultiModalityCausalLM
from model.mmdit.mmditx import MMDiTX
from model.vae_aligner import get_vae_aligner
from util.misc import process_path_for_different_machine, flatten_dict
from util.dataloader import get_dataloader


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

@torch.no_grad()
def sample_sd3_5(
    transformer,
    vae,
    noise_scheduler,
    device,
    dtype, 
    context,
    batch_size          = 1,
    height              = 192,
    width               = 192,
    num_inference_steps = 20,
    guidance_scale      = 1.0,
    seed                = None
):
    if seed is not None:
        torch.manual_seed(seed)
    
    transformer.eval()
    
    latent_height = height // 8
    latent_width = width // 8
    
    latents = torch.randn(
        (batch_size, 16, latent_height, latent_width),
        device = device,
        dtype  = dtype
    )

    noise_scheduler.set_timesteps(num_inference_steps)
    timesteps = noise_scheduler.timesteps.to(device=device, dtype=dtype)
    
    for i, t in enumerate(tqdm(timesteps, desc="Sampling")):
        if t.ndim == 0:
            t = t.unsqueeze(0)
        t = t.repeat(batch_size)

        latent_model_input = latents

        if guidance_scale > 1.0:
            latent_model_input = torch.cat([latent_model_input, latent_model_input], dim=0)
            t = torch.cat([t, t], dim=0)
            context_ = torch.cat([context, torch.zeros_like(context, device=context.device, dtype=context.dtype)], dim=0)
        else:
            context_ = context

        noise_pred = transformer(
            x           = latent_model_input,
            t           = t,
            context     = context_,
            y           = None,
        )

        if guidance_scale > 1.0:
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        step_output = noise_scheduler.step(
            model_output=noise_pred,
            timestep=t[0] if t.ndim > 0 else t,
            sample=latents,
            return_dict=False,
        )
        latents = step_output[0]
    
    latents = 1 / vae.config.scaling_factor * latents + vae.config.shift_factor
    image = vae.decode(latents).sample
    
    image = (image / 2 + 0.5).clamp(0, 1)
    
    return image

def load_pretrained_mmdit(ckpt_path):
    patch_size = 2
    depth = 24
    num_patches = 147456
    pos_embed_max_size = 384
    adm_in_channels = 2048
    qk_norm = "rms"
    x_block_self_attn_layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    context_embedder_config = {
        "target": "torch.nn.Linear",
        "params": {
            "in_features": 256,
            "out_features": 1536,
        },
    }

    device = torch.device("cpu")
    dtype = torch.bfloat16

    transformer = MMDiTX(
        input_size               = None,
        pos_embed_scaling_factor = None,
        pos_embed_offset         = None,
        pos_embed_max_size       = pos_embed_max_size,
        patch_size               = patch_size,
        in_channels              = 16,
        depth                    = depth,
        num_patches              = num_patches,
        adm_in_channels          = adm_in_channels,
        context_embedder_config  = context_embedder_config,
        qk_norm                  = qk_norm,
        x_block_self_attn_layers = x_block_self_attn_layers,
        device                   = device,
        dtype                    = dtype,
        verbose                  = False,
    )

    ckpt = load_file(os.path.join(ckpt_path, "sd3.5_medium.safetensors"))
    new_ckpt = {}
    prefix = "model.diffusion_model."
    for k, v in ckpt.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):]
            new_ckpt[new_key] = v
    del new_ckpt["context_embedder.weight"]
    m, u = transformer.load_state_dict(new_ckpt, strict=False)
    print(f"missing keys: {m}")
    print(f"unexpected keys: {u}")

    return transformer


def main(args):
    config = OmegaConf.load(args.config)
    config = process_path_for_different_machine(config)
    accelerator, output_dir = get_accelerator(config.train)
    accelerator.print(pprint.pformat(OmegaConf.to_container(config, resolve=True), indent=2, width=120).strip('{}'))

    tokenizer = VLChatProcessor.from_pretrained(config.janus_1b_path).tokenizer

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(config.sd3_5_path, subfolder="scheduler")
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    vae = AutoencoderKL.from_pretrained(config.sd3_5_path, subfolder="vae")
    vae.requires_grad_(False)

    siglip = MultiModalityCausalLM.from_pretrained(config.janus_1b_path, trust_remote_code=True).vision_model
    siglip.requires_grad_(False)

    vae_aligner = get_vae_aligner(config.vae_aligner)
    ckpt_vae_aligner = torch.load(config.vae_aligner.pretrained_path, map_location="cpu", weights_only=True)
    vae_aligner.load_state_dict(ckpt_vae_aligner, strict=True)
    vae_aligner.requires_grad_(False)
    
    transformer = load_pretrained_mmdit(config.sd3_5_path)
    if config.train.resume_path is not None:
        ckpt = torch.load(config.train.resume_path, map_location="cpu", weights_only=True)
        transformer.load_state_dict(ckpt, strict=True)
        accelerator.print(f"Transformer loaded from {config.train.resume_path}")


    global_step = config.train.global_step if config.train.global_step is not None else 0
    params_to_learn = list(p for p in transformer.parameters() if p.requires_grad)

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

    dataloader = get_dataloader(config.data, accelerator, tokenizer)

    transformer, dataloader, optimizer = accelerator.prepare(transformer, dataloader, optimizer)
    vae = vae.to(accelerator.device, dtype).eval()
    vae_aligner = vae_aligner.to(accelerator.device, dtype).eval()
    siglip = siglip.to(accelerator.device, dtype).eval()

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
    accelerator.print(f"transformer dtype: {next(transformer.parameters()).dtype}")
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

    while not training_done:
        for batch in dataloader:
            if batch["pixel_values"].shape[0] == 0:
                accelerator.print("Skip empty batch")
                continue
        
            with accelerator.accumulate(transformer):
                transformer.train()
                pixel_values = batch["pixel_values"].to(dtype)
                pixel_values = pixel_values * 2 - 1
                x_siglip = siglip(pixel_values)
                x_coarse_reference = vae_aligner(x_siglip)
                context = rearrange(x_coarse_reference, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=4, p2=4)
                if getattr(config.train, "add_noise_to_context", False):
                    context += torch.randn_like(context, device=context.device, dtype=context.dtype) * config.train.context_noise_std
                # add cfg dropout with p = config.train.cfg_drop_rate
                B, L, D = context.shape
                mask = (torch.rand(B, 1, 1) < config.train.cfg_drop_rate).repeat(1, L, D)
                context[mask] = 0

                x_vae = vae.encode(pixel_values).latent_dist.sample()
                model_input = (x_vae - vae.config.shift_factor) * vae.config.scaling_factor

                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                u = compute_density_for_timestep_sampling(
                    weighting_scheme = "logit_normal",
                    batch_size       = bsz,
                    logit_mean       = 0.0,
                    logit_std        = 1.0,
                    mode_scale       = 1.29,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                model_pred = transformer(
                    x           = noisy_model_input,
                    t           = timesteps,
                    context     = context,
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

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                logs = dict(
                    sd3_loss = accelerator.gather(loss.detach()).mean().item(),
                )
                accelerator.log(logs, step=global_step)
                progress_bar.set_postfix(**logs)

            if global_step > 0 and global_step % config.train.save_every == 0 and accelerator.is_main_process:
                transformer.eval()
                state_dict = accelerator.unwrap_model(transformer).state_dict()
                save_path = os.path.join(output_dir, f"transformer-{config.train.exp_name}-{global_step}")
                torch.save(state_dict, save_path)
                print(f"Transformer saved to {save_path}")

            if global_step > 0 and global_step % config.train.val_every == 0 and accelerator.is_main_process:
                transformer.eval()

                import torchvision.transforms as pth_transforms
                from PIL import Image


                vae_transform = pth_transforms.Compose([
                    pth_transforms.Resize(384, max_size=None),
                    pth_transforms.CenterCrop(384),
                    pth_transforms.ToTensor(),
                ])

                img_1 = Image.open("/data/phd/jinjiachun/codebase/connector/asset/kobe.png").convert("RGB")
                img_2 = Image.open("/data/phd/jinjiachun/codebase/connector/asset/004.jpg").convert("RGB")

                x_1 = vae_transform(img_1).unsqueeze(0).to(accelerator.device, dtype)
                x_2 = vae_transform(img_2).unsqueeze(0).to(accelerator.device, dtype)
                x = torch.cat([x_1, x_2], dim=0)
                x = x * 2 - 1

                x_siglip = siglip(x)
                x_coarse_reference = vae_aligner(x_siglip)
                context = rearrange(x_coarse_reference, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=4, p2=4)

                samples = sample_sd3_5(
                    transformer         = transformer,
                    vae                 = vae,
                    noise_scheduler     = noise_scheduler,
                    device              = accelerator.device,
                    dtype               = dtype,
                    context             = context,
                    batch_size          = context.shape[0],
                    height              = 384,
                    width               = 384,
                    num_inference_steps = 20,
                    guidance_scale      = 5.0,
                    seed                = 42
                )

                import torchvision.utils as vutils
                sample_path = f"samples_step_{global_step}.png"
                vutils.save_image(samples, sample_path, nrow=2, normalize=False)
                accelerator.print(f"Samples saved to {sample_path}")

                with torch.no_grad():
                    reconstructed = vae.decode(x_coarse_reference).sample
                    reconstructed = reconstructed.to(torch.float32)
                    reconstructed = (reconstructed + 1) / 2
                    reconstructed = torch.clamp(reconstructed, 0, 1)
                    vutils.save_image(reconstructed[:4], f"coarse_step_{global_step}.png", nrow=2, normalize=False)

        epoch += 1
        accelerator.print(f"epoch {epoch}: finished")
        accelerator.log({"epoch": epoch}, step=global_step)

    accelerator.end_training()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/mmdit/basic_sd3.yaml")
    args = parser.parse_args()
    main(args)