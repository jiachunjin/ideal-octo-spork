import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import argparse
import copy
from tqdm import tqdm
from omegaconf import OmegaConf
from einops import rearrange

from model.dit.diff_mlp import intern_add_diffhead_mmdit
from model.internvl.modeling_internvl_chat import InternVLChatModel
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from model.vae_aligner import get_vae_aligner
from model.internvl import extract_feature_pre_adapter

from util.misc import process_pretrained_model_path, flatten_dict
from util.my_tool_box import get_accelerator, get_t2i_dataloader

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def main(args):
    config = OmegaConf.load(args.config)
    config = process_pretrained_model_path(config)
    accelerator, output_dir = get_accelerator(config)

    internvl = InternVLChatModel.from_pretrained(config.model.internvl_path)
    internvl, train_scheduler = intern_add_diffhead_mmdit(internvl, config.model)
    if config.train.resume_path is not None:
        ckpt = torch.load(config.train.resume_path, map_location="cpu", weights_only=True)
        if hasattr(config.train, "skipped_keys"):
            ckpt = {k: v for k, v in ckpt.items() if k not in config.train.skipped_keys}
        m, u = internvl.load_state_dict(ckpt, strict=False)
        accelerator.print(f"Missing modules: {m}, unmatched modules: {u}")

    vae_aligner = get_vae_aligner(config.vae_aligner)
    ckpt = torch.load(config.vae_aligner.ckpt_path, map_location="cpu", weights_only=True)
    vae_aligner.load_state_dict(ckpt, strict=True)
    feature_down_projector = vae_aligner.siglip_feature_proj

    vision_model = InternVLChatModel.from_pretrained(config.intern_vl_8b_path).vision_model

    vae = AutoencoderKL.from_pretrained(config.sd3_5_path, subfolder="vae")
    vae.requires_grad_(False)

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
    feature_down_projector.requires_grad_(False)
    feature_down_projector = feature_down_projector.to(accelerator.device, dtype).eval()
    vision_model.requires_grad_(False)
    vision_model = vision_model.to(accelerator.device, dtype).eval()
    vae = vae.to(accelerator.device, dtype).eval()

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(config.sd3_5_path, subfolder="scheduler")
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

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
            with accelerator.accumulate([internvl]):
                internvl.train()
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype)
                input_ids = batch["input_ids"].to(accelerator.device)
                attention_mask = batch["attention_mask"].to(accelerator.device)

                x_intern = (pixel_values - imagenet_mean) / imagenet_std
                x_vae = pixel_values * 2 - 1

                with torch.no_grad():
                    x_clip = extract_feature_pre_adapter(vision_model, x_intern)
                    visual_gen_feature = feature_down_projector(x_clip)
                    vae_latent = vae.encode(x_vae).latent_dist.sample().to(dtype)

                # ----- compute AR loss -----
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

                loss_ar = torch.nn.functional.mse_loss(pred.to(dtype), target)

                # ----- compute DiT loss -----
                if getattr(config.model.mmdit, "train_it", False):
                    model_input = (vae_latent - vae.config.shift_factor) * vae.config.scaling_factor
                    noise = torch.randn_like(model_input, device=model_input.device, dtype=model_input.dtype)
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

                    model_pred = internvl.mmdit(
                        x           = noisy_model_input,
                        t           = timesteps,
                        context     = hidden_states[:, :-1, :],
                        y           = None,
                        multi_modal_context = True,
                    )
                    model_pred = model_pred * (-sigmas) + noisy_model_input
                    weighting = compute_loss_weighting_for_sd3(weighting_scheme="logit_normal", sigmas=sigmas)
                    target = model_input

                    loss_dit = torch.mean(
                        (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                        1,
                    ).mean()
                else:
                    loss_dit = 0.0

                loss = loss_ar + loss_dit

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_learn, 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                    global_step += 1
                    progress_bar.update(1)

                    logs = dict(
                        ar_loss = accelerator.gather(loss_ar.detach()).mean().item(),
                        dit_loss = accelerator.gather(loss_dit.detach()).mean().item(),
                    )
                    accelerator.log(logs, step=global_step)
                    progress_bar.set_postfix(**logs)

                    if global_step > 0 and global_step % config.train.save_every == 0 and accelerator.is_main_process:
                        internvl.eval()
                        state_dict = accelerator.unwrap_model(internvl).state_dict()
                        save_path = os.path.join(output_dir, f"internvl-{config.train.exp_name}-{global_step}")
                        torch.save(state_dict, save_path)
                        print(f"internvl saved to {save_path}")

                    accelerator.wait_for_everyone()


        epoch += 1
        accelerator.print(f"epoch {epoch}: finished")
        accelerator.log({"epoch": epoch}, step=global_step)

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/sunshine/coarse_fine.yaml")
    args = parser.parse_args()
    main(args)