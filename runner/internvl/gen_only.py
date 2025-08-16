import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import pprint
import argparse

from tqdm import tqdm
from einops import rearrange
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration

from util.misc import process_pretrained_model_path, flatten_dict
from model.vae_aligner import get_vae_aligner
from model.dit.diff_mlp import add_diffhead_to_ar_model
from model.internvl.modeling_internvl_chat import InternVLChatModel
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
    accelerator.print("Configuration:")
    accelerator.print(pprint.pformat(OmegaConf.to_container(config, resolve=True), indent=2, width=120).strip('{}'))
    AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = config.data.batch_size
    accelerator.print(AcceleratorState().deepspeed_plugin.deepspeed_config)

    # load models
    # vae_aligner = get_vae_aligner(config.vae_aligner)
    # ckpt = torch.load(config.vae_aligner.pretrained_path, map_location="cpu", weights_only=True)
    # vae_aligner.load_state_dict(ckpt, strict=True)
    # vae_aligner_projector = vae_aligner.siglip_feature_proj
    vision_model = InternVLChatModel.from_pretrained(config.intern_vl_8b_path).vision_model
    vision_model.requires_grad_(False)

    ar_model = InternVLChatModel.from_pretrained(config.intern_vl_2b_path)
    ar_model, train_scheduler = add_diffhead_to_ar_model(ar_model, config.model)
    # clip = ar_model.vision_model

    if config.train.diffhead_resume_path is not None:
        diffhead_ckpt = torch.load(config.train.diffhead_resume_path, map_location="cpu", weights_only=True)
        ar_model.diff_head.load_state_dict(diffhead_ckpt, strict=True)
        accelerator.print(f"diff_head loaded from {config.train.diffhead_resume_path}")
    if config.train.clip_projector_resume_path is not None:
        clip_projector_ckpt = torch.load(config.train.clip_projector_resume_path, map_location="cpu", weights_only=True)
        ar_model.clip_projector.load_state_dict(clip_projector_ckpt, strict=True)
        accelerator.print(f"clip_projector loaded from {config.train.clip_projector_resume_path}")
    if config.train.backbone_resume_path is not None:
        backbone_ckpt = torch.load(config.train.backbone_resume_path, map_location="cpu", weights_only=True)
        ar_model.language_model.model.load_state_dict(backbone_ckpt, strict=True)
        accelerator.print(f"backbone loaded from {config.train.backbone_resume_path}")

    global_step = config.train.global_step if config.train.global_step is not None else 0
    params_to_learn = list(p for p in ar_model.parameters() if p.requires_grad)

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
    ar_model, optimizer = accelerator.prepare(ar_model, optimizer)
    
    # vae_aligner_projector.requires_grad_(False)
    # vae_aligner_projector = vae_aligner_projector.to(accelerator.device, dtype).eval()
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
            with accelerator.accumulate([ar_model]):
                ar_model.train()
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype)
                input_ids = batch["input_ids"].to(accelerator.device)
                attention_mask = batch["attention_mask"].to(accelerator.device)

                pixel_values = (pixel_values - imagenet_mean) / imagenet_std

                with torch.no_grad():
                    # x_clip = ar_model.extract_feature(pixel_values)
                    # x_siglip_dimdown = vae_aligner_projector(x_clip)
                    # visual_gen_feature = x_siglip_dimdown
                    x_clip = vision_model(
                        pixel_values         = pixel_values,
                        output_hidden_states = False,
                        return_dict          = True
                    ).last_hidden_state[:, 1:, :]
                    visual_gen_feature = x_clip

                B, L = input_ids.shape
                text_embedding = ar_model.language_model.get_input_embeddings()(input_ids).clone()
                img_embedding = ar_model.clip_projector(visual_gen_feature)
                joint_embedding = torch.cat((text_embedding, img_embedding), dim=1)
                img_mask = torch.ones((B, config.data.num_img_token), dtype=torch.bool, device=accelerator.device)
                attention_mask = torch.cat([attention_mask, img_mask], dim=1)

                hidden_states = ar_model.language_model(
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
                pred = ar_model.diff_head(noisy_latents, timesteps, z)

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
                
                    if global_step > 0 and global_step % config.train.save_every == 0 and accelerator.is_main_process:
                        ar_model.eval()
                        state_dict = accelerator.unwrap_model(ar_model).diff_head.state_dict()
                        save_path = os.path.join(output_dir, f"diff_head-{config.train.exp_name}-{global_step}")
                        torch.save(state_dict, save_path)
                        print(f"diff_head saved to {save_path}")

                        state_dict = accelerator.unwrap_model(ar_model).clip_projector.state_dict()
                        save_path = os.path.join(output_dir, f"clip_projector-{config.train.exp_name}-{global_step}")
                        torch.save(state_dict, save_path)
                        print(f"clip_projector saved to {save_path}")

                        if config.model.tune_backbone:
                            state_dict = accelerator.unwrap_model(ar_model).language_model.model.state_dict()
                            save_path = os.path.join(output_dir, f"backbone-{config.train.exp_name}-{global_step}")
                            torch.save(state_dict, save_path)
                            print(f"backbone saved to {save_path}")
                    accelerator.wait_for_everyone()

        epoch += 1
        accelerator.print(f"epoch {epoch}: finished")
        accelerator.log({"epoch": epoch}, step=global_step)

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/internvl/gen_only.yaml")
    args = parser.parse_args()
    main(args)