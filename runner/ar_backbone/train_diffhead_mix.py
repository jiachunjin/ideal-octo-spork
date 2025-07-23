import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse
import pprint
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from einops import rearrange
from diffusers import AutoencoderKL

from model.vae_aligner import get_vae_aligner
from model.janus.models import MultiModalityCausalLM, VLChatProcessor
from model.dit.diff_mlp import equip_diffhead_query_with_janus
from util.dataloader import get_dataloader
from util.misc import process_path_for_different_machine, flatten_dict

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

    # load models
    tokenizer = VLChatProcessor.from_pretrained(config.janus_1b_path).tokenizer

    vae_aligner = get_vae_aligner(config.vae_aligner)
    # ckpt = torch.load(config.vae_aligner.pretrained_path, map_location="cpu", weights_only=True)
    # vae_aligner.load_state_dict(ckpt, strict=True)
    vae_aligner_projector = vae_aligner.siglip_feature_proj

    if config.train.ar_backbone == "janus1b":
        janus = MultiModalityCausalLM.from_pretrained(config.janus_1b_path, trust_remote_code=True)
    elif config.train.ar_backbone == "janus7b":
        janus = MultiModalityCausalLM.from_pretrained(config.janus_7b_path, trust_remote_code=True)
    janus, train_scheduler = equip_diffhead_query_with_janus(janus, config)

    if config.train.diffhead_resume_path is not None:
        diffhead_ckpt = torch.load(config.train.diffhead_resume_path, map_location="cpu", weights_only=True)
        janus.diff_head.load_state_dict(diffhead_ckpt, strict=True)
        accelerator.print(f"DiffHead model loaded from {config.train.diffhead_resume_path}")
    
    if config.train.siglip16_aligner_resume_path is not None:
        siglip16_aligner_ckpt = torch.load(config.train.siglip16_aligner_resume_path, map_location="cpu", weights_only=True)
        janus.siglip16_aligner.load_state_dict(siglip16_aligner_ckpt, strict=True)
        accelerator.print(f"siglip16_aligner model loaded from {config.train.siglip16_aligner_resume_path}")
    
    if config.train.backbone_resume_path is not None:
        backbone_ckpt = torch.load(config.train.backbone_resume_path, map_location="cpu", weights_only=True)
        janus.language_model.model.load_state_dict(backbone_ckpt, strict=True)
        accelerator.print(f"Backbone model loaded from {config.train.backbone_resume_path}")

    if config.train.gen_feature == "siglip16":
        siglip = janus.vision_model
        vae_aligner_projector.requires_grad_(False)
        siglip.requires_grad_(False)
    elif config.train.gen_feature == "vae":
        vae = AutoencoderKL.from_pretrained(config.vae_path)
        vae.requires_grad_(False)

    global_step = config.train.global_step if config.train.global_step is not None else 0
    params_to_learn = list(p for p in janus.parameters() if p.requires_grad)

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

    dataloader = get_dataloader(config, accelerator, tokenizer)

    janus, dataloader, optimizer = accelerator.prepare(janus, dataloader, optimizer)

    if config.train.gen_feature == "siglip16":
        siglip = siglip.to(accelerator.device, dtype).eval()
        vae_aligner_projector = vae_aligner_projector.to(accelerator.device, dtype).eval()
    elif config.train.gen_feature == "vae":
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
        for batch in dataloader:
            if batch["pixel_values"].shape[0] == 0:
                accelerator.print("Skip empty batch")
                continue
            
            with accelerator.accumulate([janus]):
                janus.train()
                B = batch["pixel_values"].shape[0]
                idx = torch.randperm(B, device=accelerator.device)
                half = B // 2
                gen_idx = idx[:half]
                und_idx = idx[half:]                

                input_ids_gen = batch["input_ids"][gen_idx]
                input_ids_und = batch["input_ids"][und_idx]
                # print(tokenizer.decode(input_ids_und[0], skip_special_tokens=False))
                # exit(0)
                attention_mask_gen = batch["attention_mask"][gen_idx]
                attention_mask_und = batch["attention_mask"][und_idx]
                pixel_values_gen = batch["pixel_values"][gen_idx].to(dtype)
                pixel_values_und = batch["pixel_values_und"][und_idx].to(dtype)
                pixel_values_gen = pixel_values_gen * 2 - 1

                # prepare visual features
                with torch.no_grad():
                    if config.train.gen_feature == "siglip16":
                        x_siglip = siglip(pixel_values_gen)
                        x_siglip_dimdown = vae_aligner_projector(x_siglip)
                        visual_gen_feature = x_siglip_dimdown # (B_gen, 576, 16)
                        visual_und_feature = siglip(pixel_values_und) # (B_und, 576, 1024)
                    elif config.train.gen_feature == "vae":
                        raise NotImplementedError

                eos_token = torch.ones((1, 1), dtype=torch.int64, device=accelerator.device) * tokenizer.eos_token_id
                boi_token = torch.ones((1, 1), dtype=torch.int64, device=accelerator.device) * tokenizer.convert_tokens_to_ids("<begin_of_image>")
                eoi_token = torch.ones((1, 1), dtype=torch.int64, device=accelerator.device) * tokenizer.convert_tokens_to_ids("<end_of_image>")
                eos_embedding = janus.language_model.get_input_embeddings()(eos_token)
                boi_embedding = janus.language_model.get_input_embeddings()(boi_token)
                eoi_embedding = janus.language_model.get_input_embeddings()(eoi_token)

                # generation input embedding
                B_gen, L_gen = input_ids_gen.shape
                # eos_token = torch.ones((B, 1), dtype=torch.int64, device=accelerator.device) * tokenizer.eos_token_id
                # eoi_token = torch.ones((B, 1), dtype=torch.int64, device=accelerator.device) * tokenizer.convert_tokens_to_ids("<end_of_image>")
                mask = (torch.rand(B_gen, 1) < config.train.cfg_drop_rate).repeat(1, L_gen)
                input_ids_gen[mask] = tokenizer.pad_token_id

                # boi_token = torch.ones((B, 1), dtype=torch.int64, device=accelerator.device) * tokenizer.convert_tokens_to_ids("<begin_of_image>")
                # input_ids_gen = torch.cat([input_ids_gen], dim=1)
                text_embedding = janus.language_model.get_input_embeddings()(input_ids_gen)
                img_embedding_gen = janus.siglip16_aligner(visual_gen_feature)
                joint_embedding_gen = torch.cat((text_embedding, boi_embedding.repeat(B_gen, 1, 1), img_embedding_gen, eoi_embedding.repeat(B_gen, 1, 1), eos_embedding.repeat(B_gen, 1, 1)), dim=1)

                img_mask = torch.ones((B_gen, 1 + 576 + 1 + 1), dtype=torch.bool, device=accelerator.device) # boi, img, eoi, eos
                attention_mask_gen = torch.cat([attention_mask_gen, img_mask], dim=1)

                B_und, L_und = input_ids_und.shape
                text_embedding_und = janus.language_model.get_input_embeddings()(input_ids_und)
                img_embedding_und = janus.aligner(visual_und_feature)

                joint_embedding_und = torch.cat((boi_embedding.repeat(B_und, 1, 1), img_embedding_und, eoi_embedding.repeat(B_und, 1, 1), text_embedding_und, eos_embedding.repeat(B_und, 1, 1)), dim=1)

                img_mask = torch.ones((B_und, 1 + 576 + 1), dtype=torch.bool, device=accelerator.device) # boi, img, eoi
                eos_mask = torch.ones((B_und, 1), dtype=torch.bool, device=accelerator.device)
                attention_mask_und = torch.cat([img_mask, attention_mask_und, eos_mask], dim=1)

                joint_embedding = torch.cat([joint_embedding_gen, joint_embedding_und], dim=0)
                attention_mask = torch.cat([attention_mask_gen, attention_mask_und], dim=0)
                # print(joint_embedding.shape, attention_mask.shape)

                hidden_states = janus.module.language_model(
                    inputs_embeds        = joint_embedding,
                    attention_mask       = attention_mask,
                    output_hidden_states = True,
                ).hidden_states[-1]

                print(hidden_states.shape, B_gen, B_und)
                exit(0)

                # compute gen loss
                z = hidden_states[:, -576-1:-1, :]
                gt_feature = visual_gen_feature

                z = rearrange(z, "B L D -> (B L) D")
                gt_feature = rearrange(gt_feature, "B L D -> (B L) D")
                B = z.shape[0]
                timesteps = torch.randint(0, 1000, (B,), dtype=torch.int64, device=z.device)
                noise = torch.randn_like(gt_feature, device=z.device, dtype=z.dtype)
                noisy_latents = train_scheduler.add_noise(gt_feature, noise, timesteps)
                target = train_scheduler.get_velocity(gt_feature, noise, timesteps)
                pred = janus.diff_head(noisy_latents, timesteps, z)

                loss = torch.nn.functional.mse_loss(pred.to(dtype), target)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_learn, 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                    global_step += 1
                    progress_bar.update(1)

                    logs = dict(
                        query_dit_loss = accelerator.gather(loss.detach()).mean().item(),
                    )
                    accelerator.log(logs, step=global_step)
                    progress_bar.set_postfix(**logs)
                
                if global_step > 0 and global_step % config.train.save_every == 0 and accelerator.is_main_process and accelerator.sync_gradients:
                    janus.eval()
                    state_dict = accelerator.unwrap_model(janus).diff_head.state_dict()
                    save_path = os.path.join(output_dir, f"diff_head-{config.train.exp_name}-{global_step}")
                    torch.save(state_dict, save_path)
                    print(f"diff_head saved to {save_path}")

                    state_dict = accelerator.unwrap_model(janus).siglip16_aligner.state_dict()
                    save_path = os.path.join(output_dir, f"siglip16_aligner-{config.train.exp_name}-{global_step}")
                    torch.save(state_dict, save_path)
                    print(f"siglip16_aligner saved to {save_path}")

                    if config.tune_backbone:
                        state_dict = accelerator.unwrap_model(janus).language_model.model.state_dict()
                        save_path = os.path.join(output_dir, f"janus-backbone-{config.train.exp_name}-{global_step}")
                        torch.save(state_dict, save_path)
                        print(f"janus-backbone saved to {save_path}")

        epoch += 1
        accelerator.print(f"epoch {epoch}: finished")
        accelerator.log({"epoch": epoch}, step=global_step)

    accelerator.end_training()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/ar_backbone/diffhead.yaml")
    args = parser.parse_args()
    main(args)