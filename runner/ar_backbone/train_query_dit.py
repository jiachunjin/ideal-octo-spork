import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse
import pprint
import torch
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from tqdm import tqdm

from model.vae_aligner import get_vae_aligner
from model.janus.models import MultiModalityCausalLM, VLChatProcessor
from model.dit.query_dit import equip_dit_query_with_janus
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
    tokenizer = VLChatProcessor.from_pretrained(config.janus_path).tokenizer

    vae_aligner = get_vae_aligner(config.vae_aligner)
    ckpt = torch.load(config.vae_aligner.pretrained_path, map_location="cpu", weights_only=True)
    vae_aligner.load_state_dict(ckpt, strict=True)
    vae_aligner_projector = vae_aligner.siglip_feature_proj
    
    janus = MultiModalityCausalLM.from_pretrained(config.janus_path, trust_remote_code=True)
    janus, train_scheduler = equip_dit_query_with_janus(janus, config)

    if config.train.dit_resume_path is not None:
        dit_ckpt = torch.load(config.train.dit_resume_path, map_location="cpu")
        janus.query_dit.load_state_dict(dit_ckpt, strict=True)
        accelerator.print(f"DiT model loaded from {config.train.dit_resume_path}")
    
    if config.train.query_resume_path is not None:
        query_ckpt = torch.load(config.train.query_resume_path, map_location="cpu")
        janus.query.data.copy_(query_ckpt["query"])
        accelerator.print(f"Query model loaded from {config.train.query_resume_path}")

    siglip = janus.vision_model

    vae_aligner_projector.requires_grad_(False)
    siglip.requires_grad_(False)

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

    dataloader = get_dataloader(config.data, accelerator, tokenizer)

    janus, dataloader, optimizer = accelerator.prepare(janus, dataloader, optimizer)
    siglip = siglip.to(accelerator.device, dtype).eval()
    vae_aligner_projector = vae_aligner_projector.to(accelerator.device, dtype).eval()

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
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                pixel_values = batch["pixel_values"].to(dtype)
                pixel_values = pixel_values * 2 - 1

                with torch.no_grad():
                    x_siglip = siglip(pixel_values)
                    x_siglip_dimdown = vae_aligner_projector(x_siglip)
                
                # cfg dropout
                B, L = input_ids.shape
                mask = (torch.rand(B, 1) < config.train.cfg_drop_rate).repeat(1, L)
                input_ids[mask] = tokenizer.pad_token_id

                boi_token = torch.ones((B, 1), dtype=torch.int64, device=accelerator.device) * tokenizer.convert_tokens_to_ids("<begin_of_image>")
                input_ids = torch.cat([input_ids, boi_token], dim=1)

                text_embedding = janus.language_model.get_input_embeddings()(input_ids)
                joint_embedding = torch.cat((text_embedding, janus.query.unsqueeze(0).repeat(B, 1, 1)), dim=1)

                img_mask = torch.ones((B, 1 + 576), dtype=torch.bool, device=accelerator.device)
                attention_mask = torch.cat([attention_mask, img_mask], dim=1)

                hidden_states = janus.language_model(
                    inputs_embeds        = joint_embedding,
                    attention_mask       = attention_mask,
                    output_hidden_states = True,
                ).hidden_states[-1]
                z = hidden_states[:, -576:, :] # use the hidden states of the query tokens, not the boi token

                # diffusion training
                timesteps = torch.randint(0, 1000, (B,), dtype=torch.int64, device=accelerator.device)
                noise = torch.randn_like(x_siglip_dimdown, device=accelerator.device, dtype=z.dtype)
                noisy_latents = train_scheduler.add_noise(x_siglip_dimdown, noise, timesteps)
                target = train_scheduler.get_velocity(x_siglip_dimdown, noise, timesteps)
                pred = janus.query_dit(noisy_latents, z, timesteps)
                loss = torch.nn.functional.mse_loss(pred, target)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    optimizer.zero_grad()
                    accelerator.clip_grad_norm_(params_to_learn, 1.0)
                    optimizer.step()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)

                logs = dict(
                    query_dit_loss  = accelerator.gather(loss.detach()).mean().item(),
                )
                accelerator.log(logs, step=global_step)
                progress_bar.set_postfix(**logs)

            if global_step > 0 and global_step % config.train.save_every == 0 and accelerator.is_main_process:
                janus.eval()
                state_dict = accelerator.unwrap_model(janus).query_dit.state_dict()
                save_path = os.path.join(output_dir, f"query_dit-{config.train.exp_name}-{global_step}")
                torch.save(state_dict, save_path)
                print(f"DiT model saved to {save_path}")

                state_dict = accelerator.unwrap_model(janus).query.detach().cpu()
                save_path = os.path.join(output_dir, f"query-{config.train.exp_name}-{global_step}")
                torch.save({"query": state_dict}, save_path)
                print(f"Query saved to {save_path}")

        epoch += 1
        accelerator.print(f"epoch {epoch}: finished")
        accelerator.log({"epoch": epoch}, step=global_step)

    accelerator.end_training()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/ar_backbone/query.yaml")
    args = parser.parse_args()
    main(args)