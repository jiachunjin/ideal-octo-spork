import os
import torch
import pprint
import argparse

from tqdm import tqdm
from einops import rearrange
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

from model.vae_aligner import get_vae_aligner
from model.janus.models import MultiModalityCausalLM, VLChatProcessor
from model.dit.diff_mlp import equip_diffhead_query_with_janus
from util.dual_dataloader import get_dataloader_und, get_dataloader_gen, InfiniteIterator
from util.misc import process_pretrained_model_path, flatten_dict


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

    # load models
    tokenizer = VLChatProcessor.from_pretrained(config.janus_1b_path).tokenizer
    vae_aligner = get_vae_aligner(config.vae_aligner)
    ckpt = torch.load(config.vae_aligner.pretrained_path, map_location="cpu", weights_only=True)
    vae_aligner.load_state_dict(ckpt, strict=True)
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

    dataloader_und = get_dataloader_und(config)
    dataloader_gen = get_dataloader_gen(config)

    janus, optimizer, dataloader_und, dataloader_gen = accelerator.prepare(janus, optimizer, dataloader_und, dataloader_gen)

    siglip = siglip.to(accelerator.device, dtype).eval()
    vae_aligner_projector = vae_aligner_projector.to(accelerator.device, dtype).eval()

    inf_iter_und = InfiniteIterator(dataloader_und, "understanding")
    inf_iter_gen = InfiniteIterator(dataloader_gen, "generation")
    training_done = False

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
    accelerator.print(f"siglip dtype: {next(siglip.parameters()).dtype}")
    accelerator.print(f"janus dtype: {next(janus.parameters()).dtype}")
    accelerator.print(f"Accelerator mixed precision: {accelerator.mixed_precision}")

    while not training_done:
        with accelerator.accumulate([janus]):
            janus.train()

            # load und data and gen data
            batch_gen = next(inf_iter_gen)
            pixel_value_gen = batch_gen["pixel_values"].to(dtype)
            pixel_value_gen = pixel_value_gen * 2 - 1
            input_ids_gen = batch_gen["input_ids"]
            attention_mask_gen = batch_gen["attention_mask"]
            # accelerator.print(f"pixel_value_gen shape: {pixel_value_gen.shape}")
            # accelerator.print(f"input_ids_gen shape: {input_ids_gen.shape}")
            # accelerator.print(f"attention_mask_gen shape: {attention_mask_gen.shape}")

            batch_und = next(inf_iter_und)
            pixel_values_und = batch_und["pixel_values"].to(dtype)
            input_ids_und = batch_und["input_ids"]
            attention_mask_und = batch_und["attention_mask"]
            labels_und = batch_und["labels"][:, 1:].contiguous()
            # accelerator.print(f"pixel_values_und shape: {pixel_values_und.shape}")
            # accelerator.print(f"input_ids_und shape: {input_ids_und.shape}")
            # accelerator.print(f"attention_mask_und shape: {attention_mask_und.shape}")
            # accelerator.print(f"labels_und shape: {labels_und.shape}")

            if pixel_value_gen.shape[0] == 0 or pixel_values_und.shape[0] == 0:
                continue

            with torch.no_grad():
                x_siglip = siglip(pixel_value_gen)
                visual_gen_feature = vae_aligner_projector(x_siglip)
                visual_und_feature = siglip(pixel_values_und)

            # generation input embedding
            B_gen, L_gen = input_ids_gen.shape
            mask = (torch.rand(B_gen, 1) < config.train.cfg_drop_rate).repeat(1, L_gen)
            input_ids_gen[mask] = tokenizer.pad_token_id

            text_embedding_gen = janus.language_model.get_input_embeddings()(input_ids_gen)
            img_embedding_gen = janus.siglip16_aligner(visual_gen_feature)
            joint_embedding_gen = torch.cat([text_embedding_gen, img_embedding_gen], dim=1)

            img_mask_gen = torch.ones((B_gen, 576), dtype=torch.bool, device=accelerator.device)
            attention_mask_gen = torch.cat([attention_mask_gen, img_mask_gen], dim=1)

            # accelerator.print(f"joint_embedding_gen shape: {joint_embedding_gen.shape}")
            # accelerator.print(f"attention_mask_gen shape: {attention_mask_gen.shape}")

            # understanding input embedding
            text_embedding_und = janus.language_model.get_input_embeddings()(input_ids_und)
            img_embedding_und = janus.aligner(visual_und_feature)
            text_embedding_und[:, 42:618, :] = img_embedding_und # replace img_place_holder with img_embedding_und
            joint_embedding_und = text_embedding_und

            # accelerator.print(f"text_embedding_und shape: {text_embedding_und.shape}")
            # accelerator.print(f"img_embedding_und shape: {img_embedding_und.shape}")
            # accelerator.print(f"attention_mask_und shape: {attention_mask_und.shape}")
            # accelerator.print(f"labels_und shape: {labels_und.shape}")

            joint_embedding = torch.cat([joint_embedding_gen, joint_embedding_und], dim=0)
            attention_mask = torch.cat([attention_mask_gen, attention_mask_und], dim=0)

            hidden_states = janus.module.language_model(
                inputs_embeds        = joint_embedding,
                attention_mask       = attention_mask,
                output_hidden_states = True,
            ).hidden_states[-1]

            # accelerator.print(f"hidden_states shape: {hidden_states.shape}")

            # ---------- compute gen loss ----------
            hidden_states_gen = hidden_states[:B_gen]
            z_gen = hidden_states_gen[:, -576-1:-1, :]
            gt_feature = visual_gen_feature
            z_gen = rearrange(z_gen, "B L D -> (B L) D")
            gt_feature = rearrange(gt_feature, "B L D -> (B L) D")
            B = z_gen.shape[0]
            timesteps = torch.randint(0, 1000, (B,), dtype=torch.int64, device=z_gen.device)
            noise = torch.randn_like(gt_feature, device=z_gen.device, dtype=z_gen.dtype)
            noisy_latents = train_scheduler.add_noise(gt_feature, noise, timesteps)
            target = train_scheduler.get_velocity(gt_feature, noise, timesteps)
            pred = janus.diff_head(noisy_latents, timesteps, z_gen)

            loss_gen = torch.nn.functional.mse_loss(pred.to(dtype), target)

            # ---------- compute und loss ----------
            hidden_states_und = hidden_states[B_gen:, :-1, :].contiguous()
            # z_und = hidden_states_und[:, 1 + 576:-1, :] # skip boi, img
            # 使用 z_und 进行下一个 token 的预测
            logits = janus.language_model.lm_head(hidden_states_und)
            # accelerator.print(f"logits shape: {logits.shape}")
            # accelerator.print(f"labels_und shape: {labels_und.shape}")
            # print(logits.shape, input_ids_und.shape)
            # exit(0)
            # 计算下一个 token 的预测损失（交叉熵损失）
            loss_und = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels_und.view(-1),
                ignore_index=-100
            )

            loss = 1 * loss_gen + 0.5 * loss_und

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(params_to_learn, 1.0)
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1
                progress_bar.update(1)

                logs = dict(
                    # loss_gen_und = accelerator.gather(loss.detach()).mean().item(),
                    loss_gen     = accelerator.gather(loss_gen.detach()).mean().item(),
                    loss_und     = accelerator.gather(loss_und.detach()).mean().item(),
                )
                accelerator.log(logs, step=global_step)
                progress_bar.set_postfix(**logs)

                if global_step > config.train.num_iter:
                    training_done = True
                    break

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
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/ar_backbone/hybrid_training.yaml")
    args = parser.parse_args()
    main(args)