import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import pprint
import argparse

from tqdm import tqdm
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration

from model.vae_aligner import get_vae_aligner
from model.dit.qwen_dit import modify_qwen_vl
from model.janus.models import MultiModalityCausalLM, VLChatProcessor
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

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


    # load models
    pad_token_id = 151643
    vae_aligner = get_vae_aligner(config.vae_aligner)
    ckpt = torch.load(config.vae_aligner.pretrained_path, map_location="cpu", weights_only=True)
    vae_aligner.load_state_dict(ckpt, strict=True)
    vae_aligner_projector = vae_aligner.siglip_feature_proj

    siglip = MultiModalityCausalLM.from_pretrained(config.janus_1b_path, trust_remote_code=True).vision_model
    qwen_vl = Qwen2_5_VLForConditionalGeneration.from_pretrained(config.qwen_vl_path)

    vae_aligner_projector.requires_grad_(False)
    siglip.requires_grad_(False)

    qwen_vl_plus, train_scheduler = modify_qwen_vl(qwen_vl, config.modify_qwen_vl)

    global_step = config.train.global_step if config.train.global_step is not None else 0
    params_to_learn = list(p for p in qwen_vl_plus.parameters() if p.requires_grad)

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

    qwen_vl_plus, optimizer = accelerator.prepare(qwen_vl_plus, optimizer)
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
        for x, y in dataloader:
            with accelerator.accumulate([qwen_vl_plus]):
                qwen_vl_plus.train()
                input_ids = y["input_ids"].to(accelerator.device)
                attention_mask = y["attention_mask"]
                pixel_values = x["pixel_values"].to(accelerator.device, dtype)
                pixel_values = pixel_values * 2 - 1

                # cfg dropout
                B, L = input_ids.shape
                mask = (torch.rand(B, 1) < config.train.cfg_drop_rate).repeat(1, L - 1)
                input_ids[:, :-1][mask] = pad_token_id

                with torch.no_grad():
                    x_siglip = siglip(pixel_values)
                    x_siglip_dimdown = vae_aligner_projector(x_siglip)
                    x_0 = x_siglip_dimdown

                text_embedding = qwen_vl_plus.get_input_embeddings()(input_ids)
                joint_embedding = torch.cat((text_embedding, qwen_vl_plus.query.unsqueeze(0).repeat(B, 1, 1)), dim=1)

                img_mask = torch.ones((B, 576), dtype=torch.bool, device=accelerator.device)
                attention_mask = torch.cat([attention_mask, img_mask], dim=1)

                hidden_states = qwen_vl_plus.model(
                    inputs_embeds        = joint_embedding,
                    attention_mask       = attention_mask,
                    output_hidden_states = True,
                ).hidden_states[-1]
                print(hidden_states.shape)
                exit(0)

    # for i, batch in enumerate(dataloader):
    #     x, y = batch
    #     print(y)
    #     break
        # if i % 100 == 0 and accelerator.is_main_process:
        #     print(x["pixel_value"].shape, y.shape, num_sample)
        # num_sample += x["pixel_value"].shape[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/qwen_fix/qwen_metaquery.yaml")
    args = parser.parse_args()
    main(args)