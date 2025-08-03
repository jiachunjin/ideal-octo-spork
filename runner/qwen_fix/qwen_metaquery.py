import os
import torch
import pprint
import argparse

from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

from model.vae_aligner import get_vae_aligner
from model.dit.qwen_dit import modify_qwen_vl
from model.janus.models import MultiModalityCausalLM, VLChatProcessor
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

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
    pad_token_id = 151643
    vae_aligner = get_vae_aligner(config.vae_aligner)
    ckpt = torch.load(config.vae_aligner.pretrained_path, map_location="cpu", weights_only=True)
    vae_aligner.load_state_dict(ckpt, strict=True)
    vae_aligner_projector = vae_aligner.siglip_feature_proj

    siglip = MultiModalityCausalLM.from_pretrained(config.janus_1b_path, trust_remote_code=True).vision_model
    qwen_vl = Qwen2_5_VLForConditionalGeneration.from_pretrained(config.qwen_vl_path)

    vae_aligner_projector.requires_grad_(False)
    siglip.requires_grad_(False)

    qwen_vl_plus, train_scheduler = modify_qwen_vl(qwen_vl, config)

    print(qwen_vl_plus)
