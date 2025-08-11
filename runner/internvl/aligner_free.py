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
from model.vae_aligner.vit_vae_aligner import get_feature_down_proj
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

    if config.base_model == "intern_vl_1b":
        ar_model = InternVLChatModel.from_pretrained(config.intern_vl_1b_path)
    elif config.base_model == "intern_vl_2b":
        ar_model = InternVLChatModel.from_pretrained(config.intern_vl_2b_path)
    elif config.base_model == "intern_vl_8b":
        ar_model = InternVLChatModel.from_pretrained(config.intern_vl_8b_path)
    else:
        raise ValueError(f"Invalid base model: {config.base_model}")
    
    ar_model, train_scheduler = add_diffhead_to_ar_model(ar_model, config.model)

    feature_down_projector = get_feature_down_proj(config.feature_down_projector)
    feature_down_projector_ckpt = torch.load(config.feature_down_projector.ckpt_path, map_location="cpu", weights_only=True)
    m, u = feature_down_projector.load_state_dict(feature_down_projector_ckpt, strict=False)
    print(f"missing keys: {m}")
    print(f"unexpected keys: {u}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/internvl/aligner_free.yaml")
    args = parser.parse_args()
    main(args)