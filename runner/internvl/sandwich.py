import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from transformers import Qwen2Model
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from model.internvl.modeling_internvl_chat import InternVLChatModel
from diffusers import DDPMScheduler

def add_hat_to_intern(
    model: Qwen2Model,
    num_hat: int,
):
    config = model.config
    print(config)
    new_layers = []

    current_num_layers = len(model.layers)
    for i in range(num_hat):
        layer_idx = current_num_layers + i
        new_layer = Qwen2DecoderLayer(config, layer_idx)
        new_layers.append(new_layer)
    
    model.layers.extend(new_layers)
    model.config.num_hidden_layers = len(model.layers)
    model.config.layer_types.extend(["full_attention"] * num_hat)

    return model

def equip_internvl(ar_model, config):
    ar_model.language_model.model = add_hat_to_intern(ar_model.language_model.model, config.num_hat)
    current_num_layers = len(ar_model.language_model.model.layers)
    new_layer_indices = range(current_num_layers - config.num_hat, current_num_layers)

    ar_model.requires_grad_(False)
    for idx in new_layer_indices:
        layer = ar_model.language_model.model.layers[idx]
        layer.requires_grad_(True)

    # diff_head = SimpleMLPAdaLN(
    #     in_channels    = config.diffhead.x_dim,
    #     model_channels = config.diffhead.hidden_size,
    #     out_channels   = config.diffhead.x_dim,
    #     z_channels     = config.diffhead.z_dim,
    #     num_res_blocks = config.diffhead.depth,
    # )
    train_scheduler = DDPMScheduler(
        beta_schedule          = "scaled_linear",
        beta_start             = 0.00085,
        beta_end               = 0.012,
        num_train_timesteps    = 1000,
        clip_sample            = False,
        prediction_type        = "v_prediction",
        steps_offset           = 1,
        trained_betas          = None,
        timestep_spacing       = "trailing",
        rescale_betas_zero_snr = True
    )
    
    return ar_model, train_scheduler

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
# def dev():
#     device = torch.device("cuda:0")
#     intern_vl_1b_path = "/data/phd/jinjiachun/ckpt/OpenGVLab/InternVL3-1B"

#     ar_model = InternVLChatModel.from_pretrained(intern_vl_1b_path)
#     ar_model = equip_internvl(ar_model, 8)

#     print(ar_model)
import torch
import pprint
import argparse
from tqdm import tqdm
from einops import rearrange
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration

from util.misc import process_pretrained_model_path


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

    ar_model, train_scheduler = equip_internvl(ar_model, config.model)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/internvl/sandwich.yaml")
    args = parser.parse_args()
    main(args)




