import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


import torch
import argparse
import copy
from tqdm import tqdm
from omegaconf import OmegaConf
from einops import rearrange

from model.internvl.modeling_internvl_chat import InternVLChatModel
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from model.vae_aligner import get_vae_aligner
from model.internvl import extract_feature_pre_adapter
from model.dit.diff_mlp import intern_to_fork

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
    internvl, train_scheduler = intern_to_fork(internvl, config.model)

    params_to_learn = list(p for p in internvl.parameters() if p.requires_grad)

    accelerator.print(f"Learnable parameters: {sum(p.numel() for p in params_to_learn if p.requires_grad) / 1e6} M")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/sunshine/fork_few_dit.yaml")
    args = parser.parse_args()
    main(args)