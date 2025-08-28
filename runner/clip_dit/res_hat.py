import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
from omegaconf import OmegaConf
from einops import rearrange

from model.dit.dit_head import equip_internvl_res_hat
from model.internvl.modeling_internvl_chat import InternVLChatModel
from model.internvl import extract_both_clip
from util.my_tool_box import get_accelerator, get_t2i_dataloader
from util.misc import process_pretrained_model_path, flatten_dict


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def main(args):
    config = OmegaConf.load(args.config)
    config = process_pretrained_model_path(config)

    accelerator, output_dir = get_accelerator(config)
    internvl = InternVLChatModel.from_pretrained(config.intern_vl_8b_path)
    internvl, train_scheduler = equip_internvl(internvl, config.model)

    equip_internvl_res_hat(internvl, config.model)

    num_para = sum(p.numel() for p in internvl.parameters() if p.requires_grad)
    print(f"trainable num_para: {num_para}")

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    internvl.to(device).to(dtype)

    B = 3
    joint_embedding = torch.randn(B, 512, 3584).to(device).to(dtype)
    attention_mask = torch.ones(B, 512).to(device).to(dtype)

    outputs = internvl.language_model(
        inputs_embeds = joint_embedding,
        attention_mask = attention_mask,
        output_hidden_states = True,
    )
    print(outputs.hidden_states.shape)