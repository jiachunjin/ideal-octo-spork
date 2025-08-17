import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import argparse
from omegaconf import OmegaConf
from util.misc import process_pretrained_model_path

from util.my_tool_box import get_accelerator
from model.internvl.modeling_internvl_chat import InternVLChatModel
from model.dit.hybrid_dit import HybridDiT

def main(args):
    config = OmegaConf.load(args.config)
    config = process_pretrained_model_path(config)
    accelerator, output_dir = get_accelerator(config)

    internvl = InternVLChatModel.from_pretrained(config.intern_vl_8b_path)
    vision_model = internvl.vision_model

    model = HybridDiT(config.hybrid_dit)

    if config.train.resume_path is not None:
        raise NotImplementedError("Resume is not implemented")

    params_to_learn = list(p for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(
        params_to_learn,
        lr           = config.train.lr,
        betas        = (0.9, 0.95),
        weight_decay = 5e-2,
        eps          = 1e-8,
    )

    global_step = config.train.global_step if config.train.global_step is not None else 0

    if accelerator.mixed_precision == "bf16":
        dtype = torch.bfloat16
    elif accelerator.mixed_precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32


    ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/clip_dit/ar_clip.yaml")
    args = parser.parse_args()
    main(args)
