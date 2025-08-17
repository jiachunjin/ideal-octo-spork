import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

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

    


    


    ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/clip_dit/ar_clip.yaml")
    args = parser.parse_args()
    main(args)
