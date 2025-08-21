import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


import torch
from transformers import AutoTokenizer
from model.internvl.modeling_internvl_chat import InternVLChatModel

@torch.no_grad()
def understand_generation():
    device = torch.device("cuda:7")
    dtype = torch.float16

    clip_features = torch.load("/data/phd/jinjiachun/codebase/ideal-octo-spork/asset/clip_dit/0820_overfit_dog_8000_dog_clip.pt")
    clip_features = clip_features.to(device, dtype)
    print(clip_features.shape)

    internvl_path = "/data/phd/jinjiachun/ckpt/OpenGVLab/InternVL3-8B"
    internvl = InternVLChatModel.from_pretrained(internvl_path)
    internvl = internvl.to(device, dtype).eval()

    tokenizer = AutoTokenizer.from_pretrained(internvl_path, trust_remote_code=True, use_fast=False)
    generation_config = dict(max_new_tokens=1024, do_sample=True)

    for i in range(clip_features.shape[0]):
        clip_feature = clip_features[i].unsqueeze(0)
        question = '<image>\nPlease describe the image in detail.'
        response = internvl.chat_with_clip(tokenizer, clip_feature, question, generation_config)
        print(f'User: {question}\nAssistant: {response}')


if __name__ == "__main__":
    understand_generation()