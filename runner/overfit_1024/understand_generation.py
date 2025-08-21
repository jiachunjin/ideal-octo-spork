import torch
from transformers import AutoTokenizer
from model.internvl.modeling_internvl_chat import InternVLChatModel

@torch.no_grad()
def understand_generation():
    device = torch.device("cuda:7")
    dtype = torch.float16

    clip_feature = torch.load("/data/phd/jinjiachun/codebase/ideal-octo-spork/asset/clip_dit/0820_overfit_dog_6000_dog_clip.pt")
    clip_feature = clip_feature.to(device, dtype)
    print(clip_feature.shape)
    clip_feature = clip_feature[0].unsqueeze(0)

    internvl_path = "/data/phd/jinjiachun/ckpt/OpenGVLab/InternVL3-8B"
    internvl = InternVLChatModel.from_pretrained(internvl_path)
    internvl = internvl.to(device, dtype).eval()

    tokenizer = AutoTokenizer.from_pretrained(internvl_path, trust_remote_code=True, use_fast=False)
    generation_config = dict(max_new_tokens=1024, do_sample=True)

    question = '<image>\nPlease describe the image in detail.'
    response = internvl.chat_with_clip(tokenizer, clip_feature, question, generation_config)
    print(f'User: {question}\nAssistant: {response}')



if __name__ == "__main__":
    understand_generation()