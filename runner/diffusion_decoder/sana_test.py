import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torchvision.transforms as pth_transforms
from PIL import Image
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler

from model.diffusion_decoder.sana_decoder import SanaDecoder
from model.internvl.modeling_internvl_chat import InternVLChatModel

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@torch.no_grad()
def run():
    exp_dir = "/data/phd/jinjiachun/experiment/sana_1024/0812_dev"
    config_path = os.path.join(exp_dir, "config.yaml")
    config = OmegaConf.load(config_path)

    device = torch.device("cuda:0")
    dtype = torch.float16
    sana_decoder = SanaDecoder(config).to(device, dtype).eval()
    ckpt_transformer = torch.load(os.path.join(exp_dir, "transformer-sana_1024-15000"), map_location="cpu", weights_only=True)
    ckpt_connector = torch.load(os.path.join(exp_dir, "connector-sana_1024-15000"), map_location="cpu", weights_only=True)
    sana_decoder.transformer.load_state_dict(ckpt_transformer, strict=True)
    sana_decoder.connector.load_state_dict(ckpt_connector, strict=True)

    if config.base_model == "intern_vl_1b":
        vision_model = InternVLChatModel.from_pretrained(config.intern_vl_1b_path).vision_model
    elif config.base_model == "intern_vl_2b":
        vision_model = InternVLChatModel.from_pretrained(config.intern_vl_2b_path).vision_model
    elif config.base_model == "intern_vl_8b":
        vision_model = InternVLChatModel.from_pretrained(config.intern_vl_8b_path).vision_model
    vision_model.requires_grad_(False)
    vae = AutoencoderKL.from_pretrained(config.sd3_5_path, subfolder="vae")
    vae.requires_grad_(False)
    vae = vae.to(device, dtype).eval()

    vae_transform = pth_transforms.Compose([
        pth_transforms.Resize(448, max_size=None),
        pth_transforms.CenterCrop(448),
        pth_transforms.ToTensor(),
    ])

    images = [
        Image.open("/data/phd/jinjiachun/codebase/ideal-octo-spork/asset/internet/letter.jpeg").convert("RGB"),
        Image.open("/data/phd/jinjiachun/codebase/ideal-octo-spork/asset/internet/letter1.webp").convert("RGB"),
        Image.open("/data/phd/jinjiachun/codebase/connector/asset/kobe.png").convert("RGB"),
        Image.open("/data/phd/jinjiachun/codebase/connector/asset/004.jpg").convert("RGB"),
        Image.open("/data/phd/jinjiachun/codebase/ideal-octo-spork/asset/internet/einstein.jpg"),
        Image.open("/data/phd/jinjiachun/codebase/ideal-octo-spork/asset/internet/jobs.jpg"),
        Image.open("/data/phd/jinjiachun/codebase/ideal-octo-spork/asset/internet/mcdonald.jpg"),
        Image.open("/data/phd/jinjiachun/codebase/ideal-octo-spork/asset/internet/messi_1.jpg"),
        Image.open("/data/phd/jinjiachun/codebase/ideal-octo-spork/asset/internet/messi_2.webp"),
        Image.open("/data/phd/jinjiachun/codebase/ideal-octo-spork/asset/internet/messi.jpg"),
        Image.open("/data/phd/jinjiachun/codebase/ideal-octo-spork/asset/internet/ronaldo.jpg"),
        Image.open("/data/phd/jinjiachun/codebase/ideal-octo-spork/asset/internet/trump.jpg"),
    ]
    x_list = []
    for img in images:
        x_list.append(vae_transform(img).unsqueeze(0).to(device, dtype))
    x = torch.cat(x_list, dim=0)

    imagenet_mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
    imagenet_std = torch.tensor(IMAGENET_STD, device=device, dtype=dtype).view(1, 3, 1, 1)
    x = (x - imagenet_mean) / imagenet_std

    x_clip = vision_model(
        pixel_values         = x,
        output_hidden_states = False,
        return_dict          = True
    ).last_hidden_state[:, 1:, :]

    raise NotImplementedError("孩子没写完")