import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torchvision.transforms as pth_transforms
from PIL import Image
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler


from model.mmdit import load_mmdit, sample_sd3_5
from model.internvl.modeling_internvl_chat import InternVLChatModel

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

@torch.no_grad()
def run():
    exp_dir = "/data/phd/jinjiachun/experiment/mmdit/0714_mmdit_dev"
    config_path = os.path.join(exp_dir, "config.yaml")
    config = OmegaConf.load(config_path)

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(config.sd3_5_path, subfolder="scheduler")
    ar_model = InternVLChatModel.from_pretrained(config.intern_vl_1b_path)
    vae = AutoencoderKL.from_pretrained(config.sd3_5_path, subfolder="vae")
    vae.requires_grad_(False)

    mmdit = load_mmdit(config)
    ckpt_path = os.path.join(exp_dir, "")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    mmdit.load_state_dict(ckpt, strict=True)

    device = torch.device("cuda:0")
    dtype = torch.float16
    vae = vae.to(device, dtype).eval()
    mmdit = mmdit.to(device, dtype).eval()
    ar_model = ar_model.to(device, dtype).eval()

    vae_transform = pth_transforms.Compose([
        pth_transforms.Resize(384, max_size=None),
        pth_transforms.CenterCrop(384),
        pth_transforms.ToTensor(),
    ])

    img_1 = Image.open("/data/phd/jinjiachun/codebase/connector/asset/kobe.png").convert("RGB")
    img_2 = Image.open("/data/phd/jinjiachun/codebase/connector/asset/004.jpg").convert("RGB")

    x_1 = vae_transform(img_1).unsqueeze(0).to(device, dtype)
    x_2 = vae_transform(img_2).unsqueeze(0).to(device, dtype)
    x = torch.cat([x_1, x_2], dim=0)

    imagenet_mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
    imagenet_std = torch.tensor(IMAGENET_STD, device=device, dtype=dtype).view(1, 3, 1, 1)
    x = (x - imagenet_mean) / imagenet_std

    x_clip = ar_model.extract_feature(x) # (B, 256, 896)
    context = x_clip

    samples = sample_sd3_5(
        transformer         = mmdit,
        vae                 = vae,
        noise_scheduler     = noise_scheduler,
        device              = device,
        dtype               = dtype,
        context             = context,
        batch_size          = context.shape[0],
        height              = 384,
        width               = 384,
        num_inference_steps = 20,
        guidance_scale      = 5.0,
        seed                = 42
    )

    print(samples.shape)

if __name__ == "__main__":
    run()