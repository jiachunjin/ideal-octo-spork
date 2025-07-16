import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torchvision.transforms as pth_transforms
from PIL import Image
from einops import rearrange
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from model.janus.models import MultiModalityCausalLM
from model.vae_aligner import get_vae_aligner

from runner.mmdit.train_basic_sd3 import load_pretrained_mmdit, sample_sd3_5

exp_dir = "/data/phd/jinjiachun/experiment/mmdit/0714_mmdit_dev"

config_path = os.path.join(exp_dir, "config.yaml")
config = OmegaConf.load(config_path)

noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(config.sd3_5_path, subfolder="scheduler")
vae = AutoencoderKL.from_pretrained(config.sd3_5_path, subfolder="vae")
vae.requires_grad_(False)
siglip = MultiModalityCausalLM.from_pretrained(config.janus_1b_path, trust_remote_code=True).vision_model
siglip.requires_grad_(False)
vae_aligner = get_vae_aligner(config.vae_aligner)
ckpt_vae_aligner = torch.load(config.vae_aligner.pretrained_path, map_location="cpu", weights_only=True)
vae_aligner.load_state_dict(ckpt_vae_aligner, strict=True)
vae_aligner.requires_grad_(False)

transformer = load_pretrained_mmdit(config.sd3_5_path)
ckpt_path = os.path.join(exp_dir, "transformer-mmdit-30000")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
transformer.load_state_dict(ckpt, strict=True)

device = torch.device("cuda:0")
dtype = torch.float16
vae = vae.to(device, dtype).eval()
siglip = siglip.to(device, dtype).eval()
vae_aligner = vae_aligner.to(device, dtype).eval()
transformer = transformer.to(device, dtype).eval()

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
x = x * 2 - 1

with torch.no_grad():
    x_siglip = siglip(x)
    x_coarse_reference = vae_aligner(x_siglip)
    context = rearrange(x_coarse_reference, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=4, p2=4)
    print(f"{context.shape=}")

    samples = sample_sd3_5(
        transformer         = transformer,
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

    import torchvision.utils as vutils
    sample_path = f"samples.png"
    vutils.save_image(samples, sample_path, nrow=2, normalize=False)
    print(f"Samples saved to {sample_path}")

    with torch.no_grad():
        reconstructed = vae.decode(x_coarse_reference).sample
        reconstructed = reconstructed.to(torch.float32)
        reconstructed = (reconstructed + 1) / 2
        reconstructed = torch.clamp(reconstructed, 0, 1)
        vutils.save_image(reconstructed[:4], f"coarse.png", nrow=2, normalize=False)





