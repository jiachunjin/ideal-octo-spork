import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from omegaconf import OmegaConf
from model.vae_aligner import get_vae_aligner
from model.janus.models import MultiModalityCausalLM

import matplotlib.pyplot as plt

from PIL import Image
from diffusers import AutoencoderKL
from torchvision import transforms as pth_transforms


config = OmegaConf.load("config/vae_aligner/siglip_flux.yaml")
config.janus_1b_path = "/data/phd/jinjiachun/ckpt/deepseek-ai/Janus-Pro-1B"
config.vae_path = "/data/phd/jinjiachun/ckpt/stabilityai/stable-diffusion-3.5-medium/vae"
vae_aligner = get_vae_aligner(config.vae_aligner)
siglip = MultiModalityCausalLM.from_pretrained(config.janus_1b_path, trust_remote_code=True).vision_model

ckpt_path = "/data/phd/jinjiachun/experiment/vae_aligner/0714_sd3_vae_aligner_hybrid/vae_aligner-vae_aligner-1k"
print("current ckpt: ", ckpt_path)

vae = AutoencoderKL.from_pretrained(config.vae_path)
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
vae_aligner.load_state_dict(ckpt, strict=False)

img = Image.open("/data/phd/jinjiachun/codebase/connector/asset/kobe.png").convert("RGB")
vae_transform = pth_transforms.Compose([
    pth_transforms.Resize(384, max_size=None),
    pth_transforms.CenterCrop(384),
    pth_transforms.ToTensor(),
])
x = vae_transform(img).unsqueeze(0)
x = x * 2 - 1

with torch.no_grad():
    x_siglip = siglip(x)
    z = vae.encode(x).latent_dist.sample()
    rec_latent = vae_aligner(x_siglip)
    reconstructed = vae.decode(rec_latent).sample
    residual = vae.decode(z - rec_latent).sample
    rec = vae.decode(z).sample

print(f"MSE loss: {torch.nn.functional.mse_loss(z, rec_latent)}")

original = (x + 1) / 2
original = torch.clamp(original, 0, 1)
reconstructed = (reconstructed + 1) / 2
reconstructed = torch.clamp(reconstructed, 0, 1)
residual = (residual + 1) / 2
residual = torch.clamp(residual, 0, 1)
rec = (rec + 1) / 2
rec = torch.clamp(rec, 0, 1)

# 转换为PIL图像并显示
reconstructed_img = pth_transforms.ToPILImage()(reconstructed.squeeze(0))
original_img = pth_transforms.ToPILImage()(original.squeeze(0))
residual_img = pth_transforms.ToPILImage()(residual.squeeze(0))
rec_img = pth_transforms.ToPILImage()(rec.squeeze(0))
# 创建一个包含两个子图的图像
fig, axes = plt.subplots(1, 4, figsize=(24, 6))

# 显示原始图像
axes[0].imshow(original_img)
axes[0].set_title('original', fontsize=14)
axes[0].axis('off')

# 显示重建图像
axes[1].imshow(reconstructed_img)
axes[1].set_title('reconstructed', fontsize=14)
axes[1].axis('off')

# 显示残差图像
axes[2].imshow(residual_img)
axes[2].set_title('residual', fontsize=14)
axes[2].axis('off')

axes[3].imshow(rec_img)
axes[3].set_title('rec', fontsize=14)
axes[3].axis('off')

plt.tight_layout()
plt.show()

plt.savefig("vae_siglip_feature_compare.png")