import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from omegaconf import OmegaConf
from transformers import AutoModel
from model.vae_aligner import get_vae_aligner

from PIL import Image
from diffusers import AutoencoderKL
from torchvision import transforms as pth_transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
imagenet_mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
imagenet_std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

exp_dir = "/data/phd/jinjiachun/experiment/intern_clip/0805_intern_aligner"

config = OmegaConf.load(os.path.join(exp_dir, "config.yaml"))
config.vae_path = "/data/phd/jinjiachun/ckpt/stabilityai/stable-diffusion-3.5-medium/vae"
vae_aligner = get_vae_aligner(config.vae_aligner)
intern_vl_1b = AutoModel.from_pretrained(config.intern_vl_1b_path, trust_remote_code=True)
vae = AutoencoderKL.from_pretrained(config.vae_path)

ckpt_path = os.path.join(exp_dir, "vae_aligner-intern_clip-4666")
print("current ckpt: ", ckpt_path)
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
vae_aligner.load_state_dict(ckpt, strict=True)

img = Image.open("/data/phd/jinjiachun/codebase/connector/asset/kobe.png").convert("RGB")
vae_transform = pth_transforms.Compose([
    pth_transforms.Resize(448, max_size=None),
    pth_transforms.CenterCrop(448),
    pth_transforms.ToTensor(),
])
x = vae_transform(img).unsqueeze(0)

x_intern = (x - imagenet_mean) / imagenet_std

with torch.no_grad():
    x_clip = intern_vl_1b.extract_feature(x_intern)
    rec_latent = vae_aligner(x_clip)
    reconstructed = vae.decode(rec_latent).sample
    z = vae.encode(x * 2 - 1).latent_dist.sample()
    rec = vae.decode(z).sample

print(f"MSE loss: {torch.nn.functional.mse_loss(z, rec_latent)}")


reconstructed = (reconstructed + 1) / 2
reconstructed = torch.clamp(reconstructed, 0, 1)
rec = (rec + 1) / 2
rec = torch.clamp(rec, 0, 1)

reconstructed_img = pth_transforms.ToPILImage()(reconstructed.squeeze(0))
original_img = pth_transforms.ToPILImage()(x.squeeze(0))
vae_img = pth_transforms.ToPILImage()(rec.squeeze(0))

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(16, 8))

axes[0].imshow(original_img)
axes[0].set_title('original', fontsize=14)
axes[0].axis('off')

axes[1].imshow(reconstructed_img)
axes[1].set_title('reconstructed', fontsize=14)
axes[1].axis('off')

axes[2].imshow(vae_img)
axes[2].set_title('vae', fontsize=14)
axes[2].axis('off')

plt.tight_layout()
plt.show()

plt.savefig("intern_aligner.png")