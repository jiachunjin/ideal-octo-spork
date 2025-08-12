import torch
import torchvision.transforms as pth_transforms

from PIL import Image
from diffusers import AutoencoderDC




if __name__ == "__main__":
    vae_id = "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers"
    vae = AutoencoderDC.from_pretrained(vae_id, subfolder="vae")
    vae.eval()
    vae.requires_grad_(False)

    device = "cuda:0"
    dtype = torch.float16

    vae_transform = pth_transforms.Compose([
        pth_transforms.Resize(448, max_size=None),
        pth_transforms.CenterCrop(448),
        pth_transforms.ToTensor(),
    ])
    img = Image.open("/data/phd/jinjiachun/codebase/ideal-octo-spork/asset/internet/letter.jpeg").convert("RGB")

    x = vae_transform(img).unsqueeze(0).to(device, dtype)
    x = x * 2 - 1
    print(f"{x.shape=}")

    latents = vae.encode(x).latent
    latents = latents * vae.config.scaling_factor
    print(f"{latents.shape=}")

    samples = vae.decode(latents).sample
    print(f"{samples.shape=}")
    samples = samples / vae.config.scaling_factor
    samples = (samples + 1) / 2
    samples = samples.clamp(0, 1)

    import torchvision.utils as vutils
    vutils.save_image(samples, "asset/sana_decoder/dim_1024.png", nrow=1, normalize=False)


