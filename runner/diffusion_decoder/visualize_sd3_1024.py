import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torchvision.transforms as pth_transforms
from PIL import Image
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler


from model.mmdit import load_mmdit
from runner.mmdit.train_basic_sd3 import sample_sd3_5
from model.internvl import extract_feature_pre_adapter, extract_feature_pre_shuffle_adapter
from model.internvl.modeling_internvl_chat import InternVLChatModel

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

@torch.no_grad()
def run():
    exp_dir = "/data/phd/jinjiachun/experiment/mmdit/0813_sd3_1024"
    config_path = os.path.join(exp_dir, "config.yaml")
    config = OmegaConf.load(config_path)

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(config.sd3_5_path, subfolder="scheduler")
    if not hasattr(config, "base_model"):
        config.base_model = "intern_vl_1b"
    print(config.base_model)
    if config.base_model == "intern_vl_1b":
        vision_model = InternVLChatModel.from_pretrained(config.intern_vl_1b_path).vision_model
    elif config.base_model == "intern_vl_2b":
        vision_model = InternVLChatModel.from_pretrained(config.intern_vl_2b_path).vision_model
    elif config.base_model == "intern_vl_8b":
        vision_model = InternVLChatModel.from_pretrained(config.intern_vl_8b_path).vision_model
    else:
        raise ValueError(f"Invalid base model: {config.base_model}")
    vae = AutoencoderKL.from_pretrained(config.sd3_5_path, subfolder="vae")
    vae.requires_grad_(False)

    mmdit = load_mmdit(config)
    ckpt_path = os.path.join(exp_dir, "mmdit-mmdit-90000")
    exp_name = exp_dir.split("/")[-1]
    step = ckpt_path.split("-")[-1]
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    mmdit.load_state_dict(ckpt, strict=True)

    device = torch.device("cuda:0")
    dtype = torch.float16
    vae = vae.to(device, dtype).eval()
    mmdit = mmdit.to(device, dtype).eval()
    vision_model = vision_model.to(device, dtype).eval()

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

    x_clip = extract_feature_pre_shuffle_adapter(vision_model, x)
    context = x_clip
    print(f"{context.shape=}")

    samples = sample_sd3_5(
        transformer         = mmdit,
        vae                 = vae,
        noise_scheduler     = noise_scheduler,
        device              = device,
        dtype               = dtype,
        context             = context,
        batch_size          = context.shape[0],
        height              = 448,
        width               = 448,
        num_inference_steps = 25,
        guidance_scale      = 1.0,
        seed                = 42
    )
    print(samples.shape)

    import torchvision.utils as vutils
    sample_path = f"asset/mmdit/aligner_free/{exp_name}_{step}.png"
    vutils.save_image(samples, sample_path, nrow=2, normalize=False)
    print(f"Samples saved to {sample_path}")    

if __name__ == "__main__":
    run()
