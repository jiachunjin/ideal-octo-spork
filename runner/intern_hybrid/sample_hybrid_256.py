import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from diffusers import DDIMScheduler, AutoencoderKL, FlowMatchEulerDiscreteScheduler

from model.mmdit import load_mmdit
from model.dit.hybrid_dit import HybridDiT_256
from runner.mmdit.train_basic_sd3 import sample_sd3_5
from model.internvl.modeling_internvl_chat import InternVLChatModel
from model.vae_aligner.vit_vae_aligner import get_feature_down_proj

@torch.no_grad()
def sample_imagenet():
    device = torch.device("cuda:0")
    dtype = torch.float16

    # ---------------------------------
    # ---------- load models ----------
    # ---------------------------------
    exp_dir = "/data/phd/jinjiachun/experiment/clip_dit/0819_intern_hybrid_256"
    step = 20000
    exp_name = exp_dir.split("/")[-1]
    config = OmegaConf.load(os.path.join(exp_dir, "config.yaml"))

    model = HybridDiT_256(config.hybrid_dit)
    model.load_state_dict(torch.load(os.path.join(exp_dir, f"hybrid_dit-clip_dit-{step}"), map_location="cpu", weights_only=True), state_dict=True)
    model = model.to(device, dtype).eval()

    internvl = InternVLChatModel.from_pretrained(config.intern_vl_8b_path)
    internvl = internvl.to(device, dtype).eval()

    feature_down_projector = get_feature_down_proj(config.feature_down_projector)
    ckpt = torch.load(config.feature_down_projector.path, map_location="cpu", weights_only=True)
    ckpt = {k.replace("feature_down_projector.", ""): v for k, v in ckpt.items() if k.startswith("feature_down_projector.")}
    feature_down_projector.load_state_dict(ckpt, strict=True)
    feature_down_projector = feature_down_projector.to(device, dtype).eval()

    # ---------------------------------
    # --------- load an image ---------
    # ---------------------------------
    from PIL import Image
    import torchvision.transforms as pth_transforms
    from model.internvl import extract_feature_pre_adapter

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    transform = pth_transforms.Compose([
        pth_transforms.Resize(config.data.img_size, max_size=None),
        pth_transforms.CenterCrop(config.data.img_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    img = Image.open("/data/phd/jinjiachun/codebase/ideal-octo-spork/asset/internet/imagenet_dog.png")

    img = transform(img)
    img = img.unsqueeze(0).to(device, dtype)

    x_clip = extract_feature_pre_adapter(internvl.vision_model, img)

    print(x_clip.shape)

