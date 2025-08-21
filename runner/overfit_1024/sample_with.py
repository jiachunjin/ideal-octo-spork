import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from tqdm import tqdm, trange
from omegaconf import OmegaConf
from diffusers import DDIMScheduler

from model.dit.hybrid_dit_conditional import HybridDiT_AdaLN
from model.mmdit import load_mmdit
from runner.mmdit.train_basic_sd3 import sample_sd3_5
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from model.internvl.modeling_internvl_chat import InternVLChatModel

scheduler = DDIMScheduler(
    beta_schedule          = "scaled_linear",
    beta_start             = 0.00085,
    beta_end               = 0.012,
    num_train_timesteps    = 1000,
    clip_sample            = False,
    prediction_type        = "v_prediction",
    set_alpha_to_one       = True,
    steps_offset           = 1,
    trained_betas          = None,
    timestep_spacing       = "trailing",
    rescale_betas_zero_snr = True
)
scheduler.set_timesteps(50)

@torch.no_grad()
def sample_imagenet():
    device = torch.device("cuda:0")
    dtype = torch.float16

    exp_dir = "/data/phd/jinjiachun/experiment/clip_1024/0820_overfit_1024_with_condition"
    exp_name = exp_dir.split("/")[-1]
    step = 5000

    config = OmegaConf.load(os.path.join(exp_dir, "config.yaml"))
    model = HybridDiT_AdaLN(config.hybrid_dit)
    model.load_state_dict(torch.load(os.path.join(exp_dir, f"hybrid_dit-{config.train.exp_name}-{step}"), map_location="cpu", weights_only=True))
    model = model.to(device, dtype).eval()

    internvl = InternVLChatModel.from_pretrained(config.intern_vl_8b_path)
    internvl = internvl.to(device, dtype).eval()

    # exp_dir = "/data/phd/jinjiachun/experiment/mmdit/0813_sd3_1024"
    # noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(config.sd3_5_path, subfolder="scheduler")
    # mmdit = load_mmdit(config)
    # ckpt_path = os.path.join(exp_dir, "mmdit-mmdit-140000")
    # ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    # mmdit.load_state_dict(ckpt, strict=True)
    # mmdit = mmdit.to(device, dtype).eval()

    # vae = AutoencoderKL.from_pretrained(config.sd3_5_path, subfolder="vae")
    # vae.requires_grad_(False)
    # vae = vae.to(device, dtype).eval()

    def sample_one_clip_block_with_condition(model, prefix, condition):
        B = 1
        x_t = torch.randn((B, 4, 1024), device=device, dtype=dtype)

        for t in scheduler.timesteps:
            x_t = scheduler.scale_model_input(x_t, t)
            with torch.no_grad():
                t_tensor = torch.as_tensor([t], device=device)
                noise_pred = model.forward_test_with_condition(x_t, t_tensor, prefix)
                x_t = scheduler.step(noise_pred, t, x_t).prev_sample

        return x_t
    
    def make_condition(x_clip):
        ...

    x_clip = torch.empty((1, 0, 1024), device=device, dtype=dtype)
    boi_embedding = internvl.language_model.get_input_embeddings()(torch.LongTensor([151665]).to(device)).unsqueeze(1)
    condition = boi_embedding
    for i in trange(256):
        x_clip_block = sample_one_clip_block_with_condition(model, x_clip, condition)
        x_clip = torch.cat([x_clip, x_clip_block], dim=1)
        condition = make_condition(x_clip)
        ...
        

    
    print(x_clip.shape)