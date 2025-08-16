import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from diffusers import DDIMScheduler, AutoencoderKL, FlowMatchEulerDiscreteScheduler

from model.mmdit import load_mmdit
from model.dit.ar_clip import AR_CLIP
from runner.mmdit.train_basic_sd3 import sample_sd3_5

sample_scheduler = DDIMScheduler(
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

def diff_generate(feature, diff_head):
    sample_scheduler.set_timesteps(50)
    B = feature.shape[0]

    pred_latents = torch.randn((B, 16), device=feature.device)
    pred_latents *= sample_scheduler.init_noise_sigma

    for t in sample_scheduler.timesteps:
        pred_latents = sample_scheduler.scale_model_input(pred_latents, t)
        with torch.no_grad():
            t_sample = torch.as_tensor([t], device=feature.device)
            noise_pred = diff_head(pred_latents, t_sample.repeat(B), feature)
            pred_latents = sample_scheduler.step(noise_pred, t, pred_latents).prev_sample

    return pred_latents

@torch.no_grad()
def sample_imagenet():
    device = torch.device("cuda:0")
    dtype = torch.float16

    # load ar_clip
    exp_dir = "/data/phd/jinjiachun/experiment/clip_dit/0815_ar_clip"
    config = OmegaConf.load(os.path.join(exp_dir, "config.yaml"))
    step = 5000

    ar_model = AR_CLIP(config.ar_clip)
    ar_model.load_state_dict(torch.load(os.path.join(exp_dir, f"ar_model-{config.train.exp_name}-{step}"), map_location="cpu", weights_only=True))
    ar_model = ar_model.to(device, dtype).eval()

    # load diffusion decoder
    mmdit_step = 110000
    exp_dir = "/data/phd/jinjiachun/experiment/mmdit/0813_sd3_1024"
    config_path = os.path.join(exp_dir, "config.yaml")
    config_decoder = OmegaConf.load(config_path)
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(config_decoder.sd3_5_path, subfolder="scheduler")
    mmdit = load_mmdit(config_decoder)

    # load vae
    vae = AutoencoderKL.from_pretrained(config_decoder.sd3_5_path, subfolder="vae")
    vae.requires_grad_(False)
    vae = vae.to(device, dtype).eval()

    label = [22]
    


if __name__ == "__main__":
    sample_imagenet()