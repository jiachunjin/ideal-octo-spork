import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from tqdm import tqdm, trange
from omegaconf import OmegaConf
from diffusers import DDIMScheduler

from model.dit.hybrid_dit import HybridDiT

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
    device = torch.device("cpu")
    dtype = torch.float32

    # exp_dir = "/data/phd/jinjiachun/experiment/clip_1024/0820_overfit_1024_null_condition"
    step = 5000

    # config = OmegaConf.load(os.path.join(exp_dir, "config.yaml"))
    config = OmegaConf.load("config/overfit_1024/null_condition.yaml")
    model = HybridDiT(config.hybrid_dit)
    # model.load_state_dict(torch.load(os.path.join(exp_dir, f"hybrid_dit-{config.train.exp_name}-{step}"), map_location="cpu", weights_only=True))
    model = model.to(device, dtype).eval()

    def sample_one_clip_block(model, prefix):
        B = 1
        x_t = torch.randn((B, 4, 1024), device=device, dtype=dtype)

        for t in scheduler.timesteps:
            x_t = scheduler.scale_model_input(x_t, t)
            with torch.no_grad():
                t = torch.as_tensor([t], device=device)
                noise_pred = model.forward_test(x_t, t, prefix)
                x_t = scheduler.step(noise_pred, t, x_t).prev_sample

        return x_t

    x_clip = torch.empty((1, 0, 1024), device=device, dtype=dtype)
    for i in trange(256):
        print(x_clip.shape)
        x_clip_block = sample_one_clip_block(model, x_clip)

        x_clip = torch.cat([x_clip, x_clip_block], dim=1)


if __name__ == "__main__":
    sample_imagenet()