import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from tqdm import tqdm, trange
from omegaconf import OmegaConf
from diffusers import DDIMScheduler

from model.overfit_1024.hybrid_dit_class import HybridDiT_Class
from model.mmdit import load_mmdit
from runner.mmdit.train_basic_sd3 import sample_sd3_5
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler

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

    # ---------- load model ----------
    exp_dir = "/data/phd/jinjiachun/experiment/clip_1024/0820_overfit_1024_null_condition_50000"
    exp_name = exp_dir.split("/")[-1]
    step = 5000

    config = OmegaConf.load(os.path.join(exp_dir, "config.yaml"))
    # config = OmegaConf.load("config/overfit_1024/null_condition.yaml")
    model = HybridDiT_Class(config.hybrid_dit)
    model.load_state_dict(torch.load(os.path.join(exp_dir, f"hybrid_dit-{config.train.exp_name}-{step}"), map_location="cpu", weights_only=True))
    model = model.to(device, dtype).eval()

    ...

    # ---------- do autoregressive diffusion sampling ----------
    def sample_one_clip_block(model, prefix, y, cfg_scale=2.0):
        B = y.shape[0]
        x_t = torch.randn((B, 4, 1024), device=device, dtype=dtype)

        if cfg_scale > 1.0:
            x_t = x_t.repeat(2, 1, 1)
            prefix = prefix.repeat(2, 1, 1)
            y_null = torch.full_like(y, fill_value=1000, dtype=torch.int64, device=device)
            y_cfg = torch.cat([y, y_null], dim=0)
        else:
            raise NotImplementedError("Use CFG !!!")

        for t in tqdm(scheduler.timesteps):
            x_t = scheduler.scale_model_input(x_t, t)
            with torch.no_grad():
                t_tensor = torch.as_tensor([t], device=device)
                noise_pred = model.forward_test(x_t, t_tensor, prefix, y_cfg)
                
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)

                x_out = x_t[:B]
                x_out = scheduler.step(noise_pred, t, x_out).prev_sample
                x_t = torch.cat([x_out, x_out], dim=0)

        return x_t[:B]

    B = 4
    label = torch.tensor([22]*B, dtype=torch.int64, device=device)

    x_clip = torch.empty((B, 0, 1024), device=device, dtype=dtype)
    for i in trange(256):
        x_clip_block = sample_one_clip_block(model, x_clip, label)
        print(x_clip_block.shape)
        x_clip = torch.cat([x_clip, x_clip_block], dim=1)

    print(x_clip.shape)


if __name__ == "__main__":
    sample_imagenet()