import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from diffusers import DDIMScheduler, AutoencoderKL
from model.dit.standard_dit import DiT


@torch.no_grad()
def sample_imagenet():
    device = torch.device("cuda:0")
    dtype = torch.float16

    # load dit
    exp_dir = "/data/phd/jinjiachun/experiment/clip_dit/dit_300M"
    config = OmegaConf.load(os.path.join(exp_dir, "config.yaml"))

    dit_model = DiT(config.dit)
    dit_model.load_state_dict(torch.load(os.path.join(exp_dir, "dit-clip_dit-5000"), map_location="cpu", weights_only=True))
    dit_model = dit_model.to(device, dtype).eval()

    # load diffusion decoder

    # load vae
    vae = AutoencoderKL.from_pretrained(config.sd3_5_path, subfolder="vae")
    vae.requires_grad_(False)
    vae = vae.to(device, dtype).eval()


    # sample from dit

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

    B = 3
    cfg_scale = 3.0  # 支持CFG，设置为1.0即为无CFG
    x = torch.randn((B, 1024, 1024), device=device, dtype=dtype)
    y = torch.as_tensor([22]*B, device=device).long()
    x *= scheduler.init_noise_sigma

    if cfg_scale > 1.0:
        # 复制一份uncond输入
        x = x.repeat(2, 1, 1)
        y_uncond = torch.full_like(y, fill_value=config.dit.num_classes)  # 假设最后一个类别为uncond
        y = torch.cat([y, y_uncond], dim=0)

    for t in tqdm(scheduler.timesteps):
        x_in = scheduler.scale_model_input(x, t)
        with torch.no_grad():
            t_sample = torch.as_tensor([t], device=device)
            if cfg_scale > 1.0:
                noise_pred = dit_model(x_in, t_sample, y)
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
                x_out = x[:B]  # 只保留cond部分
                x_out = scheduler.step(noise_pred, t, x_out).prev_sample
                # 更新x的前B个为新值，uncond部分保持不变
                x = torch.cat([x_out, x[B:]], dim=0)
            else:
                noise_pred = dit_model(x_in, t_sample, y)
                x = scheduler.step(noise_pred, t, x).prev_sample
    if cfg_scale > 1.0:
        x = x[:B]

    print(x.shape)


    ...