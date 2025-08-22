import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from diffusers import DDIMScheduler, AutoencoderKL, FlowMatchEulerDiscreteScheduler

from model.mmdit import load_mmdit
from model.dit.standard_dit import DiT
from runner.mmdit.train_basic_sd3 import sample_sd3_5


@torch.no_grad()
def sample_imagenet():
    device = torch.device("cuda:0")
    dtype = torch.float16

    # load dit
    # exp_dir = "/data/phd/jinjiachun/experiment/clip_dit/dit_256_8"
    # exp_dir = "/data/phd/jinjiachun/experiment/clip_dit/dit_256_8_norm"
    # exp_dir = "/data/phd/jinjiachun/experiment/clip_dit/dit_2048"
    exp_dir = "/data/phd/jinjiachun/experiment/clip_dit/dit_1024_1024_12_validation"
    exp_name = exp_dir.split("/")[-1]
    config = OmegaConf.load(os.path.join(exp_dir, "config.yaml"))
    step = 38000

    dit_model = DiT(config.dit)
    dit_model.load_state_dict(torch.load(os.path.join(exp_dir, f"dit-clip_dit-{step}"), map_location="cpu", weights_only=True))
    dit_model = dit_model.to(device, dtype).eval()

    # load diffusion decoder
    # mmdit_step = 35000
    # mmdit_step = 50000
    mmdit_step = 140000
    # exp_dir = "/data/phd/jinjiachun/experiment/mmdit/0817_sd3_256"
    exp_dir = "/data/phd/jinjiachun/experiment/mmdit/0813_sd3_1024"
    config_path = os.path.join(exp_dir, "config.yaml")
    config_decoder = OmegaConf.load(config_path)
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(config_decoder.sd3_5_path, subfolder="scheduler")
    mmdit = load_mmdit(config_decoder)
    ckpt_path = os.path.join(exp_dir, f"mmdit-mmdit-{mmdit_step}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    mmdit.load_state_dict(ckpt, strict=False)
    mmdit = mmdit.to(device, dtype).eval()

    # load vae
    vae = AutoencoderKL.from_pretrained(config_decoder.sd3_5_path, subfolder="vae")
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

    B = 16
    cfg_scale = 2.  # 支持CFG，设置为1.0即为无CFG
    x = torch.randn((B, config.dit.num_tokens, config.dit.in_channels), device=device, dtype=dtype)
    label = 22
    y = torch.as_tensor([label]*B, device=device).long()
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
                x = torch.cat([x_out, x_out], dim=0)
            else:
                noise_pred = dit_model(x_in, t_sample, y)
                x = scheduler.step(noise_pred, t, x).prev_sample
    if cfg_scale > 1.0:
        x = x[:B]

    context = x
    print(context.shape)

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
    sample_path = f"asset/clip_dit/{exp_name}_{step}_{label}_{cfg_scale}.png"
    vutils.save_image(samples, sample_path, nrow=4, normalize=False)
    print(f"Samples saved to {sample_path}")    
if __name__ == "__main__":
    sample_imagenet()