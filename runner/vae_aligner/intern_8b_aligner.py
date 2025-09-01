import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import argparse
from tqdm import tqdm
from omegaconf import OmegaConf
from einops import rearrange
from diffusers import AutoencoderKL

from model.vae_aligner import get_vae_aligner
from model.internvl.modeling_internvl_chat import InternVLChatModel
from model.internvl import extract_feature_pre_adapter

from util.misc import process_pretrained_model_path, flatten_dict
from util.my_tool_box import get_accelerator, get_t2i_dataloader

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def main(args):
    config = OmegaConf.load(args.config)
    config = process_pretrained_model_path(config)
    accelerator, output_dir = get_accelerator(config)

    vae_aligner = get_vae_aligner(config.vae_aligner)
    vae = AutoencoderKL.from_pretrained(config.vae_path)

    vision_model = InternVLChatModel.from_pretrained(config.intern_vl_8b_path).vision_model
    vision_model.requires_grad_(False)

    global_step = config.train.global_step if config.train.global_step is not None else 0
    params_to_learn = list(p for p in vae_aligner.parameters() if p.requires_grad)

    optimizer = torch.optim.AdamW(
        params_to_learn,
        lr           = config.train.lr,
        betas        = (0.9, 0.95),
        weight_decay = 5e-2,
        eps          = 1e-8,
    )

    if accelerator.mixed_precision == "bf16":
        dtype = torch.bfloat16
    elif accelerator.mixed_precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    dataloader = get_t2i_dataloader(config.data, accelerator)
    vae_aligner, optimizer = accelerator.prepare(vae_aligner, optimizer)
    vae.requires_grad_(False)
    vae = vae.to(accelerator.device, dtype).eval()
    vision_model.requires_grad_(False)
    vision_model = vision_model.to(accelerator.device, dtype).eval()

    training_done = False
    epoch = 0
    progress_bar = tqdm(
        total   = config.train.num_iter,
        initial = global_step,
        desc    = "Steps",
        disable = not accelerator.is_local_main_process,
    )

    config.device_count = accelerator.num_processes
    if accelerator.is_main_process:
        accelerator.init_trackers(config.train.wandb_proj, config=flatten_dict(config))
        with open(os.path.join(output_dir, "config.yaml"), "w") as f:
            OmegaConf.save(config, f)

    accelerator.print(f"Learnable parameters: {sum(p.numel() for p in params_to_learn if p.requires_grad) / 1e6} M")
    accelerator.print(f"Accelerator mixed precision: {accelerator.mixed_precision}")

    imagenet_mean = torch.tensor(IMAGENET_MEAN, device=accelerator.device, dtype=dtype).view(1, 3, 1, 1)
    imagenet_std = torch.tensor(IMAGENET_STD, device=accelerator.device, dtype=dtype).view(1, 3, 1, 1)

    while not training_done:
        for batch in dataloader:
            with accelerator.accumulate([vae_aligner]):
                vae_aligner.train()
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype)
                x_intern = (pixel_values - imagenet_mean) / imagenet_std
                x_vae = pixel_values * 2 - 1

                with torch.no_grad():
                    x_clip = extract_feature_pre_adapter(vision_model, x_intern)
                    vae_latent = vae.encode(x_vae).latent_dist.sample().to(dtype)

                rec_latent = vae_aligner(x_clip).to(dtype)
                loss_mse = torch.nn.functional.mse_loss(rec_latent, vae_latent)

                accelerator.backward(loss_mse)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_learn, 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                    global_step += 1
                    progress_bar.update(1)

                    logs = dict(
                        loss_mse  = accelerator.gather(loss_mse.detach()).mean().item(),
                    )
                    accelerator.log(logs, step=global_step)
                    progress_bar.set_postfix(**logs)

                    if global_step > config.train.num_iter:
                        training_done = True
                        break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/vae_aligner/intern_8b_aligner.yaml")
    args = parser.parse_args()
    main(args)