import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import pprint
import argparse
import torchvision.transforms as pth_transforms
from tqdm import tqdm
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from accelerate.state import AcceleratorState

from PIL import Image
from diffusers import AutoencoderDC, SanaTransformer2DModel
from util.intern_dataloader import get_intern_dataloader
from util.misc import process_pretrained_model_path, flatten_dict
from model.internvl.modeling_internvl_chat import InternVLChatModel
from model.diffusion_decoder.sana_decoder import SanaDecoder

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def get_accelerator(config):
    output_dir = os.path.join(config.root, config.exp_name, config.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    logging_dir = os.path.join(output_dir, config.logging_dir)
    project_config = ProjectConfiguration(project_dir=config.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        log_with                    = None if config.report_to == "no" else config.report_to,
        mixed_precision             = config.mixed_precision,
        project_config              = project_config,
        gradient_accumulation_steps = config.gradient_accumulation_steps,
    )

    return accelerator, output_dir


def main(args):
    config = OmegaConf.load(args.config)
    config = process_pretrained_model_path(config)
    accelerator, output_dir = get_accelerator(config.train)
    accelerator.print(pprint.pformat(OmegaConf.to_container(config, resolve=True), indent=2, width=120).strip('{}'))
    AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = config.data.batch_size
    accelerator.print(AcceleratorState().deepspeed_plugin.deepspeed_config)

    # load models
    # vae = AutoencoderDC.from_pretrained(config.sana0_6b_path, subfolder="vae")
    # vae.requires_grad_(False)
    sana_decoder = SanaDecoder(config)

    if config.base_model == "intern_vl_1b":
        vision_model = InternVLChatModel.from_pretrained(config.intern_vl_1b_path).vision_model
    elif config.base_model == "intern_vl_2b":
        vision_model = InternVLChatModel.from_pretrained(config.intern_vl_2b_path).vision_model
    elif config.base_model == "intern_vl_8b":
        vision_model = InternVLChatModel.from_pretrained(config.intern_vl_8b_path).vision_model
    vision_model.requires_grad_(False)

    # transformer = SanaTransformer2DModel.from_pretrained(
    #     config.sana0_6b_path, subfolder="transformer", torch_dtype=torch.bfloat16
    # )

    if config.train.resume_path is not None:
        raise NotImplementedError
        # ckpt = torch.load(config.train.resume_path, map_location="cpu", weights_only=True)
        # m, u = transformer.load_state_dict(ckpt, strict=False)
        # print(f"missing keys: {m}")
        # print(f"unexpected keys: {u}")
        # accelerator.print(f"transformer loaded from {config.train.resume_path}")

    global_step = config.train.global_step if config.train.global_step is not None else 0
    params_to_learn = list(p for p in sana_decoder.parameters() if p.requires_grad)

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

    dataloader = get_intern_dataloader(config.data, accelerator)
    sana_decoder, optimizer = accelerator.prepare(sana_decoder, optimizer)
    vision_model.requires_grad_(False)
    vision_model = vision_model.to(accelerator.device, dtype).eval()
    vae = vae.to(accelerator.device, dtype).eval()

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
    accelerator.print(f"vae_aligner dtype: {next(sana_decoder.parameters()).dtype}")
    accelerator.print(f"Accelerator mixed precision: {accelerator.mixed_precision}")

    imagenet_mean = torch.tensor(IMAGENET_MEAN, device=accelerator.device, dtype=dtype).view(1, 3, 1, 1)
    imagenet_std = torch.tensor(IMAGENET_STD, device=accelerator.device, dtype=dtype).view(1, 3, 1, 1)

    while not training_done:
        for batch in dataloader:
            with accelerator.accumulate([sana_decoder]):
                sana_decoder.train()
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype)
                pixel_values_clip = (pixel_values - imagenet_mean) / imagenet_std
                pixel_values_vae = pixel_values * 2 - 1

                with torch.no_grad():
                    x_clip = vision_model(
                        pixel_values         = pixel_values_clip,
                        output_hidden_states = False,
                        return_dict          = True
                    ).last_hidden_state[:, 1:, :] # (B, 1024, 1024)
                    x_vae = sana_decoder.vae.encode(pixel_values_vae).latent
                
                accelerator.print(f"{x_clip.shape=}")
                accelerator.print(f"{x_vae.shape=}")
                break



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/diffusion_decoder/sana_1024.yaml")
    args = parser.parse_args()
    main(args)

    # vae_id = "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers"
    # device = "cuda:0"
    # dtype = torch.float16

    # vae = AutoencoderDC.from_pretrained(vae_id, subfolder="vae")
    # vae.requires_grad_(False)
    



    # vae = vae.to(device, dtype).eval()

    # vae_transform = pth_transforms.Compose([
    #     pth_transforms.Resize(448, max_size=None),
    #     pth_transforms.CenterCrop(448),
    #     pth_transforms.ToTensor(),
    # ])
    # img = Image.open("/data/phd/jinjiachun/codebase/ideal-octo-spork/asset/internet/letter.jpeg").convert("RGB")

    # x = vae_transform(img).unsqueeze(0).to(device, dtype)
    # x = x * 2 - 1
    # print(f"{x.shape=}")

    # latents = vae.encode(x).latent
    # latents = latents * vae.config.scaling_factor
    # print(f"{latents.shape=}")

    # samples = vae.decode(latents).sample
    # print(f"{samples.shape=}")
    # samples = samples / vae.config.scaling_factor
    # samples = (samples + 1) / 2
    # samples = samples.clamp(0, 1)

    # import torchvision.utils as vutils
    # vutils.save_image(samples, "dim_1024.png", nrow=1, normalize=False)


