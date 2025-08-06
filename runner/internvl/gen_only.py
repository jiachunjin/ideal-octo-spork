import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import pprint
import argparse

from tqdm import tqdm
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration

from util.misc import process_pretrained_model_path, flatten_dict
from model.vae_aligner import get_vae_aligner
from model.dit.diff_mlp import add_diffhead_to_ar_model
from model.internvl.modeling_internvl_chat import InternVLChatModel
from util.intern_dataloader import get_intern_dataloader

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
    accelerator.print("Configuration:")
    accelerator.print(pprint.pformat(OmegaConf.to_container(config, resolve=True), indent=2, width=120).strip('{}'))
    AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = config.data.batch_size
    accelerator.print(AcceleratorState().deepspeed_plugin.deepspeed_config)

    # load models
    vae_aligner = get_vae_aligner(config.vae_aligner)
    ckpt = torch.load(config.vae_aligner.pretrained_path, map_location="cpu", weights_only=True)
    vae_aligner.load_state_dict(ckpt, strict=True)
    vae_aligner_projector = vae_aligner.siglip_feature_proj

    ar_model = InternVLChatModel.from_pretrained(config.intern_vl_1b_path)
    ar_model, train_scheduler = add_diffhead_to_ar_model(ar_model, config.model)
    # clip = ar_model.vision_model

    if config.train.diffhead_resume_path is not None:
        raise NotImplementedError
    if config.train.clip_projector_resume_path is not None:
        raise NotImplementedError
    if config.train.backbone_resume_path is not None:
        raise NotImplementedError

    global_step = config.train.global_step if config.train.global_step is not None else 0
    params_to_learn = list(p for p in ar_model.parameters() if p.requires_grad)

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
    ar_model, optimizer = accelerator.prepare(ar_model, optimizer)
    
    vae_aligner_projector.requires_grad_(False)
    vae_aligner_projector = vae_aligner_projector.to(accelerator.device, dtype).eval()

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
    accelerator.print(f"vae_aligner dtype: {next(vae_aligner.parameters()).dtype}")
    accelerator.print(f"Accelerator mixed precision: {accelerator.mixed_precision}")

    imagenet_mean = torch.tensor(IMAGENET_MEAN, device=accelerator.device, dtype=dtype).view(1, 3, 1, 1)
    imagenet_std = torch.tensor(IMAGENET_STD, device=accelerator.device, dtype=dtype).view(1, 3, 1, 1)

    while not training_done:
        for batch in dataloader:
            with accelerator.accumulate([ar_model]):
                ar_model.train()
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype)
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]

                pixel_values = (pixel_values - imagenet_mean) / imagenet_std

                with torch.no_grad():
                    x_clip = ar_model.extract_feature(pixel_values)
                    x_siglip_dimdown = vae_aligner_projector(x_clip)
                    visual_gen_feature = x_siglip_dimdown

                print(pixel_values.shape, input_ids.shape, attention_mask.shape, visual_gen_feature.shape)

                exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/internvl/gen_only.yaml")
    args = parser.parse_args()
    main(args)