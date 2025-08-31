import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import argparse
from tqdm import tqdm
from omegaconf import OmegaConf
from einops import rearrange

from model.dit.diff_mlp import add_diffhead_dit_to_ar_model
from model.internvl.modeling_internvl_chat import InternVLChatModel
from model.vae_aligner.vit_vae_aligner import get_feature_down_proj
from model.internvl import extract_dual_clip

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

    internvl = InternVLChatModel.from_pretrained(config.intern_vl_8b_path)
    internvl, train_scheduler = add_diffhead_dit_to_ar_model(internvl, config.model)

    feature_down_projector = get_feature_down_proj(config.feature_down_projector)
    ckpt = torch.load(config.feature_down_projector.ckpt, map_location="cpu", weights_only=True)
    ckpt = {k.replace("feature_down_projector.", ""): v for k, v in ckpt.items() if "feature_down_projector" in k}
    feature_down_projector.load_state_dict(ckpt, strict=True)

    # vision_model = InternVLChatModel.from_pretrained(config.intern_vl_8b_path).vision_model
    # vision_model.requires_grad_(False)

    global_step = config.train.global_step if config.train.global_step is not None else 0
    params_to_learn = list(p for p in internvl.parameters() if p.requires_grad)

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
    internvl, optimizer = accelerator.prepare(internvl, optimizer)
    feature_down_projector.requires_grad_(False)
    feature_down_projector = feature_down_projector.to(accelerator.device, dtype).eval()

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
            with accelerator.accumulate([internvl]):
                internvl.train()
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype)
                input_ids = batch["input_ids"].to(accelerator.device)
                attention_mask = batch["attention_mask"].to(accelerator.device)

                pixel_values = (pixel_values - imagenet_mean) / imagenet_std

                with torch.no_grad():
                    clip_1024, clip_256 = extract_dual_clip(internvl.vision_model, pixel_values)
                    visual_gen_feature = feature_down_projector(clip_256) # (B, 256, d)

                # ----- compute AR loss -----
                B, L = input_ids.shape
                text_embedding = internvl.language_model.get_input_embeddings()(input_ids).clone()
                img_embedding = internvl.clip_projector(visual_gen_feature)
                print(text_embedding.shape, img_embedding.shape)
                joint_embedding = torch.cat((text_embedding, img_embedding), dim=1)
                img_mask = torch.ones((B, config.data.num_img_token), dtype=torch.bool, device=accelerator.device)
                attention_mask = torch.cat([attention_mask, img_mask], dim=1)

                hidden_states = internvl.language_model(
                    inputs_embeds        = joint_embedding,
                    attention_mask       = attention_mask,
                    output_hidden_states = True,
                ).hidden_states[-1]
                hidden_state = hidden_states[:, -config.data.num_img_token-1:-1, :]
                print(hidden_state.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/yuqi/joint_hat.yaml")
    args = parser.parse_args()
    main(args)