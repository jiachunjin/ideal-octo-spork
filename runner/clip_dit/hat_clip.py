import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
from omegaconf import OmegaConf
from einops import rearrange

from model.dit.dit_head import equip_internvl
from model.internvl.modeling_internvl_chat import InternVLChatModel
from model.internvl import extract_both_clip
from util.my_tool_box import get_accelerator, get_t2i_dataloader
from util.misc import process_pretrained_model_path, flatten_dict



torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def main(args):
    config = OmegaConf.load(args.config)
    config = process_pretrained_model_path(config)

    accelerator, output_dir = get_accelerator(config)

    internvl = InternVLChatModel.from_pretrained(config.intern_vl_8b_path)
    internvl, train_scheduler = equip_internvl(internvl, config.model)

    params_to_learn = list(p for p in internvl.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(
        params_to_learn,
        lr           = config.train.lr,
        betas        = (0.9, 0.95),
        weight_decay = 5e-2,
        eps          = 1e-8,
    )

    global_step = config.train.global_step if config.train.global_step is not None else 0
    
    if accelerator.mixed_precision == "bf16":
        dtype = torch.bfloat16
    elif accelerator.mixed_precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    dataloader = get_t2i_dataloader(config.data, accelerator)
    internvl, optimizer = accelerator.prepare(internvl, optimizer)

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
                x = batch["pixel_values"].to(accelerator.device, dtype)
                input_ids = batch["input_ids"].to(accelerator.device)
                attention_mask = batch["attention_mask"].to(accelerator.device)
                x = (x - imagenet_mean) / imagenet_std

                with torch.no_grad():
                    clip_1024, clip_256 = extract_both_clip(internvl.vision_model, x)

                    text_embedding = internvl.language_model.get_input_embeddings()(input_ids)
                    # print(clip_256.shape)
                    visual_embedding = internvl.mlp1(clip_256)
                    joint_embedding = torch.cat((text_embedding, visual_embedding), dim=1)

                    img_mask = torch.ones((clip_256.shape[0], 256), dtype=torch.bool, device=accelerator.device)
                    attention_mask = torch.cat([attention_mask, img_mask], dim=1)

                hidden_states = internvl.language_model(
                    inputs_embeds        = joint_embedding,
                    attention_mask       = attention_mask,
                    output_hidden_states = True,
                ).hidden_states[-1][:, -256:, :]

                print(hidden_states.shape) # (B, 256, 3584)

                x_clip = rearrange(clip_1024, "B (J K) D -> (B J) K D", H=256, W=4) # (Bx256, 4, 1024)
                condition = rearrange(hidden_states, "B L D -> (B L) D") # (Bx256, 3584)
                timesteps = torch.randint(0, 1000, (x_clip.shape[0],), device=accelerator.device, dtype=torch.int64) # (Bx256,)
                noise = torch.randn_like(x_clip, device=accelerator.device, dtype=dtype)
                x_noisy = train_scheduler.add_noise(x_clip, noise, timesteps)
                target = train_scheduler.get_velocity(x_clip, noise, timesteps)

                pred = internvl.diff_head(x_noisy, timesteps, hidden_states)
                print(pred.shape, target.shape)
                loss = torch.nn.functional.mse_loss(pred, target)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_learn, 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                    global_step += 1
                    progress_bar.update(1)

                    logs = dict(
                        clip_loss = accelerator.gather(loss.detach()).mean().item(), 
                    )
                    accelerator.log(logs, step=global_step)
                    progress_bar.set_postfix(**logs)

# @torch.no_grad()
# def dev():
#     device = torch.device("cuda:0")
#     dtype = torch.float16
#     config = OmegaConf.load("config/clip_dit/hat_clip.yaml")
    
#     intern_vl_8b_path = "/data/phd/jinjiachun/ckpt/OpenGVLab/InternVL3-8B"
#     internvl = InternVLChatModel.from_pretrained(intern_vl_8b_path)

#     internvl, train_scheduler = equip_internvl(internvl, config.model)
#     internvl = internvl.to(device, dtype)

#     params_to_learn = list(p for p in internvl.parameters() if p.requires_grad)
#     print(f"Learnable parameters: {sum(p.numel() for p in params_to_learn) / 1e6} M")

#     joint_embedding = torch.randn(1, 256, 3584, device=device, dtype=dtype)
#     attention_mask = torch.ones(1, 256, device=device, dtype=torch.bool)

#     hidden_states = internvl.language_model(
#         inputs_embeds        = joint_embedding,
#         attention_mask       = attention_mask,
#         output_hidden_states = True,
#     ).hidden_states[-1]

#     print(hidden_states.shape)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/clip_dit/hat_clip.yaml")
    args = parser.parse_args()
    main(args)