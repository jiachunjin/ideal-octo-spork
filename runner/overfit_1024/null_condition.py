import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import argparse
from tqdm import tqdm
from einops import rearrange
from omegaconf import OmegaConf

from util.misc import process_pretrained_model_path, flatten_dict
from util.my_tool_box import get_accelerator, get_wds_dataloader
from model.internvl.modeling_internvl_chat import InternVLChatModel
# from model.dit.hybrid_dit import HybridDiT
from model.overfit_1024.hybrid_dit_class import HybridDiT_Class
from model.internvl import extract_feature_pre_adapter

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def main(args):
    config = OmegaConf.load(args.config)
    config = process_pretrained_model_path(config)
    accelerator, output_dir = get_accelerator(config)

    vision_model = InternVLChatModel.from_pretrained(config.intern_vl_8b_path).vision_model
    model = HybridDiT_Class(config.hybrid_dit)

    params_to_learn = list(p for p in model.parameters() if p.requires_grad)
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

    dataloader = get_wds_dataloader(config.data, accelerator)
    model, optimizer = accelerator.prepare(model, optimizer)
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
            with accelerator.accumulate([model]):
                model.train()
                x = batch["pixel_values"].to(accelerator.device, dtype)
                y = batch["labels"].to(accelerator.device)
                x = (x - imagenet_mean) / imagenet_std

                with torch.no_grad():
                    B = x.shape[0]
                    x_clip_condensed = extract_feature_pre_adapter(vision_model, x) # (B, 256, 4096)
                    x_clip = rearrange(x_clip_condensed, "b t (s d) -> b (t s) d", s=4, d=1024) # (B, 1024, 1024)

                x_t, target, timesteps = model.block_wise_noising(x_clip)

                pred = model(x_clip, x_t, timesteps, y)

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

                    if global_step > 0 and global_step % config.train.save_every == 0 and accelerator.is_main_process:
                        model.eval()
                        state_dict = accelerator.unwrap_model(model).state_dict()
                        save_path = os.path.join(output_dir, f"hybrid_dit-{config.train.exp_name}-{global_step}")
                        torch.save(state_dict, save_path)
                        print(f"hybrid_dit saved to {save_path}")

                    accelerator.wait_for_everyone()

        epoch += 1
        accelerator.print(f"epoch {epoch}: finished")
        accelerator.log({"epoch": epoch}, step=global_step)

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/overfit_1024/null_condition.yaml")
    args = parser.parse_args()
    main(args)