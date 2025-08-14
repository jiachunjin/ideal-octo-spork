import os
import math
import torch
import argparse
from tqdm import tqdm
from omegaconf import OmegaConf

from diffusers import DDPMScheduler
from model.dit.standard_dit import DiT
from util.misc import process_pretrained_model_path, flatten_dict
from model.internvl.modeling_internvl_chat import InternVLChatModel


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def get_accelerator(config):
    import pprint
    from accelerate import Accelerator
    from accelerate.state import AcceleratorState
    from accelerate.utils import ProjectConfiguration

    output_dir = os.path.join(config.train.root, config.train.exp_name, config.train.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    logging_dir = os.path.join(output_dir, config.train.logging_dir)
    project_config = ProjectConfiguration(project_dir=config.train.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        log_with                    = config.train.report_to,
        mixed_precision             = config.train.mixed_precision,
        project_config              = project_config,
        gradient_accumulation_steps = config.train.gradient_accumulation_steps,
    )
    AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = config.data.batch_size
    accelerator.print("Configuration:")
    accelerator.print(pprint.pformat(OmegaConf.to_container(config, resolve=True), indent=2, width=120).strip('{}'))

    return accelerator, output_dir

def get_wds_dataloader(config, accelerator):
    import glob
    import random
    import webdataset as wds
    import torchvision.transforms as pth_transforms

    urls = []
    for path in config.wds_path:
        urls.extend(glob.glob(os.path.join(path, "*.tar")))
    accelerator.print(f"Found tar files: {len(urls)}")

    pre_transform = pth_transforms.Compose([
        pth_transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        pth_transforms.Resize(config.img_size, max_size=None),
        pth_transforms.CenterCrop(config.img_size),
        pth_transforms.ToTensor(),
    ])

    def preprocess_image(image):
        width, height = image.size
        max_size = max(width, height)
        if max_size < config.img_size * 0.25:
            return None
        pixel_values = pre_transform(image)

        return pixel_values
    
    def preprocess_label(label):
        if random.random() < config.cfg_drop_rate:
            label = 1000
        label = torch.tensor(label, dtype=torch.long)

        return label

    def collation_fn(batch):
        pixel_values = []
        labels = []

        for sample in batch:
            x, y = sample
            if x == None:
                continue
            else:
                pixel_values.append(x)
                labels.append(y)

        pixel_values = torch.stack(pixel_values)
        labels = torch.stack(labels)

        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }

    pipeline = [
        wds.ResampledShards(urls),
        wds.tarfile_to_samples(handler=wds.warn_and_continue),
        wds.shuffle(bufsize=config.buffer_size, initial=config.buffer_size),
        wds.decode("pil", handler=wds.ignore_and_continue),
        wds.to_tuple("jpg", "cls"),
        wds.map_tuple(preprocess_image, preprocess_label),
        wds.batched(config.batch_size, partial=preprocess_label, collation_fn=collation_fn),
    ]

    num_train_examples = config.num_train_examples
    global_batch_size = config.batch_size * accelerator.num_processes
    num_workers_per_gpu = config.num_workers

    num_worker_batches = math.ceil(num_train_examples / 
        (global_batch_size * num_workers_per_gpu))
    
    accelerator.print(f"num_worker_batches: {num_worker_batches}")

    train_dataset = wds.DataPipeline(*pipeline).with_epoch(num_worker_batches)

    dataloader = wds.WebLoader(
        train_dataset,
        batch_size  = None,
        num_workers = config.num_workers,
        pin_memory  = True,
    )

    return dataloader

def main(args):
    config = OmegaConf.load(args.config)
    config = process_pretrained_model_path(config)

    accelerator, output_dir = get_accelerator(config)

    dit_model = DiT(config.dit)
    vision_model = InternVLChatModel.from_pretrained(config.intern_vl_8b_path).vision_model
    vision_model.requires_grad_(False)

    if config.train.resume_path is not None:
        raise NotImplementedError("Resume is not implemented")

    params_to_learn = list(p for p in dit_model.parameters() if p.requires_grad)
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
    dit_model, optimizer = accelerator.prepare(dit_model, optimizer)
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

    train_scheduler = DDPMScheduler(
        beta_schedule          = "scaled_linear",
        beta_start             = 0.00085,
        beta_end               = 0.012,
        num_train_timesteps    = 1000,
        clip_sample            = False,
        prediction_type        = "v_prediction",
        steps_offset           = 1,
        trained_betas          = None,
        timestep_spacing       = "trailing",
        rescale_betas_zero_snr = True
    )


    while not training_done:
        for batch in dataloader:
            with accelerator.accumulate([dit_model]):
                dit_model.train()
                x = batch["pixel_values"].to(accelerator.device, dtype)
                y = batch["labels"].to(accelerator.device)
                x = (x - imagenet_mean) / imagenet_std
                with torch.no_grad():
                    x_clip = vision_model(
                        pixel_values         = x,
                        output_hidden_states = False,
                        return_dict          = True
                    ).last_hidden_state[:, 1:, :]

                print(x_clip.mean(), x_clip.std(), x_clip.min(), x_clip.max())
                B = x_clip.shape[0]
                timesteps = torch.randint(0, 1000, (B,), device=accelerator.device, dtype=torch.int64)
                noise = torch.randn_like(x_clip, device=accelerator.device, dtype=dtype)
                noisy_latents = train_scheduler.add_noise(x_clip, noise, timesteps)
                target = train_scheduler.get_velocity(x_clip, noise, timesteps)
                pred = dit_model(noisy_latents, timesteps, y)
                loss = torch.nn.functional.mse_loss(pred, target)

                accelerator.print(loss.item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/clip_dit/clip_dit.yaml")
    args = parser.parse_args()
    main(args)
