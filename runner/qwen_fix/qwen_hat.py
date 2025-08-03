import os
import torch
import pprint
import argparse

from tqdm import tqdm
from einops import rearrange
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

from model.vae_aligner import get_vae_aligner
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from model.janus.models import MultiModalityCausalLM, VLChatProcessor
from model.dit.diff_mlp import equip_diffhead_query_with_janus
from util.misc import process_pretrained_model_path, flatten_dict

qwen_vl_path = "/data/phd/jinjiachun/ckpt/Qwen/Qwen2.5-VL-3B-Instruct"

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

import glob
import webdataset as wds
import torchvision.transforms as pth_transforms
from torch.utils.data import DataLoader
def get_qwen_dataloader(config):
    processor = AutoProcessor.from_pretrained(qwen_vl_path)

    config = config.data
    urls = []
    for path in config.generation.wds_path:
        urls.extend(glob.glob(os.path.join(path, "*.tar")))

    print(f"Found tar files: {len(urls)}")

    preprocess_gen = pth_transforms.Compose([
        pth_transforms.Resize(config.generation.img_size, max_size=None),
        pth_transforms.CenterCrop(config.generation.img_size),
        pth_transforms.ToTensor(),
        pth_transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
    ])

    def process_img(image):
        try:
            if image is None:
                return None
            transformed_image = preprocess_gen(image)
            return {"pixel_value": transformed_image}
        except Exception as e:
            print(f"process_img error: {e}")
            return None

    def process_prompt(prompt):
        if prompt is None or not isinstance(prompt, str):
            return None
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Generate an image: " + prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )+"<|vision_start|>"

        tokenizer_output = processor.tokenizer(
            text,
            return_tensors = "pt",
            padding        = "max_length",
            padding_side   = "left",
            truncation     = True,
            max_length     = config.max_seq_length - config.num_img_token,
        )

        input_ids = tokenizer_output["input_ids"]
        attention_mask = tokenizer_output["attention_mask"]

        return {"input_ids": input_ids, "attention_mask": attention_mask}


    def collate_fn_gen(batch):
        """处理batch, 过滤掉None值"""
        valid_batches = []
        for item in batch:
            if item is not None and len(item) == 2:
                img_data, prompt_data = item
                if img_data is not None and prompt_data is not None:
                    valid_batches.append(item)
        
        if len(valid_batches) == 0:
            # 如果没有有效数据，返回空batch
            return {
                "pixel_value": torch.empty(0, 3, config.generation.img_size, config.generation.img_size),
                "input_ids": torch.empty(0, config.max_seq_length - config.num_img_token),
                "attention_mask": torch.empty(0, config.max_seq_length - config.num_img_token),
            }
        
        # 分离图像和文本数据
        img_batches = [item[0] for item in valid_batches]
        prompt_batches = [item[1] for item in valid_batches]
        
        # 堆叠数据
        pixel_values = torch.stack([item["pixel_value"] for item in img_batches])
        input_ids = torch.cat([item["input_ids"] for item in prompt_batches], dim=0)
        attention_mask = torch.cat([item["attention_mask"] for item in prompt_batches], dim=0)
        
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    gen_wds_dataset = (
        wds.WebDataset(urls, resampled=True, shardshuffle=True, nodesplitter=None)
        .shuffle(config.generation.buffer_size, initial=config.generation.buffer_size)
        .decode("pil", handler=wds.ignore_and_continue)
        .to_tuple("jpg", "txt")
        .map_tuple(process_img, process_prompt)
    )

    dataloader = DataLoader(
        gen_wds_dataset, 
        batch_size  = config.generation.batch_size,
        num_workers = config.generation.num_workers,
        collate_fn  = collate_fn_gen
    )

    return dataloader


def main(args):
    config = OmegaConf.load(args.config)
    config = process_pretrained_model_path(config)
    accelerator, output_dir = get_accelerator(config.train)
    accelerator.print("Configuration:")
    accelerator.print(pprint.pformat(OmegaConf.to_container(config, resolve=True), indent=2, width=120).strip('{}'))

    # load models
    pad_token_id = 151643
    vae_aligner = get_vae_aligner(config.vae_aligner)
    ckpt = torch.load(config.vae_aligner.pretrained_path, map_location="cpu", weights_only=True)
    vae_aligner.load_state_dict(ckpt, strict=True)
    vae_aligner_projector = vae_aligner.siglip_feature_proj

    janus = MultiModalityCausalLM.from_pretrained(config.janus_1b_path, trust_remote_code=True)
    qwen_vl = Qwen2_5_VLForConditionalGeneration.from_pretrained(qwen_vl_path)

    siglip = janus.vision_model
    vae_aligner_projector.requires_grad_(False)
    siglip.requires_grad_(False)

    qwen_vl_plus, train_scheduler = modify_qwen_vl(qwen_vl, config)