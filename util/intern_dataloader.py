import os
import glob
import math
import random
import torch
import webdataset as wds
import torchvision.transforms as pth_transforms
from transformers import AutoTokenizer
from torch.utils.data import default_collate

from model.internvl.conversation import get_conv_template

IMG_START_TOKEN="<img>"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

intern_vl_1b_path = "/data/phd/jinjiachun/ckpt/OpenGVLab/InternVL3-1B"
tokenizer = AutoTokenizer.from_pretrained(intern_vl_1b_path, trust_remote_code=True, use_fast=False)

# def get_null_prompt():
#     template = get_conv_template("internvl2_5")
#     prompt = "Generate an image: "
#     template.append_message(template.roles[0], prompt)
#     template.append_message(template.roles[1], None)
#     prompt = template.get_prompt() + IMG_START_TOKEN

#     tokenizer_output = tokenizer(
#         prompt,
#         return_tensors = "pt",
#         padding        = "max_length",
#         padding_side   = "left",
#         truncation     = True,
#         max_length     = config.max_seq_length - config.num_img_token,
#     )
#     input_ids = tokenizer_output["input_ids"]
#     attention_mask = tokenizer_output["attention_mask"]

#     return input_ids, attention_mask

def get_intern_dataloader(config, accelerator):
    urls = []
    for path in config.wds_path:
        urls.extend(glob.glob(os.path.join(path, "*.tar")))
    accelerator.print(f"Found tar files: {len(urls)}")

    preprocess_gen = pth_transforms.Compose([
        pth_transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        pth_transforms.Resize(config.img_size, max_size=None),
        pth_transforms.CenterCrop(config.img_size),
        pth_transforms.ToTensor(),
        # pth_transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    def preprocess_image(image):
        width, height = image.size
        max_size = max(width, height)
        if max_size < config.img_size * 0.75:
            return None
        pixel_values = preprocess_gen(image)

        return pixel_values
    
    def preprocess_text(text):
        if random.random() < config.cfg_drop_rate:
            text = ""

        template = get_conv_template("internvl2_5")
        prompt = f"Generate an image: {text}"

        template.append_message(template.roles[0], prompt)
        template.append_message(template.roles[1], None)
        prompt = template.get_prompt() + IMG_START_TOKEN

        tokenizer_output = tokenizer(
            prompt,
            return_tensors = "pt",
            padding        = "max_length",
            padding_side   = "left",
            truncation     = True,
            max_length     = config.max_seq_length - config.num_img_token,
        )
        input_ids = tokenizer_output["input_ids"]
        attention_mask = tokenizer_output["attention_mask"]

        return input_ids, attention_mask
    
    def collation_fn(batch):
        pixel_values = []
        input_ids_list = []
        attention_mask_list = []

        for sample in batch:
            pixel_value, (input_ids, attention_mask) = sample
            if pixel_value == None:
                print("image too small, skip")
                continue
            else:
                pixel_values.append(pixel_value)
                input_ids_list.append(input_ids[0])
                attention_mask_list.append(attention_mask[0])

        pixel_values = torch.stack(pixel_values)
        input_ids = torch.stack(input_ids_list)
        attention_mask = torch.stack(attention_mask_list)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    pipeline = [
        wds.ResampledShards(urls),
        wds.tarfile_to_samples(handler=wds.warn_and_continue),
        wds.shuffle(bufsize=config.buffer_size, initial=config.buffer_size),
        wds.decode("pil", handler=wds.ignore_and_continue),
        # wds.to_tuple("jpg", "cls"),
        wds.to_tuple("jpg", "txt"),
        wds.map_tuple(preprocess_image, preprocess_text),
        wds.batched(config.batch_size, partial=False, collation_fn=collation_fn),
    ]

    # num_train_examples = 1281167 + 50000
    num_train_examples = 35000000
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


if __name__ == "__main__":
    from omegaconf import OmegaConf
    from accelerate import Accelerator
    from util.misc import process_pretrained_model_path
    intern_vl_1b_path = "/data/phd/jinjiachun/ckpt/OpenGVLab/InternVL3-1B"
    tokenizer = AutoTokenizer.from_pretrained(intern_vl_1b_path, trust_remote_code=True, use_fast=False)

    config = OmegaConf.load("/data/phd/jinjiachun/codebase/ideal-octo-spork/config/internvl/gen_only.yaml")
    config = process_pretrained_model_path(config)
    accelerator = Accelerator()
    config.data.batch_size = 4
    config.data.max_seq_length = 512
    config.data.num_img_token = 256
    config.data.buffer_size = 100
    dataloader = get_intern_dataloader(config.data, accelerator)

    for batch in dataloader:
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        print(pixel_values.shape, input_ids.shape, attention_mask.shape)
        print(tokenizer.decode(input_ids[0], skip_special_tokens=False))
        print(attention_mask[0])
        ...