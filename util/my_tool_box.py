def get_accelerator(config):
    import os
    import pprint
    from omegaconf import OmegaConf
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
    import os
    import glob
    import math
    import torch
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
        if max_size < config.img_size:
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

def get_t2i_dataloader(config, accelerator):
    import os
    import glob
    import math
    import torch
    import random
    import webdataset as wds
    import torchvision.transforms as pth_transforms
    
    from transformers import AutoTokenizer
    from model.internvl.conversation import get_conv_template

    intern_vl_1b_path = "/data/phd/jinjiachun/ckpt/OpenGVLab/InternVL3-1B"
    tokenizer = AutoTokenizer.from_pretrained(intern_vl_1b_path, trust_remote_code=True, use_fast=False)
    IMG_START_TOKEN = "<img>"

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
        if max_size < config.img_size * 0.75:
            return None
        pixel_values = pre_transform(image)

        return pixel_values
    
    def preprocess_text(text):
        if random.random() < config.cfg_drop_rate:
            text = ""
        
        if config.use_template:
            template = get_conv_template("internvl2_5")
            prompt = f"Generate an image: {text}"

            template.append_message(template.roles[0], prompt)
            template.append_message(template.roles[1], None)
            prompt = template.get_prompt() + IMG_START_TOKEN
        else:
            prompt = text

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
        wds.to_tuple("jpg", "txt"),
        wds.map_tuple(preprocess_image, preprocess_text),
        wds.batched(config.batch_size, partial=False, collation_fn=collation_fn),
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