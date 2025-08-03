def process_pretrained_model_path(config):
    if config.machine == "g4":
        raise NotImplementedError("Not implemented for g4")
    elif config.machine == "ks":
        config.train.root = "/data/phd/jinjiachun/experiment"
        config.janus_1b_path = "/data/phd/jinjiachun/ckpt/deepseek-ai/Janus-Pro-1B"
        config.janus_7b_path = "/data/phd/jinjiachun/ckpt/deepseek-ai/Janus-Pro-7B"
        config.sd3_5_path = "/data/phd/jinjiachun/ckpt/stabilityai/stable-diffusion-3.5-medium"
        config.vae_path = "/data/phd/jinjiachun/ckpt/stabilityai/stable-diffusion-3.5-medium/vae"
        config.qwen_vl_path = "/data/phd/jinjiachun/ckpt/Qwen/Qwen2.5-VL-3B-Instruct"
    else:
        raise ValueError(f"Invalid machine: {config.machine}")
    return config

def process_path_for_different_machine(config):
    if config.machine == "g3":
        config.train.root = "/data1/jjc/experiment"
        config.janus_1b_path = "/data1/ckpts/deepseek-ai_/Janus-Pro-1B"
        config.janus_7b_path = None
        config.sd3_5_path = "/data1/ckpts/stabilityai/stable-diffusion-3.5-medium"
        config.vae_path = "/data1/ckpts/black-forest-labs/FLUX.1-dev/vae"
        if config.data.name == "imagenet_wds":
            config.data.train_path = "/data1/LargeData/timm/imagenet-1k-wds"
        elif config.data.name == "hybrid":
            config.data.train_path = ["/data1/LargeData/timm/imagenet-1k-wds"]
        elif config.data.name == "t2i":
            config.data.train_path = ["/data1/LargeData/BLIP3o-tmp"]
    elif config.machine == "ks":
        config.train.root = "/data/phd/jinjiachun/experiment"
        config.janus_1b_path = "/data/phd/jinjiachun/ckpt/deepseek-ai/Janus-Pro-1B"
        config.janus_7b_path = "/data/phd/jinjiachun/ckpt/deepseek-ai/Janus-Pro-7B"
        config.sd3_5_path = "/data/phd/jinjiachun/ckpt/stabilityai/stable-diffusion-3.5-medium"
        config.vae_path = "/data/phd/jinjiachun/ckpt/stabilityai/stable-diffusion-3.5-medium/vae"
        if config.data.name == "imagenet_wds":
            config.data.train_path = "/data/phd/jinjiachun/dataset/timm/imagenet-1k-wds"
        elif config.data.name == "hybrid":
            config.data.train_path = [
                "/data/phd/jinjiachun/dataset/timm/imagenet-1k-wds",
                "/data/phd/jinjiachun/dataset/timm/imagenet-22k-wds",
                "/data/phd/jinjiachun/dataset/BLIP3o/BLIP3o-Pretrain-Long-Caption",
                "/data/phd/jinjiachun/dataset/BLIP3o/BLIP3o-Pretrain-Short-Caption",
            ]
        elif config.data.name == "t2i":
            config.data.train_path = [
                "/data/phd/jinjiachun/dataset/BLIP3o/BLIP3o-Pretrain-Long-Caption",
                "/data/phd/jinjiachun/dataset/BLIP3o/BLIP3o-Pretrain-Short-Caption",
                "/data/phd/jinjiachun/dataset/BLIP3o/BLIP3o-Pretrain-JourneyDB",
            ]
    else:
        raise ValueError(f"Invalid machine: {config.machine}")

    return config

def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)