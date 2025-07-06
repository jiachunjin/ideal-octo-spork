def process_path_for_different_machine(config):
    if config.machine == "g3":
        config.train.root = "/data1/jjc/experiment"
        config.janus_path = "/data1/ckpts/deepseek-ai_/Janus-Pro-1B"
        config.vae_path = "/data1/ckpts/black-forest-labs/FLUX.1-dev/vae"
        if config.data.name == "imagenet_wds":
            config.data.train_path = "/data1/LargeData/timm/imagenet-1k-wds"
        elif config.data.name == "hybrid":
            config.data.train_path = ["/data1/LargeData/timm/imagenet-1k-wds"]

    elif config.machine == "ks":
        config.train.root = "/data/phd/jinjiachun/experiment"
        config.janus_path = "/data/phd/jinjiachun/ckpt/deepseek-ai/Janus-Pro-1B"
        config.vae_path = "/data/phd/jinjiachun/ckpt/black-forest-labs/FLUX.1-dev/vae"
        if config.data.name == "imagenet_wds":
            config.data.train_path = "/data/phd/jinjiachun/dataset/timm/imagenet-1k-wds"
        elif config.data.name == "hybrid":
            config.data.train_path = [
                                    #   "/data/phd/jinjiachun/dataset/timm/imagenet-1k-wds",
                                    #   "/data/phd/jinjiachun/dataset/timm/imagenet-22k-wds",
                                      "/data/phd/jinjiachun/dataset/BLIP3o/BLIP3o-Pretrain-Long-Caption",
                                      "/data/phd/jinjiachun/dataset/BLIP3o/BLIP3o-Pretrain-Short-Caption",
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