import os
import io
import glob
import torch
from datasets import load_dataset, Features, Value
from torch.utils.data import DataLoader
from torchvision import transforms as pth_transforms
from PIL import Image, UnidentifiedImageError


def get_dataloader(config, accelerator=None):
    data_files = []
    for path in config.train_path:
        data_files.extend(glob.glob(os.path.join(path, "*.tar")))
    print(f"Found {len(data_files)} tar files")
    
    # 定义数据集特性，将图片字段设置为二进制类型，防止自动解码
    features = Features({
        "jpg": Value("binary"),  # 图片作为二进制数据，不自动解码
        "txt": Value("string"),  # 文本字段
    })
    
    # 为分布式训练设置不同的seed
    seed = None
    if accelerator is not None and accelerator.num_processes > 1:
        # 使用进程rank作为seed的一部分，确保每个进程看到不同的数据
        seed = 42 + accelerator.process_index
    
    dataset = load_dataset(
        "webdataset",
        data_files = data_files,
        split      = "train",
        streaming  = True,
        features   = features,  # 使用自定义特性
    )
    
    # 为streaming数据集设置seed
    if seed is not None:
        dataset = dataset.shuffle(seed=seed, buffer_size=10000)

    img_transform_train = pth_transforms.Compose([
        pth_transforms.Resize(config.img_size, max_size=None),
        pth_transforms.CenterCrop(config.img_size),
        pth_transforms.ToTensor(),
        pth_transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
    ])

    def decode_image(img_bytes):
        try:
            # 从二进制数据创建 PIL Image
            img = Image.open(io.BytesIO(img_bytes))
            img = img.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as e:
            print(f"Image decode error: {e}")
            raise ValueError("Corrupted or unsupported image, skip")
        width, height = img.size
        if min(width, height) < config.img_size:
            raise ValueError(f"Image too small: {width}x{height}, skip")
        pixel_value = img_transform_train(img)
        return pixel_value

    def collate_fn(batch):
        pixel_values = []
        texts = []
        
        for item in batch:
            try:
                # 获取原始二进制数据
                img_bytes = item["jpg"]
                pixel_value = decode_image(img_bytes)
                pixel_values.append(pixel_value)
            except Exception as e:
                if isinstance(e, ValueError) and ("Image too small" in str(e) or "Corrupted or unsupported image" in str(e)):
                    # 静默跳过已知的图片问题
                    pass
                else:
                    print(f"Unexpected error in collate_fn(): {e}")
                continue

        if len(pixel_values) == 0:
            print(f"No valid image in this batch, return an empty batch.")
            return {"pixel_values": torch.empty(0, 3, config.img_size, config.img_size), "texts": []}

        pixel_values = torch.stack(pixel_values)
        
        return {
            "pixel_values": pixel_values,
            "texts": texts,
        }

    dataloader = DataLoader(
        dataset,
        batch_size  = config.batch_size,
        shuffle     = False,  # 对于streaming数据集，shuffle在dataset层面处理
        num_workers = config.num_workers,
        pin_memory  = True,
        drop_last   = True,
        collate_fn  = collate_fn,
    )

    return dataloader