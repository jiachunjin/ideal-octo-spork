import os
import glob
import torch
from datasets import load_dataset, Features, Value
from datasets.features import Image as DatasetsImage
from torch.utils.data import DataLoader
from torchvision import transforms as pth_transforms
from PIL import Image, UnidentifiedImageError
from io import BytesIO

def get_dataloader(config):
    data_files = []
    for path in config.train_path:
        data_files.extend(glob.glob(os.path.join(path, "*.tar")))
    print(f"Found {len(data_files)} tar files")

    features = Features({
        "jpg": DatasetsImage(decode=False),
    })

    dataset = load_dataset(
        "webdataset",
        data_files = data_files,
        split      = "train",
        streaming  = True,
        features   = features,
    )

    img_transform_train = pth_transforms.Compose([
        pth_transforms.Resize(config.img_size, max_size=None),
        pth_transforms.CenterCrop(config.img_size),
        pth_transforms.ToTensor(),
        pth_transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
    ])

    def decode_image(img):
        try:
            if not isinstance(img, Image.Image):
                img = Image.open(BytesIO(img["bytes"]))  # 用字节流打开
            img.info.pop("exif", None)
            img = img.convert("RGB")
        except (UnicodeDecodeError, UnidentifiedImageError, OSError) as e:
            print(f"Image decode error: {e}")
            raise ValueError("Corrupted or non-UTF8 image, skip")
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
                pixel_value = decode_image(item["jpg"])
                pixel_values.append(pixel_value)
            except UnicodeDecodeError as e:
                print(f"UnicodeDecodeError in collate_fn(): {e}")
                print(f"Problematic item keys: {list(item.keys()) if hasattr(item, 'keys') else 'No keys'}")
                continue
            except Exception as e:
                if isinstance(e, ValueError) and "Image too small" in str(e):
                    pass
                else:
                    print(f"Error in collate_fn(): {e}")
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
        shuffle     = False,
        num_workers = config.num_workers,
        pin_memory  = True,
        drop_last   = True,
        collate_fn  = collate_fn,
    )

    return dataloader