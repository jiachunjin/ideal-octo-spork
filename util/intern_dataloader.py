import os
import glob
import math
import webdataset as wds
import torchvision.transforms as pth_transforms
from torch.utils.data import default_collate

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def get_intern_dataloader(config, accelerator):
    urls = []
    for path in config.wds_path:
        urls.extend(glob.glob(os.path.join(path, "*.tar")))
    accelerator.print(f"Found tar files: {len(urls)}")

    preprocess_gen = pth_transforms.Compose([
        pth_transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
        pth_transforms.Resize(config.img_size, max_size=None),
        pth_transforms.CenterCrop(config.img_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    def preprocess_image(image):
        pixel_values = preprocess_gen(image)

        return pixel_values

    pipeline = [
        wds.ResampledShards(urls),
        wds.tarfile_to_samples(handler=wds.warn_and_continue),
        wds.shuffle(bufsize=config.buffer_size, initial=config.buffer_size),
        wds.decode("pil", handler=wds.ignore_and_continue),
        wds.to_tuple("jpg", "cls"),
        wds.map_tuple(preprocess_image, None),
        wds.batched(config.batch_size, partial=False, collation_fn=default_collate),
    ]

    num_train_examples = 1281167 + 50000
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

    config = OmegaConf.load("../../config/vae_aligner/intern_clip.yaml")
    accelerator = Accelerator()
    dataloader = get_intern_dataloader(config.data, accelerator)
    
    for x in dataloader:
        print(x.shape)
        break