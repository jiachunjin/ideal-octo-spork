import os
import glob
import webdataset as wds
import torchvision.transforms as pth_transforms


def get_imagenet_dataloader(config, accelerator):
    urls = []
    for path in config.wds_path:
        urls.extend(glob.glob(os.path.join(path, "*.tar")))
    accelerator.print(f"Found tar files: {len(urls)}")

    preprocess_gen = pth_transforms.Compose([
        pth_transforms.Resize(config.generation.img_size, max_size=None),
        pth_transforms.CenterCrop(config.generation.img_size),
        pth_transforms.ToTensor(),
        pth_transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
    ])

    def preprocess_image(image):
        transformed_image = preprocess_gen(image)

        return {"pixel_value": transformed_image}

    wds_dataset = (
        wds.WebDataset(urls, resampled=True)
        .shuffle(config.generation.buffer_size, initial=config.generation.buffer_size)
        .split_by_node(rank=accelerator.process_index, world_size=accelerator.num_processes)
        .split_by_worker(worker_info=wds.worker_info)
        .decode("pil", handler=wds.ignore_and_continue)
        .to_tuple("jpg", "cls")
        .map_tuple(preprocess_image, None)
    )

    dataloader = wds.WebLoader(
        wds_dataset,
        batch_size      = config.batch_size,
        num_workers     = config.num_workers,
        pin_memory      = True,
        prefetch_factor = 8
    )

    return dataloader


if __name__ == "__main__":
    from omegaconf import OmegaConf
    from accelerate import Accelerator

    config = OmegaConf.load("../../config/qwen_fix/qwen_metaquery.yaml")
    accelerator = Accelerator()
    dataloader = get_imagenet_dataloader(config.data, accelerator)

    for batch in dataloader:
        print(batch)
        break