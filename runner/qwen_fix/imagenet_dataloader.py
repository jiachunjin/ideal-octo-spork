import os
import glob
import webdataset as wds
import torchvision.transforms as pth_transforms
from torch.utils.data import default_collate

def get_imagenet_dataloader(config, accelerator):
    urls = []
    for path in config.wds_path:
        urls.extend(glob.glob(os.path.join(path, "*.tar")))
    accelerator.print(f"Found tar files: {len(urls)}")

    preprocess_gen = pth_transforms.Compose([
        pth_transforms.Resize(config.img_size, max_size=None),
        pth_transforms.CenterCrop(config.img_size),
        pth_transforms.ToTensor(),
        pth_transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
    ])

    def preprocess_image(image):
        transformed_image = preprocess_gen(image)

        return {"pixel_value": transformed_image}

    pipeline = [
        wds.ResampledShards(urls),
        wds.tarfile_to_samples(handler=wds.warn_and_continue),
        wds.shuffle(bufsize=5000, initial=1000),
        wds.decode("pil", handler=wds.ignore_and_continue),
        wds.to_tuple("jpg", "cls"),
        wds.map_tuple(preprocess_image, None),
        wds.batched(config.batch_size, partial=False, collation_fn=default_collate),
    ]

    train_dataset = wds.DataPipeline(*pipeline)


    # dataset = (
    #     wds.WebDataset(urls, resampled=True)
    #     .shuffle(1000)
    # )
    # # dataset = wds.split_by_node(dataset, rank=accelerator.process_index, world_size=accelerator.num_processes)
    # dataset = wds.split_by_worker(dataset, worker_info=wds.worker_info)
    # dataset = (
    #     dataset.decode("pil", handler=wds.warn_and_continue)
    #     .to_tuple("jpg", "cls")
    #     .map_tuple(preprocess_image, None)
    #     .batched(config.batch_size)
    # )
    # wds_dataset = (
    #     wds.WebDataset(urls, resampled=True)
    #     .shuffle(config.buffer_size, initial=config.buffer_size)
    #     .pipe(wds.split_by_node, rank=accelerator.process_index, world_size=accelerator.num_processes)
    #     .pipe(wds.split_by_worker, worker_info=wds.worker_info)
    #     # .split_by_node(rank=accelerator.process_index, world_size=accelerator.num_processes)
    #     # .split_by_worker(worker_info=wds.worker_info)
    #     .decode("pil", handler=wds.ignore_and_continue)
    #     .to_tuple("jpg", "cls")
    #     .map_tuple(preprocess_image, None)
    # )

    dataloader = wds.WebLoader(
        train_dataset,
        batch_size      = None,
        num_workers     = config.num_workers,
        pin_memory      = True,
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