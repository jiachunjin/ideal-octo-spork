import os
import pprint
import argparse


from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

from util.dual_dataloader import get_dataloader_und, get_dataloader_gen
from util.misc import process_pretrained_model_path, flatten_dict

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


def main(args):
    config = OmegaConf.load(args.config)
    config = process_pretrained_model_path(config)
    accelerator, output_dir = get_accelerator(config.train)
    accelerator.print("Configuration:")
    accelerator.print(pprint.pformat(OmegaConf.to_container(config, resolve=True), indent=2, width=120).strip('{}'))

    dataloader_und = get_dataloader_und(config)
    dataloader_gen = get_dataloader_gen(config)

    dataloader_und, dataloader_gen = accelerator.prepare(dataloader_und, dataloader_gen)

    class InfiniteIterator:
        """无限迭代器，自动重新初始化"""
        def __init__(self, dataloader, name):
            self.dataloader = dataloader
            self.name = name
            self.iterator = iter(dataloader)
            self.epoch_count = 0
        
        def __next__(self):
            try:
                return next(self.iterator)
            except StopIteration:
                self.epoch_count += 1
                print(f"{self.name} 第 {self.epoch_count} 轮结束，重新初始化...")
                self.iterator = iter(self.dataloader)
                return next(self.iterator)
        
        def __iter__(self):
            return self

    # 创建无限迭代器
    inf_iter_und = InfiniteIterator(dataloader_und, "理解数据集")
    inf_iter_gen = InfiniteIterator(dataloader_gen, "生成数据集")

    while True:
        # 使用无限迭代器获取数据
        # try:
        batch_und = next(inf_iter_und)
        und_pixel_values = batch_und["pixel_values"]
        und_input_ids = batch_und["input_ids"]
        und_attention_mask = batch_und["attention_mask"]
        und_labels = batch_und["labels"]

        # print(und_pixel_values.shape)
        # print(und_input_ids.shape)
        # print(und_attention_mask.shape)
        # print(und_labels.shape)
        # print("理解数据集 batch:")
        # for k, v in batch_und.items():
        #     print(f"{k}: {v.shape if hasattr(v, 'shape') else type(v)}")
        # except Exception as e:
        #     print(f"理解数据集错误: {e}")

        # try:
        batch_gen_img, batch_gen_prompt = next(inf_iter_gen)
        gen_pixel_value = batch_gen_img["pixel_value"]
        gen_input_ids = batch_gen_prompt["input_ids"]
        gen_attention_mask = batch_gen_prompt["attention_mask"]

        print(gen_pixel_value.shape)
        print(gen_input_ids.shape)
        print(gen_attention_mask.shape)

        # print("生成数据集 batch:")

        # print(f"batch_gen[0] keys: {batch_gen[0].keys()}")
        # print(f"batch_gen[1] keys: {batch_gen[1].keys()}")
        # print(f"pixel_value shape: {batch_gen[0]['pixel_value'].shape}")
        # print(f"input_ids shape: {batch_gen[1]['input_ids'].shape}")
        # print(f"attention_mask shape: {batch_gen[1]['attention_mask'].shape}")
        # except Exception as e:
        #     print(f"生成数据集错误: {e}")

    # 如果需要继续迭代更多数据（无限迭代）
    # try:
    #     batch_und_2 = next(inf_iter_und)
    #     batch_gen_2 = next(inf_iter_gen)
    #     print("获取了第二批数据")
    #     
    #     # 可以继续无限迭代
    #     batch_und_3 = next(inf_iter_und)
    #     batch_gen_3 = next(inf_iter_gen)
    #     print("获取了第三批数据")
    # except Exception as e:
    #     print(f"获取更多数据时出错: {e}")

    accelerator.end_training()

    exit(0)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/ar_backbone/hybrid_training.yaml")
    args = parser.parse_args()
    main(args)