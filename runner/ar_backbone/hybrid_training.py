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

    print(dataloader_und)
    print(dataloader_gen)

    accelerator.end_training()

    exit(0)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/ar_backbone/hybrid_training.yaml")
    args = parser.parse_args()
    main(args)