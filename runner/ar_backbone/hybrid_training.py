import os
import torch
import pprint
import argparse

from tqdm import tqdm
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

from model.vae_aligner import get_vae_aligner
from model.janus.models import MultiModalityCausalLM
from model.dit.diff_mlp import equip_diffhead_query_with_janus
from util.dual_dataloader import get_dataloader_und, get_dataloader_gen, InfiniteIterator
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

    # load models
    vae_aligner = get_vae_aligner(config.vae_aligner)
    ckpt = torch.load(config.vae_aligner.pretrained_path, map_location="cpu", weights_only=True)
    vae_aligner.load_state_dict(ckpt, strict=True)
    vae_aligner_projector = vae_aligner.siglip_feature_proj

    if config.train.ar_backbone == "janus1b":
        janus = MultiModalityCausalLM.from_pretrained(config.janus_1b_path, trust_remote_code=True)
    elif config.train.ar_backbone == "janus7b":
        janus = MultiModalityCausalLM.from_pretrained(config.janus_7b_path, trust_remote_code=True)
    janus, train_scheduler = equip_diffhead_query_with_janus(janus, config)

    if config.train.diffhead_resume_path is not None:
        diffhead_ckpt = torch.load(config.train.diffhead_resume_path, map_location="cpu", weights_only=True)
        janus.diff_head.load_state_dict(diffhead_ckpt, strict=True)
        accelerator.print(f"DiffHead model loaded from {config.train.diffhead_resume_path}")
    
    if config.train.siglip16_aligner_resume_path is not None:
        siglip16_aligner_ckpt = torch.load(config.train.siglip16_aligner_resume_path, map_location="cpu", weights_only=True)
        janus.siglip16_aligner.load_state_dict(siglip16_aligner_ckpt, strict=True)
        accelerator.print(f"siglip16_aligner model loaded from {config.train.siglip16_aligner_resume_path}")
    
    if config.train.backbone_resume_path is not None:
        backbone_ckpt = torch.load(config.train.backbone_resume_path, map_location="cpu", weights_only=True)
        janus.language_model.model.load_state_dict(backbone_ckpt, strict=True)
        accelerator.print(f"Backbone model loaded from {config.train.backbone_resume_path}")

    siglip = janus.vision_model
    vae_aligner_projector.requires_grad_(False)
    siglip.requires_grad_(False)

    global_step = config.train.global_step if config.train.global_step is not None else 0
    params_to_learn = list(p for p in janus.parameters() if p.requires_grad)

    optimizer = torch.optim.AdamW(
        params_to_learn,
        lr           = config.train.lr,
        betas        = (0.9, 0.95),
        weight_decay = 5e-2,
        eps          = 1e-8,
    )

    if accelerator.mixed_precision == "bf16":
        dtype = torch.bfloat16
    elif accelerator.mixed_precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    dataloader_und = get_dataloader_und(config)
    dataloader_gen = get_dataloader_gen(config)

    janus, optimizer, dataloader_und, dataloader_gen = accelerator.prepare(janus, optimizer, dataloader_und, dataloader_gen)

    siglip = siglip.to(accelerator.device, dtype).eval()
    vae_aligner_projector = vae_aligner_projector.to(accelerator.device, dtype).eval()

    inf_iter_und = InfiniteIterator(dataloader_und, "理解数据集")
    inf_iter_gen = InfiniteIterator(dataloader_gen, "生成数据集")
    training_done = False

    progress_bar = tqdm(
        total   = config.train.num_iter,
        initial = global_step,
        desc    = "Steps",
        disable = not accelerator.is_local_main_process,
    )

    config.device_count = accelerator.num_processes
    if accelerator.is_main_process:
        accelerator.init_trackers(config.train.wandb_proj, config=flatten_dict(config))
        with open(os.path.join(output_dir, "config.yaml"), "w") as f:
            OmegaConf.save(config, f)

    accelerator.print(f"Learnable parameters: {sum(p.numel() for p in params_to_learn if p.requires_grad) / 1e6} M")
    accelerator.print(f"vae_aligner dtype: {next(vae_aligner.parameters()).dtype}")
    accelerator.print(f"siglip dtype: {next(siglip.parameters()).dtype}")
    accelerator.print(f"janus dtype: {next(janus.parameters()).dtype}")
    accelerator.print(f"Accelerator mixed precision: {accelerator.mixed_precision}")

    while not training_done:
        # load und data and gen data
        batch_gen = next(inf_iter_gen)
        gen_pixel_value = batch_gen["pixel_values"]
        gen_pixel_value = gen_pixel_value * 2 - 1
        gen_input_ids = batch_gen["input_ids"]
        gen_attention_mask = batch_gen["attention_mask"]
        accelerator.print(f"gen_pixel_value shape: {gen_pixel_value.shape}")
        accelerator.print(f"gen_input_ids shape: {gen_input_ids.shape}")
        accelerator.print(f"gen_attention_mask shape: {gen_attention_mask.shape}")

        batch_und = next(inf_iter_und)
        und_pixel_values = batch_und["pixel_values"]
        und_input_ids = batch_und["input_ids"]
        und_attention_mask = batch_und["attention_mask"]
        und_labels = batch_und["labels"]
        accelerator.print(f"und_pixel_values shape: {und_pixel_values.shape}")
        accelerator.print(f"und_input_ids shape: {und_input_ids.shape}")
        accelerator.print(f"und_attention_mask shape: {und_attention_mask.shape}")
        accelerator.print(f"und_labels shape: {und_labels.shape}")

        if gen_pixel_value.shape[0] == 0 or und_pixel_values.shape[0] == 0:
            continue
    
        with accelerator.accumulate([janus]):
            janus.train()

            # forward
            with torch.no_grad():
                x_siglip = siglip(gen_pixel_value)
                visual_gen_feature = vae_aligner_projector(x_siglip)
        # 使用无限迭代器获取数据
        # try:
        # batch_und = next(inf_iter_und)
        # und_pixel_values = batch_und["pixel_values"]
        # und_input_ids = batch_und["input_ids"]
        # und_attention_mask = batch_und["attention_mask"]
        # und_labels = batch_und["labels"]
        # und_samples += und_pixel_values.shape[0]

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
        # batch_gen_img, batch_gen_prompt = next(inf_iter_gen)
        # gen_pixel_value = batch_gen_img["pixel_value"]
        # gen_input_ids = batch_gen_prompt["input_ids"]
        # gen_attention_mask = batch_gen_prompt["attention_mask"]
        # gen_samples += gen_pixel_value.shape[0]
        # batch_gen = next(inf_iter_gen)
        # gen_pixel_value = batch_gen["pixel_value"]
        # gen_input_ids = batch_gen["input_ids"]
        # gen_attention_mask = batch_gen["attention_mask"]
        # gen_samples += gen_pixel_value.shape[0]

        # if und_samples % 10000 == 0:
        #     print(f"理解数据集样本数: {und_samples}")
        # if gen_samples % 10000 == 0:
        #     print(f"生成数据集样本数: {gen_samples}")
        #     print(gen_pixel_value.shape)
        #     print(gen_input_ids.shape)
        #     print(gen_attention_mask.shape)
        progress_bar.update(1)

        # print(gen_pixel_value.shape)
        # print(gen_input_ids.shape)
        # print(gen_attention_mask.shape)

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