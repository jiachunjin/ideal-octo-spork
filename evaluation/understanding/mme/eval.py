import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import torch
from tqdm import tqdm, trange
from omegaconf import OmegaConf
from datasets import load_dataset


from model.vae_aligner import get_vae_aligner
from model.janus.models import MultiModalityCausalLM, VLChatProcessor
from model.janus.utils.io import load_pil_images
from model.dit.diff_mlp import equip_diffhead_query_with_janus


device = "cuda:0"
dtype = torch.bfloat16
exp_dir = "/data/phd/jinjiachun/experiment/query_dit/0730_hybrid_new_aligner_joint_training"
config_path = os.path.join(exp_dir, "config.yaml")
config = OmegaConf.load(config_path)

# -----------------------------------------------------
# -------------------- load models --------------------
# -----------------------------------------------------
vl_chat_processor = VLChatProcessor.from_pretrained(config.janus_1b_path)
tokenizer = vl_chat_processor.tokenizer

janus = MultiModalityCausalLM.from_pretrained(config.janus_1b_path, trust_remote_code=True)

llm_ckpt = torch.load(os.path.join(exp_dir, "janus-backbone-query_dit-2000"), map_location="cpu", weights_only=True)
janus.language_model.model.load_state_dict(llm_ckpt, strict=True)
janus = janus.to(device, dtype).eval()

# -----------------------------------------------------
# ------------------ load eval data -------------------
# -----------------------------------------------------
# 定义数据文件路径
data_files = {
    "test": "/data/phd/jinjiachun/dataset/benchmark/darkyarding/MME/data/test-*-of-*.parquet"
}
dataset = load_dataset("parquet", data_files=data_files)

for data in dataset["test"]:
    img_name = data["question_id"].split("/")[-1]
    category = data["category"]
    image = data["image"]
    question = data["question"]
    gt_answer = data["answer"]

    print(category, img_name)

    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{question}",
            # "images": [image],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    # pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=[image], force_batchify=True
    ).to(device, dtype)

    # # run image encoder to get the image embeddings
    inputs_embeds = janus.prepare_inputs_embeds(**prepare_inputs)

    # # run the model to get the response
    outputs = janus.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True,
    )

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    print(category, img_name, answer, gt_answer)