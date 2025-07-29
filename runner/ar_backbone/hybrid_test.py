import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torchvision
from tqdm import tqdm, trange
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, DDIMScheduler

from model.vae_aligner import get_vae_aligner
from model.janus.models import MultiModalityCausalLM, VLChatProcessor
from model.janus.utils.io import load_pil_images
from model.dit.diff_mlp import equip_diffhead_query_with_janus

@torch.no_grad()
def main():
    device = "cuda:7"
    dtype = torch.bfloat16
    exp_dir = "/data/phd/jinjiachun/experiment/query_dit/0728_hybrid_dataloader"
    config_path = os.path.join(exp_dir, "config.yaml")
    config = OmegaConf.load(config_path)

    # ----- load models -----
    vl_chat_processor = VLChatProcessor.from_pretrained(config.janus_1b_path)
    tokenizer = vl_chat_processor.tokenizer

    janus = MultiModalityCausalLM.from_pretrained(config.janus_1b_path, trust_remote_code=True)
    janus, _ = equip_diffhead_query_with_janus(janus, config)

    diffhead_ckpt = torch.load(os.path.join(exp_dir, "diff_head-query_dit-36000"), map_location="cpu", weights_only=True)
    janus.diff_head.load_state_dict(diffhead_ckpt, strict=True)

    siglip16_aligner_ckpt = torch.load(os.path.join(exp_dir, "siglip16_aligner-query_dit-36000"), map_location="cpu", weights_only=True)
    janus.siglip16_aligner.load_state_dict(siglip16_aligner_ckpt, strict=True)
    
    llm_ckpt = torch.load(os.path.join(exp_dir, "janus-backbone-query_dit-36000"), map_location="cpu", weights_only=True)
    janus.language_model.model.load_state_dict(llm_ckpt, strict=True)

    # ----- test understanding -----
    janus = janus.to(dtype).to(device).eval()
    question = "Describe the image in detail."
    image = "/data/phd/jinjiachun/codebase/connector/asset/004.jpg"
    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{question}",
            "images": [image],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    # load images and prepare for inputs
    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(device).to(dtype)

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
    print(f"{prepare_inputs['sft_format'][0]}", answer)



if __name__ == "__main__":
    main()