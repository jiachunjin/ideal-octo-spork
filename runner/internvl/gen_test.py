import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torchvision
from tqdm import trange
from omegaconf import OmegaConf
from diffusers import DDIMScheduler, AutoencoderKL
from transformers import AutoTokenizer

from model.vae_aligner import get_vae_aligner
from model.dit.diff_mlp import add_diffhead_to_ar_model
from model.internvl.modeling_internvl_chat import InternVLChatModel
from model.internvl.conversation import get_conv_template

IMG_START_TOKEN = "<img>"

@torch.no_grad()
def generate_image():
    exp_dir = "/data/phd/jinjiachun/experiment/intern_gen/0806_gen_only"
    config_path = os.path.join(exp_dir, "config.yaml")
    config = OmegaConf.load(config_path)

    device = "cuda:1"
    dtype = torch.bfloat16

    #######################################
    ############# load model ##############
    #######################################
    tokenizer = AutoTokenizer.from_pretrained(config.intern_vl_1b_path, trust_remote_code=True, use_fast=False)
    ar_model = InternVLChatModel.from_pretrained(config.intern_vl_1b_path)
    ar_model, _ = add_diffhead_to_ar_model(ar_model, config.model)

    diffhead_ckpt = torch.load(os.path.join(exp_dir, "diff_head-intern_gen-300"), map_location="cpu", weights_only=True)
    ar_model.diff_head.load_state_dict(diffhead_ckpt, strict=True)

    clip_projector_ckpt = torch.load(os.path.join(exp_dir, "clip_projector-intern_gen-300"), map_location="cpu", weights_only=True)
    ar_model.clip_projector.load_state_dict(clip_projector_ckpt, strict=True)

    llm_ckpt = torch.load(os.path.join(exp_dir, "backbone-intern_gen-300"), map_location="cpu", weights_only=True)
    ar_model.language_model.model.load_state_dict(llm_ckpt, strict=True)
    ar_model = ar_model.to(device, dtype).eval()

    vae_aligner = get_vae_aligner(config.vae_aligner)
    ckpt = torch.load(config.vae_aligner.pretrained_path, map_location="cpu", weights_only=True)
    vae_aligner.load_state_dict(ckpt, strict=True)
    vae_aligner = vae_aligner.to(device, dtype).eval()
    
    vae = AutoencoderKL.from_pretrained(config.vae_path)
    vae = vae.to(device, dtype).eval()

    #######################################
    ########### test generation ###########
    #######################################
    sample_scheduler = DDIMScheduler(
        beta_schedule          = "scaled_linear",
        beta_start             = 0.00085,
        beta_end               = 0.012,
        num_train_timesteps    = 1000,
        clip_sample            = False,
        prediction_type        = "v_prediction",
        set_alpha_to_one       = True,
        steps_offset           = 1,
        trained_betas          = None,
        timestep_spacing       = "trailing",
        rescale_betas_zero_snr = True
    )

    def diff_generate(feature, diff_head):
        sample_scheduler.set_timesteps(50)
        B = feature.shape[0]

        pred_latents = torch.randn((B, config.vae_aligner.siglip_feature_dim_down), device=feature.device, dtype=dtype)
        pred_latents *= sample_scheduler.init_noise_sigma

        for t in sample_scheduler.timesteps:
            pred_latents = sample_scheduler.scale_model_input(pred_latents, t)
            with torch.no_grad():
                t_sample = torch.as_tensor([t], device=feature.device, dtype=dtype)
                noise_pred = diff_head(pred_latents, t_sample.repeat(B), feature)
                pred_latents = sample_scheduler.step(noise_pred, t, pred_latents).prev_sample

        return pred_latents

    prompts = [
        "A stunning princess from kabul in red, white traditional clothing, blue eyes, brown hair",
        "A soft, natural portrait photograph captures a young woman with fair skin and long, ash-blonde hair cascading gently over her shoulders. At the very bottom of the frame, in simple, elegant lettering, appears the phrase 'BE KIND'",
    ]

    cfg_scale = 3
    for img_idx, prompt in enumerate(prompts):
        template = get_conv_template("internvl2_5")
        prompt = f"Generate an image: {prompts}"
        template.append_message(template.roles[0], prompt)
        template.append_message(template.roles[1], None)
        prompt = template.get_prompt() + IMG_START_TOKEN

        prompt_null = f"Generate an image: "
        template.append_message(template.roles[0], prompt_null)
        template.append_message(template.roles[1], None)
        prompt_null = template.get_prompt() + IMG_START_TOKEN

        tokenizer_output = tokenizer(
            [prompt, prompt_null],
            padding        = True,
            padding_side   = "left",
            truncation     = True,
            return_tensors = "pt",
        )

        input_ids = torch.LongTensor(tokenizer_output["input_ids"]).to(device)
        attention_mask = tokenizer_output["attention_mask"].to(device)
        
        text_embedding = ar_model.language_model.get_input_embeddings()(input_ids).to(device)
        print(text_embedding.shape)

        generated_tokens = torch.zeros((1, config.data.num_img_token, 8)).to(device, dtype)

        for i in trange(config.data.num_img_token):
            outputs = ar_model.language_model.model(inputs_embeds=text_embedding, attention_mask=attention_mask, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
            hidden_states = outputs.last_hidden_state

            cond_z = hidden_states[0, -1, :]
            uncond_z = hidden_states[1, -1, :]
            z = uncond_z + cfg_scale * (cond_z - uncond_z)
            z = z.unsqueeze(0)

            next_token = diff_generate(z, ar_model.diff_head)
            generated_tokens[:, i] = next_token.squeeze()
            img_embeds = ar_model.clip_projector(next_token.unsqueeze(0))

            text_embedding = img_embeds.repeat(2, 1, 1)

            img_mask = torch.ones((2, 1), dtype=torch.bool, device=device)
            attention_mask = torch.cat([attention_mask, img_mask], dim=1)

        print(generated_tokens.shape)
        rec = vae_aligner.forward_with_low_dim(generated_tokens)
        print(rec.shape)

        reconstructed = vae.decode(rec).sample
        reconstructed = (reconstructed + 1) / 2
        reconstructed = torch.clamp(reconstructed, 0, 1)
        grid = torchvision.utils.make_grid(reconstructed, nrow=4)
        os.makedirs("asset/intern_gen", exist_ok=True)
        torchvision.utils.save_image(grid, f"asset/intern_gen/coarse_{img_idx:02d}.png")

if __name__ == "__main__":
    generate_image()