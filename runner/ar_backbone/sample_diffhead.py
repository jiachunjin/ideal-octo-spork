import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torchvision
from tqdm import tqdm, trange
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, DDIMScheduler
from torchvision import transforms as pth_transforms

from model.vae_aligner import get_vae_aligner
from model.janus.models import MultiModalityCausalLM, VLChatProcessor
from model.dit.diff_mlp import equip_diffhead_query_with_janus

def main():
    exp_dir = "/data/phd/jinjiachun/experiment/query_dit/0717_diff_head_fixbackbone"
    config_path = os.path.join(exp_dir, "config.yaml")
    config = OmegaConf.load(config_path)

    tokenizer = VLChatProcessor.from_pretrained(config.janus_1b_path).tokenizer
    vae = AutoencoderKL.from_pretrained(config.vae_path)
    vae_aligner = get_vae_aligner(config.vae_aligner)
    ckpt = torch.load(config.vae_aligner.pretrained_path, map_location="cpu", weights_only=True)
    vae_aligner.load_state_dict(ckpt, strict=True)
    vae_aligner_projector = vae_aligner.siglip_feature_proj

    janus = MultiModalityCausalLM.from_pretrained(config.janus_1b_path, trust_remote_code=True)
    janus, _ = equip_diffhead_query_with_janus(janus, config)

    diffhead_ckpt = torch.load(os.path.join(exp_dir, "diff_head-query_dit-5000"), map_location="cpu", weights_only=True)
    janus.diff_head.load_state_dict(diffhead_ckpt, strict=True)

    siglip16_aligner_ckpt = torch.load(os.path.join(exp_dir, "siglip16_aligner-query_dit-5000"), map_location="cpu", weights_only=True)
    janus.siglip16_aligner.load_state_dict(siglip16_aligner_ckpt, strict=True)

    siglip = janus.vision_model

    device = "cuda:7"
    dtype = torch.float32

    vae_aligner_projector = vae_aligner_projector.to(device, dtype)
    vae_aligner_projector.eval()
    vae_aligner = vae_aligner.to(device, dtype)
    vae_aligner.eval()
    janus = janus.to(device, dtype)
    janus.eval()
    vae = vae.to(device, dtype)
    vae.eval()


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

        pred_latents = torch.randn((B, 1024), device=feature.device)
        pred_latents *= sample_scheduler.init_noise_sigma

        for t in sample_scheduler.timesteps:
            pred_latents = sample_scheduler.scale_model_input(pred_latents, t)
            with torch.no_grad():
                t_sample = torch.as_tensor([t], device=feature.device)
                noise_pred = diff_head(pred_latents, t_sample.repeat(B), feature)
                pred_latents = sample_scheduler.step(noise_pred, t, pred_latents).prev_sample

        return pred_latents

    prompt = "A man in a white shirt and black pants is playing guitar on the street, with a crowd of people watching him. The background is a city street with buildings and trees."
    cfg_scale = 5

    with torch.no_grad():
        if cfg_scale > 1:
            input_ids = tokenizer.encode(prompt)
            input_ids = torch.LongTensor(input_ids)
            input_ids = torch.cat([input_ids, torch.tensor([100003])]).to(device)
            input_ids = input_ids.unsqueeze(0)
            input_ids = input_ids.repeat(2, 1)
            input_ids[1, :-1] = 100002        
        else:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
            input_ids = torch.LongTensor(input_ids)
            input_ids = torch.cat([input_ids, torch.tensor([100003])]).to(device).unsqueeze(0)

        inputs_embeds = janus.language_model.get_input_embeddings()(input_ids).to(device)

        generated_tokens = torch.zeros((1, 576, 16)).to(device)

        with torch.no_grad():
            for i in trange(576):
                outputs = janus.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
                hidden_states = outputs.last_hidden_state

                if cfg_scale > 1:
                    cond_z = hidden_states[0, -1, :]
                    uncond_z = hidden_states[1, -1, :]
                    z = uncond_z + cfg_scale * (cond_z - uncond_z)
                    z = z.unsqueeze(0)
                else:
                    z = hidden_states[:, -1, :]
                next_token = diff_generate(z, janus.diff_head)
                generated_tokens[:, i] = next_token.squeeze()
                img_embeds = janus.siglip16_aligner(next_token.unsqueeze(0))
                
                if cfg_scale > 1:
                    inputs_embeds = img_embeds.repeat(2, 1, 1)
                else:
                    inputs_embeds = img_embeds

        print(generated_tokens.shape)

if __name__ == "__main__":
    main()