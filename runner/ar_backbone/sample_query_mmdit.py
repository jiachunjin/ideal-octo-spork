import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torchvision
from tqdm import tqdm
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, DDIMScheduler
from torchvision import transforms as pth_transforms

from model.vae_aligner import get_vae_aligner
from model.janus.models import MultiModalityCausalLM, VLChatProcessor
from model.mmdit import equip_mmdit_query_with_janus

def diff_generate(z, dit, sample_scheduler):
    sample_scheduler.set_timesteps(50)
    B = z.shape[0]

    pred_latents = torch.randn((B, 16, 24, 24), device=z.device, dtype=z.dtype)
    pred_latents *= sample_scheduler.init_noise_sigma

    for t in tqdm(sample_scheduler.timesteps):
        pred_latents = sample_scheduler.scale_model_input(pred_latents, t)
        with torch.no_grad():
            t_sample = torch.as_tensor([t], device=z.device)
            # noise_pred = dit(pred_latents, z, t_sample.repeat(B))
            noise_pred = dit(
                x = pred_latents,
                t = t_sample.repeat(B),
                context = z,
            )

            pred_latents = sample_scheduler.step(noise_pred, t, pred_latents).prev_sample
    
    return pred_latents

@torch.no_grad()
def main():
    exp_dir = "/data/phd/jinjiachun/experiment/query_dit/0716_query_mmdit"
    config_path = os.path.join(exp_dir, "config.yaml")
    config = OmegaConf.load(config_path)

    tokenizer = VLChatProcessor.from_pretrained(config.janus_1b_path).tokenizer
    vae = AutoencoderKL.from_pretrained(config.vae_path)
    vae_aligner = get_vae_aligner(config.vae_aligner)
    ckpt = torch.load(config.vae_aligner.pretrained_path, map_location="cpu", weights_only=True)
    vae_aligner.load_state_dict(ckpt, strict=True)
    vae_aligner_projector = vae_aligner.siglip_feature_proj

    janus = MultiModalityCausalLM.from_pretrained(config.janus_1b_path, trust_remote_code=True)
    janus, _ = equip_mmdit_query_with_janus(janus, config)
    query_dit_ckpt = torch.load(os.path.join(exp_dir, "query_dit-query_dit-2000"), map_location="cpu", weights_only=True)
    janus.query_dit.load_state_dict(query_dit_ckpt, strict=True)

    query_ckpt = torch.load(os.path.join(exp_dir, "query-query_dit-2000"), map_location="cpu", weights_only=True)
    janus.query.data.copy_(query_ckpt["query"]);

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

    device = torch.device("cuda:0")
    dtype = torch.float32

    vae_aligner_projector = vae_aligner_projector.to(device, dtype)
    vae_aligner_projector.eval()
    vae_aligner = vae_aligner.to(device, dtype)
    vae_aligner.eval()
    janus = janus.to(device, dtype)
    janus.eval()
    vae = vae.to(device, dtype)
    vae.eval()

    prompts = [
        "A stunning princess from kabul in red, white traditional clothing, blue eyes, brown hair",
        "A soft, natural portrait photograph captures a young woman with fair skin and long, ash-blonde hair cascading gently over her shoulders. At the very bottom of the frame, in simple, elegant lettering, appears the phrase 'BE KIND'",
        "The image depicts a modern, multi-story building with a white facade and numerous balconies. The structure is partially covered in purple scaffolding on the right side, indicating ongoing construction or renovation. The building is situated in an urban area with clear blue skies above. In front of the building, there is a paved plaza with some greenery and a few palm trees. A street lamp stands prominently on the left side of the plaza. To the right, part of another building with a beige exterior is visible. The scene suggests a sunny day in a developed cityscape.",
        "A photo of 4 TVs in a row, with a white background",
        "The image depicts the American Red Cross building, characterized by its neoclassical architectural style. The structure features tall, white columns supporting a pediment and a balustrade at the top. The facade is adorned with large windows, some of which have red crosses, symbolizing the organization's humanitarian mission. The building is set against a clear blue sky, with a tree partially obscuring the right side of the image. The overall appearance suggests a sense of stability and dedication to service, reflecting the Red Cross's commitment to aid and support.",
        "A photo of a red dog",
        "Scientist at Sunway University conducts research in a laboratory setting.",
        "A serious Santa Claus in a rustic setting.",
        "Muscular man in workout attire, standing confidently by a railing.",
        "Confident man in leather jacket leaning against a wall.",
    ]
    for i, prompt in enumerate(prompts):
        B = 8
        cfg_scale = 3

        input_ids = tokenizer.encode(prompt)
        input_ids = torch.LongTensor(input_ids)
        input_ids = torch.cat([input_ids, torch.tensor([100003])]).to(device).unsqueeze(0)

        if cfg_scale > 1:
            input_ids = input_ids.repeat(2, 1)
            input_ids[1, :-1] = 100002
            text_embedding = janus.language_model.get_input_embeddings()(input_ids).to(device)
            joint_embedding = torch.cat((text_embedding, janus.query.unsqueeze(0).repeat(2, 1, 1)), dim=1)
        else:
            text_embedding = janus.language_model.get_input_embeddings()(input_ids).to(device)
            joint_embedding = torch.cat((text_embedding, janus.query.unsqueeze(0)), dim=1)

        hidden_states = janus.language_model(
            inputs_embeds        = joint_embedding,
            attention_mask       = None,
            output_hidden_states = True,
        ).hidden_states[-1]
        z = hidden_states[:, -576:, :]

        if cfg_scale > 1:
            z_cond = z[0]
            z_uncond = z[1]
            z = z_uncond + cfg_scale * (z_cond - z_uncond)
            z = z.unsqueeze(0)
        else:
            z = z

        z = z.repeat(B, 1, 1)
        gen = diff_generate(z, janus.query_dit, sample_scheduler)
        rec = vae_aligner.forward_with_low_dim(gen)

        reconstructed = vae.decode(rec).sample
        reconstructed = (reconstructed + 1) / 2
        reconstructed = torch.clamp(reconstructed, 0, 1)
        grid = torchvision.utils.make_grid(reconstructed, nrow=4)
        torchvision.utils.save_image(grid, f"asset/mmdit/{i:02d}.png")

if __name__ == "__main__":
    main()