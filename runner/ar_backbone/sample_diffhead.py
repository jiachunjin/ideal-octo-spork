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
from model.dit.diff_mlp import equip_diffhead_query_with_janus

@torch.no_grad()
def main():
    device = "cuda:7"
    dtype = torch.float32
#     exp_dir = "/data/phd/jinjiachun/experiment/query_dit/0717_diff_head_fixbackbone"
    exp_dir = "/data/phd/jinjiachun/experiment/query_dit/0717_diff_head_fixbackbone_ar"
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

    diffhead_ckpt = torch.load(os.path.join(exp_dir, "diff_head-query_dit-13000"), map_location="cpu", weights_only=True)
    janus.diff_head.load_state_dict(diffhead_ckpt, strict=True)

    siglip16_aligner_ckpt = torch.load(os.path.join(exp_dir, "siglip16_aligner-query_dit-13000"), map_location="cpu", weights_only=True)
    janus.siglip16_aligner.load_state_dict(siglip16_aligner_ckpt, strict=True)
    
    llm_ckpt = torch.load(os.path.join(exp_dir, "janus-backbone-query_dit-13000"), map_location="cpu", weights_only=True)
    janus.language_model.model.load_state_dict(llm_ckpt, strict=True)

    # the refiner
    from runner.mmdit.train_basic_sd3 import load_pretrained_mmdit, sample_sd3_5
    from diffusers import FlowMatchEulerDiscreteScheduler
    from einops import rearrange

    exp_dir = "/data/phd/jinjiachun/experiment/mmdit/0714_mmdit_dev"

    config_path = os.path.join(exp_dir, "config.yaml")
    config = OmegaConf.load(config_path)

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(config.sd3_5_path, subfolder="scheduler")
    transformer = load_pretrained_mmdit(config.sd3_5_path)
    ckpt_path = os.path.join(exp_dir, "transformer-mmdit-30000")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    transformer.load_state_dict(ckpt, strict=True)

    transformer = transformer.to(device, dtype).eval()

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

        pred_latents = torch.randn((B, 16), device=feature.device)
        pred_latents *= sample_scheduler.init_noise_sigma

        for t in sample_scheduler.timesteps:
            pred_latents = sample_scheduler.scale_model_input(pred_latents, t)
            with torch.no_grad():
                t_sample = torch.as_tensor([t], device=feature.device)
                noise_pred = diff_head(pred_latents, t_sample.repeat(B), feature)
                pred_latents = sample_scheduler.step(noise_pred, t, pred_latents).prev_sample

        return pred_latents

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
    cfg_scale = 3

    for i, prompt in enumerate(prompts): 
        input_ids = tokenizer.encode(prompt)
        input_ids = torch.LongTensor(input_ids)
        input_ids = torch.cat([input_ids, torch.tensor([100003])]).to(device)

        if cfg_scale > 1:
            input_ids = input_ids.repeat(2, 1)
            input_ids[1, :-1] = 100002
            text_embedding = janus.language_model.get_input_embeddings()(input_ids).to(device)
        else:
            text_embedding = janus.language_model.get_input_embeddings()(input_ids).to(device)

        generated_tokens = torch.zeros((1, 576, 16)).to(device)

        for i in trange(576):
            outputs = janus.language_model.model(inputs_embeds=text_embedding, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
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
                text_embedding = img_embeds.repeat(2, 1, 1)
            else:
                text_embedding = img_embeds

        print(generated_tokens.shape)
        rec = vae_aligner.forward_with_low_dim(generated_tokens)
        print(rec.shape)

        context = rearrange(rec, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=4, p2=4)
        samples = sample_sd3_5(
            transformer         = transformer,
            vae                 = vae,
            noise_scheduler     = noise_scheduler,
            device              = device,
            dtype               = dtype,
            context             = context,
            batch_size          = context.shape[0],
            height              = 384,
            width               = 384,
            num_inference_steps = 50,
            guidance_scale      = 2.0,
            seed                = 42
        )


        reconstructed = vae.decode(rec).sample
        reconstructed = (reconstructed + 1) / 2
        reconstructed = torch.clamp(reconstructed, 0, 1)
        grid = torchvision.utils.make_grid(reconstructed, nrow=4)
        os.makedirs("asset/diffhead", exist_ok=True)
        torchvision.utils.save_image(grid, f"asset/diffhead/coarse_{i:02d}.png")

        import torchvision.utils as vutils
        sample_path = f"asset/diffhead/fine_{i:02d}.png"
        vutils.save_image(samples, sample_path, nrow=2, normalize=False)
        print(f"Samples saved to {sample_path}")

if __name__ == "__main__":
    main()