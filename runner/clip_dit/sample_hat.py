import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from einops import rearrange
from diffusers import DDIMScheduler

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

def diffusion_generate(feature, diff_head):
    sample_scheduler.set_timesteps(50)
    B = feature.shape[0]

    pred_latents = torch.randn((B, 4, 1024), device=feature.device, dtype=feature.dtype)
    pred_latents *= sample_scheduler.init_noise_sigma

    for t in sample_scheduler.timesteps:
        pred_latents = sample_scheduler.scale_model_input(pred_latents, t)
        t_sample = torch.as_tensor([t], device=feature.device)
        noise_pred = diff_head(pred_latents, t_sample.repeat(B), feature)
        pred_latents = sample_scheduler.step(noise_pred, t, pred_latents).prev_sample

    return pred_latents

@torch.no_grad()
def sample_t2i():
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from tqdm import trange

    
    from model.internvl.conversation import get_conv_template
    from model.internvl.modeling_internvl_chat import InternVLChatModel
    from model.dit.dit_head import equip_internvl
    from runner.overfit_1024.sample_class import block_sequence_to_row_major_tensor

    device = torch.device("cuda:6")
    dtype = torch.float16

    # ----- load modified internvl -----
    exp_dir = "/data/phd/jinjiachun/experiment/clip_1024/0827_hat_clip_fullpara"
    config = OmegaConf.load(os.path.join(exp_dir, "config.yaml"))
    tokenizer = AutoTokenizer.from_pretrained(config.intern_vl_1b_path, trust_remote_code=True, use_fast=False)
    step = 80000
    
    internvl = InternVLChatModel.from_pretrained(config.intern_vl_8b_path)
    internvl, _ = equip_internvl(internvl, config.model)
    if config.model.full_tune:
        ckpt_path = os.path.join(exp_dir, f"internvl_full-clip_1024-{step}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        m, u = internvl.load_state_dict(ckpt, strict=False)
        print(f"missing: {m}")
        print(f"unmatched: {u}")
    else:
        ckpt_path = os.path.join(exp_dir, f"internvl-clip_1024-{step}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

        diff_head_loaded = 0
        hat_layer_loaded = 0
        missing_keys = []
        
        model_state_dict = internvl.state_dict()
        
        for name, param in ckpt.items():
            if name in model_state_dict:
                model_state_dict[name].copy_(param)
                if name.startswith('diff_head.'):
                    diff_head_loaded += 1
                elif 'language_model.model.layers.' in name:
                    hat_layer_loaded += 1
            else:
                missing_keys.append(name)
        
        internvl.load_state_dict(model_state_dict, strict=False)
    internvl = internvl.to(device, dtype).eval()
    
    print(f"Resume training: loaded {diff_head_loaded} diff_head parameters and {hat_layer_loaded} HAT layer parameters")
    if missing_keys:
        print(f"Warning: some keys in checkpoint not found in model: {missing_keys[:5]}...")
    
    # ----- load decoder -----
    from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
    from model.mmdit import load_mmdit
    from runner.mmdit.train_basic_sd3 import sample_sd3_5

    mmdit_step = 140000
    exp_dir = "/data/phd/jinjiachun/experiment/mmdit/0813_sd3_1024"
    config_path = os.path.join(exp_dir, "config.yaml")
    config_decoder = OmegaConf.load(config_path)
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(config_decoder.sd3_5_path, subfolder="scheduler")
    mmdit = load_mmdit(config_decoder)
    ckpt_path = os.path.join(exp_dir, f"mmdit-mmdit-{mmdit_step}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    mmdit.load_state_dict(ckpt, strict=False)
    mmdit = mmdit.to(device, dtype).eval()
    vae = AutoencoderKL.from_pretrained(config_decoder.sd3_5_path, subfolder="vae")
    vae.requires_grad_(False)
    vae = vae.to(device, dtype).eval()

    # ----- sampling -----
    IMG_START_TOKEN = "<img>"
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

    for idx, prompt_txt in enumerate(prompts):
        template = get_conv_template("internvl2_5")
        prompt = f"Generate an image: {prompt_txt}"
        template.append_message(template.roles[0], prompt)
        template.append_message(template.roles[1], None)
        prompt = template.get_prompt() + IMG_START_TOKEN

        template = get_conv_template("internvl2_5")
        prompt_null = f"Generate an image: "
        template.append_message(template.roles[0], prompt_null)
        template.append_message(template.roles[1], None)
        prompt_null = template.get_prompt() + IMG_START_TOKEN
        print(prompt)
        print(prompt_null)

        tokenizer_output = tokenizer(
            [prompt, prompt_null],
            padding        = True,
            padding_side   = "left",
            truncation     = True,
            return_tensors = "pt",
        )
        input_ids = torch.LongTensor(tokenizer_output["input_ids"]).to(device)
        attention_mask = tokenizer_output["attention_mask"].to(device)
        text_embedding = internvl.language_model.get_input_embeddings()(input_ids).to(device)

        generated_tokens = torch.zeros((1, 256, 4096)).to(device, dtype)
        for i in trange(256):
            outputs = internvl.language_model.model(inputs_embeds=text_embedding, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
            hidden_states = outputs.last_hidden_state

            if cfg_scale > 1:
                cond_z = hidden_states[0, -1, :]
                uncond_z = hidden_states[1, -1, :]
                z = uncond_z + cfg_scale * (cond_z - uncond_z)
                z = z.unsqueeze(0)
            else:
                z = hidden_states[:, -1, :]
            
            next_token = diffusion_generate(z, internvl.diff_head)
            next_token = rearrange(next_token, "B L D -> B (L D)")
            generated_tokens[:, i] = next_token.squeeze()
            img_embeds = internvl.mlp1(next_token.unsqueeze(0))
            if cfg_scale > 1:
                text_embedding = img_embeds.repeat(2, 1, 1)
            else:
                text_embedding = img_embeds
        print(generated_tokens.shape)

        context = rearrange(generated_tokens, "b t (s d) -> b (t s) d", s=4, d=1024)
        context = block_sequence_to_row_major_tensor(context, 32)

        samples = sample_sd3_5(
            transformer         = mmdit,
            vae                 = vae,
            noise_scheduler     = noise_scheduler,
            device              = device,
            dtype               = dtype,
            context             = context,
            batch_size          = context.shape[0],
            height              = 448,
            width               = 448,
            num_inference_steps = 25,
            guidance_scale      = 1.0,
            seed                = 42
        )
        print(samples.shape)

        import torchvision.utils as vutils
        sample_path = f"asset/clip_dit/t2i_hat_{prompt_txt[:20]}_{step}.png"
        vutils.save_image(samples, sample_path, nrow=4, normalize=False)
        print(f"Samples saved to {sample_path}")    

if __name__ == "__main__":
    sample_t2i()