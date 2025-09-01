import os
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from tqdm import tqdm, trange
from model.internvl.modeling_internvl_chat import InternVLChatModel
from model.dit.diff_mlp import intern_add_diffhead_mmdit
from model.internvl.conversation import get_conv_template

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

    pred_latents = torch.randn((B, 8), device=feature.device, dtype=feature.dtype)
    pred_latents *= sample_scheduler.init_noise_sigma

    for t in sample_scheduler.timesteps:
        pred_latents = sample_scheduler.scale_model_input(pred_latents, t)
        with torch.no_grad():
            t_sample = torch.as_tensor([t], device=feature.device)
            noise_pred = diff_head(pred_latents, t_sample.repeat(B), feature)
            pred_latents = sample_scheduler.step(noise_pred, t, pred_latents).prev_sample

    return pred_latents

@torch.no_grad()
def sample_t2i():
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer

    device = "cuda:0"
    dtype = torch.float16

    exp_dir = "/data/phd/jinjiachun/experiment/sunshine/0901_coarse_fine"

    exp_name = exp_dir.split("/")[-1]
    step = 2000

    config = OmegaConf.load(os.path.join(exp_dir, "config.yaml"))
    tokenizer = AutoTokenizer.from_pretrained(config.model.internvl_path, trust_remote_code=True, use_fast=False)

    internvl = InternVLChatModel.from_pretrained(config.model.internvl_path)
    internvl, _ = intern_add_diffhead_mmdit(internvl, config.model)    
    ckpt_path = os.path.join(exp_dir, f"internvl-sunshine-{step}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    m, u = internvl.load_state_dict(ckpt, strict=False)
    print(f"missing: {m}")
    print(f"unmatched: {u}")

    internvl.to(device, dtype).eval()

    vae = AutoencoderKL.from_pretrained(config.sd3_5_path, subfolder="vae")
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
    cfg_scale = 4

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

        generated_tokens = torch.zeros((1, 256, 8)).to(device, dtype)
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
            
            next_token = diff_generate(z, internvl.diff_head)
            img_embeds = internvl.clip_projector(next_token)

            generated_tokens[:, i] = next_token.squeeze()
            
            if cfg_scale > 1:
                text_embedding = img_embeds.repeat(2, 1, 1)
            else:
                text_embedding = img_embeds
        print(generated_tokens.shape)