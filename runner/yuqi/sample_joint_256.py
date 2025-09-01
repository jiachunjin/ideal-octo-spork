import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torchvision
from tqdm import tqdm, trange
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, DDIMScheduler
from einops import rearrange

from model.internvl.conversation import get_conv_template
from model.internvl.modeling_internvl_chat import InternVLChatModel
from model.dit.diff_mlp import add_diffhead_dit_to_ar_model
from model.vae_aligner.vit_vae_aligner import get_feature_down_proj


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

    device = "cuda:6"
    dtype = torch.float16

    exp_dir = "/data/phd/jinjiachun/experiment/clip_1024/0831_joint_256x8"

    exp_name = exp_dir.split("/")[-1]
    step = 110000

    config = OmegaConf.load(os.path.join(exp_dir, "config.yaml"))
    tokenizer = AutoTokenizer.from_pretrained(config.intern_vl_1b_path, trust_remote_code=True, use_fast=False)

    internvl = InternVLChatModel.from_pretrained(config.model.internvl_path)
    internvl, train_scheduler = add_diffhead_dit_to_ar_model(internvl, config.model)
    ckpt_path = os.path.join(exp_dir, f"internvl-clip_1024-{step}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    m, u = internvl.load_state_dict(ckpt, strict=False)
    print(f"missing: {m}")
    print(f"unmatched: {u}")

    internvl.to(device, dtype).eval()

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


    mmdit_step = 95000
    exp_dir = "/data/phd/jinjiachun/experiment/mmdit/0817_sd3_256"
    config_path = os.path.join(exp_dir, "config.yaml")
    config_decoder = OmegaConf.load(config_path)
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(config_decoder.sd3_5_path, subfolder="scheduler")
    mmdit_256x8 = load_mmdit(config_decoder)
    ckpt_path = os.path.join(exp_dir, f"mmdit-mmdit-{mmdit_step}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    mmdit_256x8.load_state_dict(ckpt, strict=False)
    mmdit_256x8 = mmdit_256x8.to(device, dtype).eval()

    # load vae
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
        hidden_states_store = torch.zeros((1, 256, config.model.dit.latent_embedding_size)).to(device, dtype)
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
            # print(z.shape, hidden_states_store[:, i].shape)
            hidden_states_store[:, i] = z.squeeze()

            generated_tokens[:, i] = next_token.squeeze()
            
            if cfg_scale > 1:
                text_embedding = img_embeds.repeat(2, 1, 1)
            else:
                text_embedding = img_embeds
        print(generated_tokens.shape)
        print(hidden_states_store.shape) # (1, 256, 1536)

        samples = sample_sd3_5(
            transformer         = mmdit_256x8,
            vae                 = vae,
            noise_scheduler     = noise_scheduler,
            device              = device,
            dtype               = dtype,
            context             = generated_tokens,
            batch_size          = generated_tokens.shape[0],
            height              = 448,
            width               = 448,
            num_inference_steps = 25,
            guidance_scale      = 1.0,
            seed                = 42
        )
        print(samples.shape)

        import torchvision.utils as vutils
        sample_path = f"asset/joint/{exp_name}_256x8_{prompt_txt[:20]}_{step}.png"
        vutils.save_image(samples, sample_path, nrow=4, normalize=False)
        print(f"Samples saved to {sample_path}")   

        sample_scheduler.set_timesteps(50)

        B = 16
        x = torch.randn((B, 1024, 32, 32), device=device, dtype=dtype)
        x *= sample_scheduler.init_noise_sigma

        for t in tqdm(sample_scheduler.timesteps):
            x_in = sample_scheduler.scale_model_input(x, t)
            t_sample = torch.as_tensor([t], device=device)
            # if cfg_scale > 1.0:
            #     noise_pred = internvl.dit(x_in, t_sample, hidden_states_store)
            #     noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2, dim=0)
            #     noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
            #     x_out = x[:B]  # 只保留cond部分
            #     x_out = sample_scheduler.step(noise_pred, t, x_out).prev_sample
            #     # 更新x的前B个为新值，uncond部分保持不变
            #     x = torch.cat([x_out, x_out], dim=0)
            # else:
            noise_pred = internvl.dit(x_in, t_sample, hidden_states_store.repeat(B, 1, 1))
            x = sample_scheduler.step(noise_pred, t, x).prev_sample    


        print(x.shape)

        context = rearrange(x, "B D H W -> B (H W) D")
        print(context.shape)

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
        sample_path = f"asset/joint/t2i_joint_{prompt_txt[:20]}_{step}.png"
        vutils.save_image(samples, sample_path, nrow=4, normalize=False)
        print(f"Samples saved to {sample_path}")


if __name__ == "__main__":
    sample_t2i()