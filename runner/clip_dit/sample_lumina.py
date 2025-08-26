import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from einops import rearrange

@torch.no_grad()
def sample_t2i():
    from omegaconf import OmegaConf
    from diffusers import DDIMScheduler, AutoencoderKL, FlowMatchEulerDiscreteScheduler
    from transformers import AutoTokenizer
    from tqdm import tqdm

    from model.internvl.modeling_internvl_chat import InternVLChatModel
    from model.dit.lumina_next.nextdit import NextDiTCrossAttn, NextDiTCrossAttnConfig
    from model.internvl.conversation import get_conv_template
    from runner.clip_dit.lumina_dit import add_query

    from model.mmdit import load_mmdit
    from runner.mmdit.train_basic_sd3 import sample_sd3_5

    device = torch.device("cuda:2")
    dtype = torch.float16

    # ----- load dit -----
    exp_dir = "/data/phd/jinjiachun/experiment/clip_1024/0825_metaquery_lumina_dit"
    step = 140000
    config = OmegaConf.load(os.path.join(exp_dir, "config.yaml"))
    dit_config = NextDiTCrossAttnConfig(**config.dit)
    model = NextDiTCrossAttn(dit_config)
    model = add_query(model, config.query)

    ckpt = torch.load(os.path.join(exp_dir, f"dit-{config.train.exp_name}-{step}"), map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt, strict=True)
    model = model.to(device, dtype).eval()

    # ----- load internvl -----
    internvl = InternVLChatModel.from_pretrained(config.intern_vl_8b_path)
    internvl = internvl.to(device, dtype).eval()
    tokenizer = AutoTokenizer.from_pretrained(config.intern_vl_1b_path, trust_remote_code=True, use_fast=False)

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

    # load vae
    vae = AutoencoderKL.from_pretrained(config_decoder.sd3_5_path, subfolder="vae")
    vae.requires_grad_(False)
    vae = vae.to(device, dtype).eval()

    # ----- prepare llm hidden state as condition -----
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

    for prompt_txt in prompts:
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
        # print(attention_mask)
        text_embedding = internvl.language_model.get_input_embeddings()(input_ids).to(device)
        # print(text_embedding.shape)

        joint_embedding = torch.cat((text_embedding, model.query.repeat(2, 1, 1)), dim=1)
        img_mask = torch.ones((2, config.query.num_query), dtype=torch.bool, device=device)
        attention_mask = torch.cat([attention_mask, img_mask], dim=1)

        hidden_states = internvl.language_model(
            inputs_embeds        = joint_embedding,
            attention_mask       = attention_mask,
            output_hidden_states = True,
        ).hidden_states[-1][:, -config.query.num_query:, :]

        print(hidden_states.shape)

        # ----- do diffusion sampling -----
        scheduler = DDIMScheduler(
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

        scheduler.set_timesteps(50)

        B = 16
        cfg_scale = 3.5
        x = torch.randn((B, 1024, 32, 32), device=device, dtype=dtype)
        x *= scheduler.init_noise_sigma

        if cfg_scale > 1.0:
            x = x.repeat(2, 1, 1, 1)
            print(x.shape)
            hidden_states = hidden_states.repeat_interleave(B, dim=0)
            print(hidden_states.shape)

        for t in tqdm(scheduler.timesteps):
            x_in = scheduler.scale_model_input(x, t)
            t_sample = torch.as_tensor([t], device=device)
            if cfg_scale > 1.0:
                noise_pred = model(x_in, t_sample, hidden_states)
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
                x_out = x[:B]  # 只保留cond部分
                x_out = scheduler.step(noise_pred, t, x_out).prev_sample
                # 更新x的前B个为新值，uncond部分保持不变
                x = torch.cat([x_out, x_out], dim=0)
            else:
                noise_pred = model(x_in, t_sample, hidden_states)
                x = scheduler.step(noise_pred, t, x).prev_sample                

        if cfg_scale > 1.0:
            x = x[:B]

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
        sample_path = f"asset/clip_dit/t2i_lumina_{prompt_txt[:10]}_{step}.png"
        vutils.save_image(samples, sample_path, nrow=4, normalize=False)
        print(f"Samples saved to {sample_path}")    


if __name__ == "__main__":
    sample_t2i()