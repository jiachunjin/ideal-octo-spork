import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


from model.internvl.modeling_internvl_chat import InternVLChatModel
from runner.pos.joint_proj import intern_add_diffhead_projector
from omegaconf import OmegaConf
import os
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from model.internvl.conversation import get_conv_template
from transformers import AutoTokenizer
from tqdm import tqdm, trange
from diffusers import FlowMatchEulerDiscreteScheduler
from runner.mmdit.train_basic_sd3 import sample_sd3_5

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

    pred_latents = torch.randn((B, 16), device=feature.device, dtype=feature.dtype)
    pred_latents *= sample_scheduler.init_noise_sigma

    for t in sample_scheduler.timesteps:
        pred_latents = sample_scheduler.scale_model_input(pred_latents, t)
        with torch.no_grad():
            t_sample = torch.as_tensor([t], device=feature.device)
            noise_pred = diff_head(pred_latents, t_sample.repeat(B), feature)
            pred_latents = sample_scheduler.step(noise_pred, t, pred_latents).prev_sample

    return pred_latents

@torch.no_grad()
def sample():
    # exp_dir = "/data/phd/jinjiachun/experiment/pos/0905_joint_proj_2b"
    # exp_dir = "/data/phd/jinjiachun/experiment/pos/0905_joint_proj_2b_vf"
    # exp_dir = "/data/phd/jinjiachun/experiment/pos/0908_joint_proj_2b_vf_xgen_context_query"
    exp_dir = "/data/phd/jinjiachun/experiment/pos/0908_joint_proj_2b_vf_xgen_context_query_hat"
    exp_name = exp_dir.split("/")[-1]
    step = 46000
    device = "cuda:0"
    dtype = torch.float16


    config_path = os.path.join(exp_dir, "config.yaml")
    config = OmegaConf.load(config_path)
    config.model.mmdit.load_pretrained = False

    tokenizer = AutoTokenizer.from_pretrained(config.model.internvl_path, trust_remote_code=True, use_fast=False)

    internvl = InternVLChatModel.from_pretrained(config.model.internvl_path)
    internvl, train_scheduler, noise_scheduler, noise_scheduler_copy = intern_add_diffhead_projector(internvl, config.model)

    ckpt_path = os.path.join(exp_dir, f"internvl-pos-{step}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    m, u = internvl.load_state_dict(ckpt, strict=False)
    print(f"missing: {m}")
    print(f"unmatched: {u}")

    internvl = internvl.to(device, dtype).eval()

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
        "a small office made out of car parts",
        "an old rusted robot wearing pants and a jacket riding skis in a supermarket.",
        "a white paper with black text, 'Hello SJTU ' on it.",
        "Organized and stylish walk-in closet with vibrant clothing and accessories.",
        "Little girl and her huskies share a cozy moment indoors.",
        "A woman with a scarf, holding her head and chest, appears unwell while checking her temperature.",
    ]
    cfg_scale = 2

    for idx, prompt_txt in enumerate(prompts):
        if config.data.use_template:
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
        else:
            prompt = f"Generate an image: {prompt_txt}" + IMG_START_TOKEN
            prompt_null = f"Generate an image: " + IMG_START_TOKEN

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
        print(text_embedding.shape)


        generated_tokens = torch.zeros((1, 256, 16)).to(device, dtype)
        hidden_states_store = []
        for i in trange(256):
            if config.model.use_query:
                text_embedding[:, -1, :] += internvl.query[i].unsqueeze(0).repeat(2, 1)
            # print(text_embedding[:, -1, :].shape, internvl.query[i].unsqueeze(0).repeat(2, 1).shape)
            outputs = internvl.language_model.model(inputs_embeds=text_embedding, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
            hidden_states = outputs.last_hidden_state
            hidden_states_store.append(hidden_states[0].unsqueeze(0))

            if cfg_scale > 1:
                cond_z = hidden_states[0, -1, :]
                uncond_z = hidden_states[1, -1, :]
                z = uncond_z + cfg_scale * (cond_z - uncond_z)
                z = z.unsqueeze(0)
                # hidden_states_store.append(cond_z)
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

        # hidden_states_store = torch.cat(hidden_states_store, dim=1)
        # print(hidden_states_store.shape)
        # img_path = "/data/phd/jinjiachun/codebase/ideal-octo-spork/asset/joint/princess_tgt.png"
        # from PIL import Image
        # from torchvision import transforms
        # from model.internvl import extract_feature_pre_adapter
        # transform = transforms.Compose([
        #     transforms.Resize((448, 448)),
        #     transforms.ToTensor(),
        # ])

        # IMAGENET_MEAN = (0.485, 0.456, 0.406)
        # IMAGENET_STD = (0.229, 0.224, 0.225)
        # imagenet_mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
        # imagenet_std = torch.tensor(IMAGENET_STD, device=device, dtype=dtype).view(1, 3, 1, 1)
        # img = transform(Image.open(img_path)).unsqueeze(0).to(device, dtype)
        # img = (img - imagenet_mean) / imagenet_std

        # x_clip = extract_feature_pre_adapter(internvl.vision_model, img)
        # generated_tokens = internvl.down_projector(x_clip) / int(config.model.diffhead.x_dim ** 0.5)
        # img_embedding = internvl.clip_projector(x_gen)
        # joint_embedding = torch.cat((text_embedding, img_embedding), dim=1)
        # img_mask = torch.ones((1, config.data.num_img_token), dtype=torch.bool, device=device)
        # attention_mask = torch.cat([attention_mask, img_mask], dim=1)

        # hidden_states = internvl.language_model(
        #     inputs_embeds        = joint_embedding,
        #     attention_mask       = attention_mask,
        #     output_hidden_states = True,
        # ).hidden_states[-1]
        # hidden_states_store = hidden_states[:, :-1, :]

        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(config.sd3_5_path, subfolder="scheduler")

        samples = sample_sd3_5(
            transformer         = internvl.mmdit,
            vae                 = vae,
            noise_scheduler     = noise_scheduler,
            device              = device,
            dtype               = dtype,
            # context             = hidden_states_store,
            # batch_size          = hidden_states_store.shape[0],
            context             = generated_tokens,
            batch_size          = generated_tokens.shape[0],
            multi_modal_context = True,
            height              = 448,
            width               = 448,
            num_inference_steps = 25,
            guidance_scale      = 1.0,
            seed                = 42
        )
        print(samples.shape)

        import torchvision.utils as vutils
        sample_path = f"asset/sunshine/{exp_name}_{prompt_txt[:20]}_sd3_{step}_{cfg_scale}.png"
        vutils.save_image(samples, sample_path, nrow=2, normalize=False)
        print(f"Samples saved to {sample_path}")

if __name__ == "__main__":
    sample()