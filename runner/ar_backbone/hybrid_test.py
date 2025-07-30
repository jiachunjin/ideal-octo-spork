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
def people_recognition():
    device = "cuda:7"
    dtype = torch.bfloat16
    exp_dir = "/data/phd/jinjiachun/experiment/query_dit/0728_hybrid_dataloader"
    config_path = os.path.join(exp_dir, "config.yaml")
    config = OmegaConf.load(config_path)

    # ----- load models -----
    vl_chat_processor = VLChatProcessor.from_pretrained(config.janus_1b_path)
    tokenizer = vl_chat_processor.tokenizer

    janus = MultiModalityCausalLM.from_pretrained(config.janus_1b_path, trust_remote_code=True)
    janus = janus.to(device, dtype).eval()

    question_1 = "Do you know the person in the image?"
    questions = [question_1]
    image = "/data/phd/jinjiachun/codebase/ideal-octo-spork/asset/internet/messi.jpg"
    for question in questions:
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
        ).to(device, dtype)

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

    question_1 = "Do you know the person in the image?"
    questions = [question_1]
    image = "/data/phd/jinjiachun/codebase/ideal-octo-spork/asset/internet/trump.jpg"
    for question in questions:
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
        ).to(device, dtype)

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

    question_1 = "Do you know the person in the image?"
    questions = [question_1]
    image = "/data/phd/jinjiachun/codebase/ideal-octo-spork/asset/internet/einstein.jpg"
    for question in questions:
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
        ).to(device, dtype)

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

    question_1 = "Do you know the person in the image?"
    questions = [question_1]
    image = "/data/phd/jinjiachun/codebase/ideal-octo-spork/asset/internet/ronaldo.jpg"
    for question in questions:
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
        ).to(device, dtype)

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

    question_1 = "Do you know the person in the image?"
    questions = [question_1]
    image = "/data/phd/jinjiachun/codebase/ideal-octo-spork/asset/internet/jobs.jpg"
    for question in questions:
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
        ).to(device, dtype)

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

@torch.no_grad()
def main():
    device = "cuda:7"
    dtype = torch.bfloat16
    exp_dir = "/data/phd/jinjiachun/experiment/query_dit/0729_hybrid_debug"
    config_path = os.path.join(exp_dir, "config.yaml")
    config = OmegaConf.load(config_path)

    # ----- load models -----
    vl_chat_processor = VLChatProcessor.from_pretrained(config.janus_1b_path)
    tokenizer = vl_chat_processor.tokenizer

    janus = MultiModalityCausalLM.from_pretrained(config.janus_1b_path, trust_remote_code=True)
    janus, _ = equip_diffhead_query_with_janus(janus, config)

    diffhead_ckpt = torch.load(os.path.join(exp_dir, "diff_head-query_dit-20000"), map_location="cpu", weights_only=True)
    janus.diff_head.load_state_dict(diffhead_ckpt, strict=True)

    siglip16_aligner_ckpt = torch.load(os.path.join(exp_dir, "siglip16_aligner-query_dit-20000"), map_location="cpu", weights_only=True)
    janus.siglip16_aligner.load_state_dict(siglip16_aligner_ckpt, strict=True)
    
    llm_ckpt = torch.load(os.path.join(exp_dir, "janus-backbone-query_dit-20000"), map_location="cpu", weights_only=True)
    janus.language_model.model.load_state_dict(llm_ckpt, strict=True)
    janus = janus.to(device, dtype).eval()

    # the refiner
    from runner.mmdit.train_basic_sd3 import load_pretrained_mmdit, sample_sd3_5
    from diffusers import FlowMatchEulerDiscreteScheduler
    from einops import rearrange

    exp_dir = "/data/phd/jinjiachun/experiment/mmdit/0714_mmdit_dev"

    config_path = os.path.join(exp_dir, "config.yaml")
    config = OmegaConf.load(config_path)

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(config.sd3_5_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(config.vae_path)
    vae_aligner = get_vae_aligner(config.vae_aligner)
    ckpt = torch.load(config.vae_aligner.pretrained_path, map_location="cpu", weights_only=True)
    vae_aligner.load_state_dict(ckpt, strict=True)
    vae_aligner_projector = vae_aligner.siglip_feature_proj
    transformer = load_pretrained_mmdit(config.sd3_5_path)
    ckpt_path = os.path.join(exp_dir, "transformer-mmdit-30000")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    transformer.load_state_dict(ckpt, strict=True)

    transformer = transformer.to(device, dtype).eval()

    vae_aligner_projector = vae_aligner_projector.to(device, dtype).eval()
    vae_aligner = vae_aligner.to(device, dtype).eval()
    vae = vae.to(device, dtype).eval()

    # ----------------------------------------
    # ---------- test understanding ----------
    # ----------------------------------------
    # question_1 = "Describe this image in great detail, what is on this man's shirt?"
    # question_2 = "What is the color of the scarf? Answer in one word."
    # question_3 = "图中的文字是什么？"
    # question_4 = "Is there any text in the image?"
    # questions = [question_1]
    # image = "/data/phd/jinjiachun/codebase/ideal-octo-spork/asset/internet/messi_1.jpg"
    # for question in questions:
    #     conversation = [
    #         {
    #             "role": "<|User|>",
    #             "content": f"<image_placeholder>\n{question}",
    #             "images": [image],
    #         },
    #         {"role": "<|Assistant|>", "content": ""},
    #     ]

    #     # load images and prepare for inputs
    #     pil_images = load_pil_images(conversation)
    #     prepare_inputs = vl_chat_processor(
    #         conversations=conversation, images=pil_images, force_batchify=True
    #     ).to(device, dtype)

    #     # # run image encoder to get the image embeddings
    #     inputs_embeds = janus.prepare_inputs_embeds(**prepare_inputs)

    #     # # run the model to get the response
    #     outputs = janus.language_model.generate(
    #         inputs_embeds=inputs_embeds,
    #         attention_mask=prepare_inputs.attention_mask,
    #         pad_token_id=tokenizer.eos_token_id,
    #         bos_token_id=tokenizer.bos_token_id,
    #         eos_token_id=tokenizer.eos_token_id,
    #         max_new_tokens=512,
    #         do_sample=False,
    #         use_cache=True,
    #     )

    #     answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    #     print(f"{prepare_inputs['sft_format'][0]}", answer)


    # question_1 = "Describe the image in detail."
    # question_2 = "Do you know the person in the image?"
    # question_3 = "Is there any text in the image?"
    # questions = [question_1, question_2, question_3]
    # image = "/data/phd/jinjiachun/codebase/connector/asset/kobe.png"
    # for question in questions:
    #     conversation = [
    #         {
    #             "role": "<|User|>",
    #             "content": f"<image_placeholder>\n{question}",
    #             "images": [image],
    #         },
    #         {"role": "<|Assistant|>", "content": ""},
    #     ]

    #     # load images and prepare for inputs
    #     pil_images = load_pil_images(conversation)
    #     prepare_inputs = vl_chat_processor(
    #         conversations=conversation, images=pil_images, force_batchify=True
    #     ).to(device, dtype)

    #     # # run image encoder to get the image embeddings
    #     inputs_embeds = janus.prepare_inputs_embeds(**prepare_inputs)

    #     # # run the model to get the response
    #     outputs = janus.language_model.generate(
    #         inputs_embeds=inputs_embeds,
    #         attention_mask=prepare_inputs.attention_mask,
    #         pad_token_id=tokenizer.eos_token_id,
    #         bos_token_id=tokenizer.bos_token_id,
    #         eos_token_id=tokenizer.eos_token_id,
    #         max_new_tokens=512,
    #         do_sample=False,
    #         use_cache=True,
    #     )

    #     answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    #     print(f"{prepare_inputs['sft_format'][0]}", answer)

    # -------------------------------------
    # ---------- test generation ----------
    # -------------------------------------
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

        pred_latents = torch.randn((B, 16), device=feature.device, dtype=dtype)
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
    ]
    cfg_scale = 3
    for img_idx, prompt in enumerate(prompts):
        conversation = [
            {
                "role": "<|User|>",
                "content": prompt,
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations = conversation,
            sft_format    = vl_chat_processor.sft_format,
            system_prompt = "",
        )
        prompt = sft_format + vl_chat_processor.image_start_tag
        input_ids = tokenizer.encode(prompt)
        input_ids = torch.LongTensor(input_ids).to(device)

        if cfg_scale > 1:
            input_ids = input_ids.repeat(2, 1)
            input_ids[1, :-1] = tokenizer.pad_token_id
            text_embedding = janus.language_model.get_input_embeddings()(input_ids).to(device)
        else:
            text_embedding = janus.language_model.get_input_embeddings()(input_ids).to(device)

        generated_tokens = torch.zeros((1, 576, 16)).to(device, dtype)

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
        os.makedirs("asset/diffhead_hybrid", exist_ok=True)
        torchvision.utils.save_image(grid, f"asset/diffhead_hybrid/coarse_{img_idx:02d}.png")

        import torchvision.utils as vutils
        sample_path = f"asset/diffhead_hybrid/fine_{img_idx:02d}.png"
        vutils.save_image(samples, sample_path, nrow=2, normalize=False)
        print(f"Samples saved to {sample_path}")

if __name__ == "__main__":
    main()
    # people_recognition()