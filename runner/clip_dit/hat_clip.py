from model.internvl.modeling_internvl_chat import InternVLChatModel

from diffusers import DDPMScheduler
from transformers import Qwen2Model
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from model.dit.dit_head import DiT_Head


def equip_internvl(internvl, config):
    # ----- add additional trainable LLM layers -----
    internvl.language_model.model = add_hat_to_intern(internvl.language_model.model, config.num_hat)
    
    current_num_layers = len(internvl.language_model.model.layers)
    new_layer_indices = range(current_num_layers - config.num_hat, current_num_layers)
    internvl.requires_grad_(False)
    for idx in new_layer_indices:
        layer = internvl.language_model.model.layers[idx]
        layer.requires_grad_(True)

    # ----- add diffusion head and scheduler -----
    train_scheduler = DDPMScheduler(
        beta_schedule          = "scaled_linear",
        beta_start             = 0.00085,
        beta_end               = 0.012,
        num_train_timesteps    = 1000,
        clip_sample            = False,
        prediction_type        = "v_prediction",
        steps_offset           = 1,
        trained_betas          = None,
        timestep_spacing       = "trailing",
        rescale_betas_zero_snr = True
    )

    internvl.diff_head = DiT_Head(config.dit_head)
    internvl.diff_head.requires_grad_(True)

    return internvl, train_scheduler

def add_hat_to_intern(model: Qwen2Model, num_hat: int):
    new_layers = []
    config = model.config
    current_num_layers = len(model.layers)

    for i in range(num_hat):
        layer_idx = current_num_layers + i
        new_layer = Qwen2DecoderLayer(config, layer_idx)
        new_layers.append(new_layer)

    model.layers.extend(new_layers)
    model.config.num_hidden_layers = len(model.layers)
    model.config.layer_types.extend(["full_attention"] * num_hat)

    return model

def dev():
    from omegaconf import OmegaConf
    config = OmegaConf.load("config/clip_dit/hat_clip.yaml")
    
    intern_vl_8b_path = "/data/phd/jinjiachun/ckpt/OpenGVLab/InternVL3-8B"
    internvl = InternVLChatModel.from_pretrained(intern_vl_8b_path)

    internvl, train_scheduler = equip_internvl(internvl, config.model)

    params_to_learn = list(p for p in internvl.parameters() if p.requires_grad)
    print(f"Learnable parameters: {sum(p.numel() for p in params_to_learn) / 1e6} M")


if __name__ == "__main__":
    dev()