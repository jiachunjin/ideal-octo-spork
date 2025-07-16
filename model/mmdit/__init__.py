import torch
import torch.nn as nn
from diffusers import DDPMScheduler
from .mmditx import MMDiTX

def construct_mmdit(config):
    context_embedder_config = {
        "target": "torch.nn.Linear",
        "params": {
            "in_features": config.z_dim, # the context dim
            "out_features": config.hidden_size,
        },
    }
    x_block_self_attn_layers = [0, 1, 2, 3, 4, 5, 6, 7]

    mmdit = MMDiTX(
        input_size               = None,
        pos_embed_scaling_factor = None,
        pos_embed_offset         = None,
        pos_embed_max_size       = 24,
        patch_size               = 1,
        in_channels              = config.x_dim,
        depth                    = config.depth,
        hidden_size              = config.hidden_size,
        num_patches              = 576,
        adm_in_channels          = None,
        context_embedder_config  = context_embedder_config,
        qk_norm                  = None,
        x_block_self_attn_layers = x_block_self_attn_layers,
        verbose                  = False,
    )
    return mmdit

def equip_mmdit_query_with_janus(janus, config):
    mmdit = construct_mmdit(config.query_dit)
    query = nn.Parameter(torch.randn(config.query.num_queries, config.query.query_dim))

    if getattr(config.query_dit, "freeze_janus", True):
        janus.requires_grad_(False)
        janus.eval()
    else:
        janus.requires_grad_(False)
        janus.language_model.model.layers.requires_grad_(True)
        janus.train()

    janus.query = query
    janus.query.requires_grad_(True)

    janus.query_dit = mmdit
    janus.query_dit.requires_grad_(True)

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

    return janus, train_scheduler