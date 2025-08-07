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

import os
from safetensors.torch import load_file
from model.vae_aligner.vit_vae_aligner import get_feature_down_proj

def load_mmdit(config):
    patch_size = 2
    depth = 24
    pos_embed_max_size = 384
    num_patches = 147456
    adm_in_channels = 2048
    qk_norm = "rms"
    x_block_self_attn_layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    context_embedder_config = {
        "target": "torch.nn.Linear",
        "params": {
            "in_features": config.feature_down_projector.feature_dim_output,
            "out_features": 1536,
        },
    }

    device = torch.device("cpu")
    dtype = torch.bfloat16

    transformer = MMDiTX(
        input_size               = None,
        pos_embed_scaling_factor = None,
        pos_embed_offset         = None,
        pos_embed_max_size       = pos_embed_max_size,
        patch_size               = patch_size,
        in_channels              = 16,
        depth                    = depth,
        num_patches              = num_patches,
        adm_in_channels          = adm_in_channels,
        context_embedder_config  = context_embedder_config,
        qk_norm                  = qk_norm,
        x_block_self_attn_layers = x_block_self_attn_layers,
        device                   = device,
        dtype                    = dtype,
        verbose                  = False,
    )

    ckpt = load_file(os.path.join(config.sd3_5_path, "sd3.5_medium.safetensors"))
    new_ckpt = {}
    prefix = "model.diffusion_model."
    for k, v in ckpt.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):]
            new_ckpt[new_key] = v
    del new_ckpt["context_embedder.weight"]
    m, u = transformer.load_state_dict(new_ckpt, strict=False)
    print(f"missing keys: {m}")
    print(f"unexpected keys: {u}")

    feature_down_projector = get_feature_down_proj(config.feature_down_projector)
    feature_down_projector.requires_grad_(True)
    transformer.feature_down_projector = feature_down_projector

    return transformer