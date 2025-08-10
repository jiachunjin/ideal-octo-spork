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
    if hasattr(config, "feature_down_projector"):
        in_features = config.feature_down_projector.feature_dim_output
    elif hasattr(config, "vae_aligner"):
        in_features = config.vae_aligner.siglip_feature_dim_down
    else:
        raise ValueError("No feature down projector or vae aligner found")

    context_embedder_config = {
        "target": "torch.nn.Linear",
        "params": {
            "in_features": in_features,
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

    if hasattr(config, "feature_down_projector"): # for aligner-free mmdit, apply projection
        feature_down_projector = get_feature_down_proj(config.feature_down_projector)
        feature_down_projector.requires_grad_(True)
        transformer.feature_down_projector = feature_down_projector

        if hasattr(config.feature_down_projector, "feature_mixer"):
            feature_mixer = FeatureMixer(config.feature_down_projector.feature_mixer)
            feature_mixer.requires_grad_(True)
            transformer.feature_mixer = feature_mixer

    return transformer

from model.vae_aligner.vit_basic import precompute_freqs_cis_2d, Block

class FeatureMixer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.depth = config.depth
        self.num_heads = config.num_heads
        self.grid_size = config.grid_size

        self.precompute_pos = dict()
        self.input_proj = nn.Linear(config.input_dim, config.hidden_size)
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.blocks = nn.ModuleList([Block(config.hidden_size, config.num_heads) for _ in range(config.depth)])
        self.norm2 = nn.LayerNorm(config.hidden_size)

    def forward(self, x):
        """
        x: [B, L, D]
        """
        pos = self.fetch_pos(self.grid_size, self.grid_size, x.device)
        B, L, D = x.shape

        x = self.input_proj(x)
        x = self.norm1(x)
        x = x.to(x.dtype)
        for block in self.blocks:
            x = block(x, pos)
        x = self.norm2(x)

        return x

    def fetch_pos(self, height, width, device):
        if (height, width) in self.precompute_pos:
            return self.precompute_pos[(height, width)].to(device)
        else:
            pos = precompute_freqs_cis_2d(self.hidden_size // self.num_heads, height, width).to(device)
            self.precompute_pos[(height, width)] = pos
            return pos