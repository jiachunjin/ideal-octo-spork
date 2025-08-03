import torch
import torch.nn as nn
from diffusers import DDPMScheduler

from .diff_mlp import SimpleMLPAdaLN
from .query_dit import QueryDiT

# ----------------------------------------------------------------------------------
# --------------------------- modify qwen_vl to generate ---------------------------
# ----------------------------------------------------------------------------------
def modify_qwen_vl(qwen_vl, config):
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

    if config.mode == "metaquery":
        qwen_vl.requires_grad_(False)
        query = nn.Parameter(torch.randn(config.query.num_queries, config.query.query_dim))
        query_dit = QueryDiT(config.query_dit)
        qwen_vl.query = query
        qwen_vl.query.requires_grad_(True)

        qwen_vl.query_dit = query_dit
        qwen_vl.query_dit.requires_grad_(True)

        return qwen_vl, train_scheduler

    elif config.modify_qwen_vl.mode == "hat":
        siglip16_aligner = nn.Sequential(
            nn.Linear(16, config.diffhead.z_dim),
            nn.GELU(),
            nn.Linear(config.diffhead.z_dim, config.diffhead.z_dim),
        )
        diff_head = SimpleMLPAdaLN(
            in_channels    = config.diffhead.x_dim,
            model_channels = config.diffhead.hidden_size,
            out_channels   = config.diffhead.x_dim,
            z_channels     = config.diffhead.z_dim,
            num_res_blocks = config.diffhead.depth,
        )

        raise NotImplementedError