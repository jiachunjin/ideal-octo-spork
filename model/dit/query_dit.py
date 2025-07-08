
import torch
import torch.nn as nn
from diffusers import DDPMScheduler
from .query_dit_basic import DiTBlock, TimestepEmbedder, FinalLayer, precompute_freqs_cis_2d

def equip_dit_query_with_janus(janus, config):
    query_dit = QueryDiT(config.query_dit)
    query = nn.Parameter(torch.randn(config.query.num_queries, config.query.query_dim))

    if getattr(config.query_dit, "freeze_janus", True):
        janus.requires_grad_(False)
        janus.eval()
    else:
        janus.requires_grad_(True)
        janus.train()

    janus.query = query
    janus.query.requires_grad_(True)

    janus.query_dit = query_dit
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

class QueryDiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.depth = config.depth
        self.x_dim = config.x_dim
        self.z_dim = config.z_dim

        self.x_proj = nn.Linear(self.x_dim, self.hidden_size)
        self.z_proj = nn.Linear(self.z_dim, self.hidden_size)
        self.t_embedder = TimestepEmbedder(self.hidden_size)
        self.blocks = nn.ModuleList([
            DiTBlock(self.hidden_size, self.num_heads) for _ in range(self.depth)
        ])
        self.final_layer = FinalLayer(self.hidden_size, self.x_dim)
        self.precompute_pos = dict()

    def fetch_pos(self, height, width, device):
        if (height, width) in self.precompute_pos:
            return self.precompute_pos[(height, width)].to(device)
        else:
            pos = precompute_freqs_cis_2d(self.hidden_size // self.num_heads, height, width).to(device)
            self.precompute_pos[(height, width)] = pos
            return pos

    def forward(self, x_t, z, t):
        B, L, d = x_t.shape
        pos = self.fetch_pos(24, 24, x_t.device)
        x = self.x_proj(x_t)
        z = self.z_proj(z)

        t = self.t_embedder(t.view(-1), x_t.dtype).view(B, -1, self.hidden_size)
        c = t + z
        for i, block in enumerate(self.blocks):
            x = block(x, c, pos, None)
        
        x = self.final_layer(x, c)
        
        return x


if __name__ == "__main__":
    from omegaconf import OmegaConf
    config = OmegaConf.load("config/ar_backbone/query_dit.yaml")
    dit = QueryDiT(config.query_dit)
    x_t = torch.randn(1, 576, 16)
    z = torch.randn(1, 576, 2048)
    t = torch.randint(0, 1000, (1,))
    x = dit(x_t, z, t)
    print(x.shape)
