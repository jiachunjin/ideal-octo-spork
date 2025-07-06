import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from .vit_basic import precompute_freqs_cis_2d, Block

class ViTVAEAligner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.depth = config.depth
        self.num_heads = config.num_heads
        self.grid_size = config.grid_size

        self.siglip_feature_dim_down = config.siglip_feature_dim_down
        self.siglip_feature_proj = nn.Linear(config.siglip_feature_dim, config.siglip_feature_dim_down)
        
        self.precompute_pos = dict()
        self.input_proj = nn.Linear(config.siglip_feature_dim_down, config.hidden_size)
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.blocks = nn.ModuleList([Block(config.hidden_size, config.num_heads) for _ in range(config.depth)])
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.output_proj = nn.Sequential(
            nn.Conv2d(self.hidden_size, 64, 1, padding=0, bias=True),
            Rearrange("b (p1 p2 c) h w -> b c (h p1) (w p2)", p1=2, p2=2)
        )

    def forward(self, x_siglip):
        """
        x: [B, L, D], original siglip feature
        """
        pos = self.fetch_pos(self.grid_size, self.grid_size, x_siglip.device)
        B, L, D = x_siglip.shape

        x = self.siglip_feature_proj(x_siglip)
        x = self.input_proj(x)
        x = self.norm1(x)
        x = x.to(x_siglip.dtype)
        for block in self.blocks:
            x = block(x, pos)
        x = self.norm2(x)
        x = x.permute(0, 2, 1).reshape(-1, self.hidden_size, self.grid_size, self.grid_size).contiguous()
        x = self.output_proj(x)
        return x

    def fetch_pos(self, height, width, device):
        if (height, width) in self.precompute_pos:
            return self.precompute_pos[(height, width)].to(device)
        else:
            pos = precompute_freqs_cis_2d(self.hidden_size // self.num_heads, height, width).to(device)
            self.precompute_pos[(height, width)] = pos
            return pos


if __name__ == "__main__":
    from omegaconf import OmegaConf
    config = OmegaConf.load("config/vae_aligner/siglip_flux.yaml")
    model = ViTVAEAligner(config.vae_aligner)
    x_siglip = torch.randn(1, 576, 1024)
    rec_vae_feature = model(x_siglip)
    print(rec_vae_feature.shape)