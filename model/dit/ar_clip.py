import torch
import torch.nn as nn
from timm.layers.mlp import Mlp

from model.dit.standard_dit import LabelEmbedder, get_2d_sincos_pos_embed
from model.dit.diff_mlp import SimpleMLPAdaLN

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class CausalAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Transformer_Block(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = CausalAttention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class AR_CLIP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.x_embedder = nn.Linear(config.in_channels, config.hidden_size)
        self.y_embedder = LabelEmbedder(config.num_classes, config.hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.num_tokens, config.hidden_size), requires_grad=False)
        self.blocks = nn.ModuleList([
            Transformer_Block(config.hidden_size, config.num_heads, mlp_ratio=4) for _ in range(config.depth)
        ])

        self.final_layer = nn.Linear(config.hidden_size, config.out_channels)
        self.initialize_weights()

        self.diff_head = SimpleMLPAdaLN(
            in_channels    = config.diff_head.x_dim,
            model_channels = config.diff_head.hidden_size,
            out_channels   = config.diff_head.x_dim,
            z_channels     = config.diff_head.z_dim,
            num_res_blocks = config.diff_head.depth,
        )
    
    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.config.num_tokens ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))       
        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
    
    def forward(self, x, y):
        x = self.x_embedder(x) + self.pos_embed
        y_embed = self.y_embedder(y).unsqueeze(1)
        x = torch.cat([x, y_embed], dim=1)
    
        for block in self.blocks:
            x = block(x)
        x = self.final_layer(x)

        return x
    
if __name__ == "__main__":
    from omegaconf import OmegaConf
    config = OmegaConf.load("config/clip_dit/ar_clip.yaml")
    model = AR_CLIP(config.ar_clip)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params / 1e6:.2f}M")

    B = 3
    x = torch.randn(B, 1024, 1024)
    y = torch.randint(0, 1000, (B,))
    out = model(x, y)
    print(out.shape)
