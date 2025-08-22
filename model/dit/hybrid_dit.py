import torch
import torch.nn as nn

from einops import rearrange
from timm.layers.mlp import Mlp
from diffusers import DDPMScheduler
from model.dit.standard_dit import TimestepEmbedder
from model.dit.standard_dit import get_2d_sincos_pos_embed


class Attention(nn.Module):
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

    def forward(self, x: torch.Tensor, mask: torch.Tensor, kv_cache=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if kv_cache is not None:
            # 使用KV cache进行增量计算
            cached_k, cached_v = kv_cache
            if cached_k is not None and cached_v is not None:
                # 将新的k,v与缓存的k,v拼接
                k = torch.cat([cached_k, k], dim=2)  # dim=2 是序列长度维度
                v = torch.cat([cached_v, v], dim=2)
            
            # 更新cache，返回完整的k,v用于下次缓存
            new_cache = (k, v)
        else:
            new_cache = None

        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        if kv_cache is not None:
            return x, new_cache
        else:
            return x


class HybridBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4):
        super().__init__()
        self.attn = Attention(hidden_size, num_heads, qkv_bias=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = Mlp(hidden_size, hidden_size * mlp_ratio, hidden_size)
    
    def forward(self, x, mask):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


class HybridDiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.x_embedder = nn.Linear(config.in_channels, config.hidden_size)
        self.x_t_embedder = nn.Linear(config.in_channels, config.hidden_size)
        if config.use_condition:
            self.y_embedder = nn.Linear(config.intern_hidden_size, config.hidden_size)
        self.t_embedder = TimestepEmbedder(config.hidden_size)

        self.pos_embed = nn.Parameter(torch.randn(1, config.seq_len, config.hidden_size), requires_grad=True)

        self.blocks = nn.ModuleList([
            HybridBlock(config.hidden_size, config.num_heads, mlp_ratio=4) for _ in range(config.depth)
        ])

        self.final_layer = nn.Linear(config.hidden_size, config.out_channels)

        self.register_buffer("mask", self._create_attn_mask(config.seq_len, config.block_size))

        self.train_scheduler = DDPMScheduler(
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

    def _create_attn_mask(self, seq_len, block_size):
        mask = torch.zeros(2 * seq_len, 2 * seq_len)

        total_size = 2 * seq_len
        num_blocks = int(seq_len / block_size * 2)
        
        for i in range(num_blocks):
            start_row = i * block_size
            end_row = min((i + 1) * block_size, total_size)
            start_col = i * block_size
            end_col = min((i + 1) * block_size, total_size)
            
            mask[start_row:end_row, start_col:end_col] = 1

        for i in range(num_blocks - 1):
            start_row = seq_len + (i+1) * block_size
            end_row = seq_len + (i+2) * block_size
            start_col = 0
            end_col = (i+1) * block_size

            mask[start_row:end_row, start_col:end_col] = 1

        mask[mask == 0] = float("-inf")
        mask[mask == 1] = 0

        mask = mask.unsqueeze(0).unsqueeze(0)

        return mask

    def forward(self, x, x_t, t, y=None):
        """
        only used for training
        x: (B, 1024, 1024), the clean clip tokens
        x_t: (B, 1024, 1024), the noisy clip tokens
        y: (B, 256, intern_hidden_size), the intern hidden states
        t: (B, 256), the timesteps, the timesteps are the same within one block
        """
        assert self.training
        B, _ = t.shape
        t = rearrange(t, "B N -> (B N)")
        t_embed = self.t_embedder(t, x.dtype)
        t_embed = rearrange(t_embed, "(B N) D -> B N D", B=B)

        t_embed = t_embed.repeat_interleave(self.config.block_size, dim=1) # (B, 1024, config.hidden_size)

        x_embed = self.x_embedder(x) + self.pos_embed # (B, 1024, config.hidden_size)
        x_t_embed = self.x_t_embedder(x_t) + t_embed + self.pos_embed

        if self.config.use_condition:
            assert y is not None
            y_embed = self.y_embedder(y) # (B, 256, config.hidden_size)
            y_embed = y_embed.repeat_interleave(self.config.block_size, dim=1) # (B, 1024, config.hidden_size)
            x_t_embed = x_t_embed + y_embed

        x = torch.cat([x_embed, x_t_embed], dim=1)

        for block in self.blocks:
            x = block(x, mask=self.mask)
        
        x = self.final_layer(x)

        return x[:, self.config.seq_len:, :]
    
    def forward_test_null_condition(self, x_t, t, prefix):
        assert self.training is False

        B = x_t.shape[0]

        t = t.unsqueeze(0)
        t = rearrange(t, "B N -> (B N)")
        t_embed = self.t_embedder(t, x_t.dtype)
        t_embed = rearrange(t_embed, "(B N) D -> B N D", B=B)
        t_embed = t_embed.repeat_interleave(self.config.block_size, dim=1)

        if prefix.shape[1] == 0:
            curr_pos = 0
            x = self.x_t_embedder(x_t) + t_embed + self.pos_embed[:, curr_pos:curr_pos+self.config.block_size, :]

            # mask = torch.zeros(self.config.block_size, self.config.block_size, device=x_t.device, dtype=x_t.dtype).unsqueeze(0).unsqueeze(0)
            for block in self.blocks:
                x = block(x, mask=None)
            x = self.final_layer(x)

            return x
        else:
            curr_pos = prefix.shape[1]
            x_t_embed = self.x_t_embedder(x_t) + t_embed + self.pos_embed[:, curr_pos:curr_pos+self.config.block_size, :] 

            x_embed = self.x_embedder(prefix) + self.pos_embed[:, :curr_pos, :]
            x = torch.cat([x_embed, x_t_embed], dim=1)

            # mask = torch.zeros(curr_pos + self.config.block_size, curr_pos + self.config.block_size, device=x_t.device, dtype=x_t.dtype).unsqueeze(0).unsqueeze(0)
            for block in self.blocks:
                x = block(x, mask=None)
            x = self.final_layer(x)

            return x[:, curr_pos:, :]

    def block_wise_noising(self, x):
        """
        add noise to x, the same noise for each block with size 4
        x: (B, 1024, 1024)
        """
        B, seq_len, _ = x.shape
        block_size = self.config.block_size
        num_blocks = seq_len // block_size

        # sample block-wise timesteps
        timesteps = torch.randint(0, 1000, (B, num_blocks), dtype=torch.int64, device=x.device)

        # sample block-wise noise and repeat within each block
        noise = torch.randn_like(x, device=x.device, dtype=x.dtype)

        # expand timesteps to per-token and flatten for scheduler API
        timesteps_token = timesteps.repeat_interleave(block_size, dim=1)  # (B, seq_len)
        x_flat = rearrange(x, "b n d -> (b n) d")
        noise_flat = rearrange(noise, "b n d -> (b n) d")
        t_flat = rearrange(timesteps_token, "b n -> (b n)")

        noisy_flat = self.train_scheduler.add_noise(x_flat, noise_flat, t_flat)
        target_flat = self.train_scheduler.get_velocity(x_flat, noise_flat, t_flat)

        noisy_latents = rearrange(noisy_flat, "(b n) d -> b n d", b=B, n=seq_len)
        target = rearrange(target_flat, "(b n) d -> b n d", b=B, n=seq_len)

        return noisy_latents, target, timesteps


# class HybridDiT_256(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.x_embedder = nn.Linear(config.in_channels, config.hidden_size)
#         self.x_t_embedder = nn.Linear(config.in_channels, config.hidden_size)

#         self.y_embedder = nn.Linear(config.intern_hidden_size, config.hidden_size)
#         self.t_embedder = TimestepEmbedder(config.hidden_size)

#         self.pos_embed = nn.Parameter(torch.randn(1, config.seq_len, config.hidden_size), requires_grad=False)
#         pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(config.seq_len ** 0.5))
#         self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

#         self.blocks = nn.ModuleList([
#             HybridBlock(config.hidden_size, config.num_heads, mlp_ratio=4) for _ in range(config.depth)
#         ])

#         self.final_layer = nn.Linear(config.hidden_size, config.out_channels)

#         self.register_buffer("mask", self._create_attn_mask(config.seq_len))

#     def _create_attn_mask(self, seq_len):
#         mask = torch.eye(2 * seq_len)
#         mask[seq_len+1:, :seq_len-1] = torch.tril(torch.ones(seq_len-1, seq_len-1))
#         mask[mask == 0] = float('-inf')
#         mask[mask == 1] = 0

#         mask = mask.unsqueeze(0).unsqueeze(0)

#         return mask
    
#     def forward(self, x, x_t, y, t):
#         assert self.training
#         B, L, _ = x.shape
#         x_embed = self.x_embedder(x) # (B, 256, config.hidden_size)
#         x_t_embed = self.x_t_embedder(x_t) # (B, 256, config.hidden_size)
#         y_embed = self.y_embedder(y) # (B, 256, config.hidden_size)
#         t = rearrange(t, "b n -> (b n)")
#         t_embed = self.t_embedder(t, x.dtype)
#         t_embed = rearrange(t_embed, "(b n) c -> b n c", b=B)

#         x_t_embed = x_t_embed + y_embed + t_embed

#         x_embed = x_embed + self.pos_embed
#         x_t_embed = x_t_embed + self.pos_embed

#         x = torch.cat([x_embed, x_t_embed], dim=1)
#         for block in self.blocks:
#             x = block(x, mask=self.mask)
        
#         x = self.final_layer(x)

#         return x[:, self.config.seq_len:, :]
    
#     def block_wise_noising(self, x, train_scheduler):
#         B, L, _ = x.shape
#         timesteps = torch.randint(0, 1000, (B, L), dtype=torch.int64, device=x.device)
#         noise = torch.randn_like(x, device=x.device, dtype=x.dtype)

#         t_flat = rearrange(timesteps, "B L -> (B L)")
#         x_flat = rearrange(x, "B L D -> (B L) D")
#         noise_flat = rearrange(noise, "B L D -> (B L) D")

#         noisy_flat = train_scheduler.add_noise(x_flat, noise_flat, t_flat)
#         target_flat = train_scheduler.get_velocity(x_flat, noise_flat, t_flat)

#         noisy_latents = rearrange(noisy_flat, "(B L) D -> B L D", B=B, L=L)
#         target = rearrange(target_flat, "(B L) D -> B L D", B=B, L=L)

#         return noisy_latents, target, timesteps

if __name__ == "__main__":
    from omegaconf import OmegaConf
    from diffusers import DDPMScheduler

    config = OmegaConf.load("config/overfit_1024/null_condition.yaml")
    model = HybridDiT(config.hybrid_dit)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params / 1e6:.2f}M")

    B = 3
    x = torch.randn(B, 1024, 1024)
    x_t = torch.randn(B, 1024, 1024)
    t = torch.randint(0, 1000, (B, 256))
    y = None

    out = model(x, x_t, t, y)
    print(out.shape)
    
    # out = model(x, x_t, y, t)
    # print(out.shape)

    noisy_latents, target, timesteps = model.block_wise_noising(x)
    # print(noisy_latents.shape, target.shape, timesteps.shape)