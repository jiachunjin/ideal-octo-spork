import torch
import torch.nn as nn

from einops import rearrange
from timm.layers.mlp import Mlp
from diffusers import DDPMScheduler
from model.dit.standard_dit import TimestepEmbedder, LabelEmbedder
from model.dit.hybrid_dit import Attention

def modulate_per_token(x, shift, scale):
    return x * (1 + scale) + shift

class HybridBlock_AdaLN(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = Mlp(hidden_size, hidden_size * mlp_ratio, hidden_size)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, mask, kv_cache=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        
        # 注意力计算
        normed_x = modulate_per_token(self.norm1(x), shift_msa, scale_msa)
        if kv_cache is not None:
            attn_out, new_cache = self.attn(normed_x, mask, kv_cache)
            x = x + gate_msa * attn_out
        else:
            attn_out = self.attn(normed_x, mask)
            x = x + gate_msa * attn_out
            new_cache = None
        
        # MLP计算
        x = x + gate_mlp * self.mlp(modulate_per_token(self.norm2(x), shift_mlp, scale_mlp))

        if kv_cache is not None:
            return x, new_cache
        else:
            return x

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate_per_token(self.norm_final(x), shift, scale)
        x = self.linear(x)

        return x

class HybridDiT_Class(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.x_embedder = nn.Linear(config.in_channels, config.hidden_size)
        self.x_t_embedder = nn.Linear(config.in_channels, config.hidden_size)
        self.t_embedder = TimestepEmbedder(config.hidden_size)
        self.y_embedder = LabelEmbedder(config.num_classes, config.hidden_size)

        self.pos_embed = nn.Parameter(torch.randn(1, config.seq_len, config.hidden_size), requires_grad=True)

        self.blocks = nn.ModuleList([
            HybridBlock_AdaLN(config.hidden_size, config.num_heads, mlp_ratio=4) for _ in range(config.depth)
        ])

        self.final_layer = FinalLayer(config.hidden_size, config.out_channels)

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

    def forward(self, x, x_t, t, y):
        print(x.shape, x_t.shape, t.shape, y.shape)
        exit(0)
        assert self.training
        B, _ = t.shape
        t = rearrange(t, "B N -> (B N)")
        t_embed = self.t_embedder(t, x.dtype)
        t_embed = rearrange(t_embed, "(B N) D -> B N D", B=B)
        t_embed = t_embed.repeat_interleave(self.config.block_size, dim=1) # (B, 1024, D)

        y_embed = self.y_embedder(y) # (B, D)
        y_embed = y_embed.unsqueeze(1) # (B, 1, D)

        c = t_embed + y_embed # (B, 1024, D)
        c = torch.cat([torch.zeros_like(c), c], dim=1) # (B, 2048, D)

        x_embed = self.x_embedder(x) + self.pos_embed
        x_t_embed = self.x_t_embedder(x_t) + self.pos_embed
        x = torch.cat([x_embed, x_t_embed], dim=1)

        for block in self.blocks:
            x = block(x, c, mask=self.mask)

        x = self.final_layer(x, c)

        return x[:, self.config.seq_len:, :]

    def forward_test(self, x_t, t, prefix, y, kv_caches=None):
        """
        x_t: (B, 4, 1024)
        t: (1,)
        prefix: (B, ?, 1024)
        y: (B,)
        kv_caches: list of KV cache tuples for each layer, or None
        """
        assert self.training is False
        B = x_t.shape[0]

        t_embed = self.t_embedder(t, x_t.dtype).repeat(B, 1)
        y_embed = self.y_embedder(y)
        c = t_embed + y_embed
        c = c.unsqueeze(1).repeat(1, self.config.block_size, 1)

        curr_pos = prefix.shape[1]
        
        if kv_caches is not None and curr_pos > 0:
            # 使用KV cache时，只需要处理新的x_t tokens
            x_t_embed = self.x_t_embedder(x_t) + self.pos_embed[:, curr_pos:curr_pos+self.config.block_size, :]
            x = x_t_embed
            c_input = c
        else:
            # 第一次调用或不使用cache时，处理所有tokens
            x_embed = self.x_embedder(prefix) + self.pos_embed[:, :curr_pos, :]
            x_t_embed = self.x_t_embedder(x_t) + self.pos_embed[:, curr_pos:curr_pos+self.config.block_size, :]
            c_input = torch.cat([torch.zeros_like(x_embed), c], dim=1)
            x = torch.cat([x_embed, x_t_embed], dim=1)

        # 创建mask（如果需要的话，这里简化为None，因为我们使用因果注意力）
        mask = None
        
        new_kv_caches = []
        for i, block in enumerate(self.blocks):
            if kv_caches is not None:
                cache = kv_caches[i] if i < len(kv_caches) else None
                x, new_cache = block(x, c_input, mask=mask, kv_cache=cache)
                new_kv_caches.append(new_cache)
            else:
                x = block(x, c_input, mask=mask)
        
        x = self.final_layer(x, c_input)

        if kv_caches is not None:
            # 返回新生成的tokens和更新后的KV caches
            return x[:, -self.config.block_size:, :], new_kv_caches
        else:
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


if __name__ == "__main__":
    from omegaconf import OmegaConf
    from diffusers import DDPMScheduler

    config = OmegaConf.load("config/overfit_1024/null_condition.yaml")
    model = HybridDiT_Class(config.hybrid_dit)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params / 1e6:.2f}M")

    B = 3
    x = torch.randn(B, 1024, 1024)
    x_t = torch.randn(B, 1024, 1024)
    t = torch.randint(0, 1000, (B, 256))
    y = torch.randint(0, 1000, (B,))

    out = model(x, x_t, t, y)
    print(out.shape)
    
    # out = model(x, x_t, y, t)
    # print(out.shape)

    noisy_latents, target, timesteps = model.block_wise_noising(x)