import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from tqdm import tqdm, trange
from omegaconf import OmegaConf
from diffusers import DDIMScheduler

from model.overfit_1024.hybrid_dit_class import HybridDiT_Class
from model.mmdit import load_mmdit
from runner.mmdit.train_basic_sd3 import sample_sd3_5
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler

scheduler = DDIMScheduler(
    beta_schedule          = "scaled_linear",
    beta_start             = 0.00085,
    beta_end               = 0.012,
    num_train_timesteps    = 1000,
    clip_sample            = False,
    prediction_type        = "v_prediction",
    set_alpha_to_one       = True,
    steps_offset           = 1,
    trained_betas          = None,
    timestep_spacing       = "trailing",
    rescale_betas_zero_snr = True
)
scheduler.set_timesteps(50)

def block_sequence_to_row_major(sequence, n):
    """
    将2x2块排列的序列转换回行优先排列
    
    Args:
        sequence: 按2x2块排列的token序列
        n: 网格的边长 (nxn grid)
    
    Returns:
        按行优先排列的token序列
    
    Example:
        输入: [1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 13, 14, 11, 12, 15, 16], n=4
        输出: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    """
    if len(sequence) != n * n:
        raise ValueError(f"序列长度 {len(sequence)} 与网格大小 {n}x{n} 不匹配")
    
    if n % 2 != 0:
        raise ValueError(f"网格边长 {n} 必须是偶数")
    
    # 创建结果数组
    result = [0] * (n * n)
    
    # 计算每行/列有多少个2x2块
    blocks_per_row = n // 2
    
    # 遍历每个2x2块
    block_idx = 0
    for block_row in range(blocks_per_row):
        for block_col in range(blocks_per_row):
            # 每个2x2块包含4个元素
            block_start = block_idx * 4
            
            # 获取2x2块的4个元素
            top_left = sequence[block_start]
            top_right = sequence[block_start + 1]
            bottom_left = sequence[block_start + 2]
            bottom_right = sequence[block_start + 3]
            
            # 计算这些元素在原始网格中的位置
            # 2x2块的左上角在原始网格中的位置
            start_row = block_row * 2
            start_col = block_col * 2
            
            # 将元素放回原始位置
            result[start_row * n + start_col] = top_left          # 左上
            result[start_row * n + start_col + 1] = top_right     # 右上
            result[(start_row + 1) * n + start_col] = bottom_left # 左下
            result[(start_row + 1) * n + start_col + 1] = bottom_right # 右下
            
            block_idx += 1
    
    return result

def block_sequence_to_row_major_tensor(tokens: torch.Tensor, n: int) -> torch.Tensor:
    """
    将2x2块排列的token tensor转换回行优先排列
    
    Args:
        tokens: 形状为(B, L, D)的token tensor，其中L = n*n
        n: 网格的边长 (nxn grid)
    
    Returns:
        转换后的token tensor，形状仍为(B, L, D)
    
    Example:
        输入: tensor形状(2, 16, 768)，n=4
        在L维度上将2x2块排列转换为行优先排列
    """
    B, L, D = tokens.shape
    
    if L != n * n:
        raise ValueError(f"序列长度 {L} 与网格大小 {n}x{n} 不匹配")
    
    if n % 2 != 0:
        raise ValueError(f"网格边长 {n} 必须是偶数")
    
    # 创建索引映射
    indices = list(range(L))
    reordered_indices = block_sequence_to_row_major(indices, n)
    
    # 使用索引重排tensor
    reordered_tokens = tokens[:, reordered_indices, :]
    
    return reordered_tokens

@torch.no_grad()
def sample_imagenet():
    device = torch.device("cuda:7")
    dtype = torch.float16

    # ---------- load model ----------
    # exp_dir = "/data/phd/jinjiachun/experiment/clip_1024/0820_overfit_1024_null_condition_50000"
    exp_dir = "/data/phd/jinjiachun/experiment/clip_1024/0820_overfit_dog"
    # exp_dir = "/data/phd/jinjiachun/experiment/clip_1024/0821_imagenet_class_conditional"
    exp_name = exp_dir.split("/")[-1]
    step = 10000

    config = OmegaConf.load(os.path.join(exp_dir, "config.yaml"))
    # config = OmegaConf.load("config/overfit_1024/null_condition.yaml")
    model = HybridDiT_Class(config.hybrid_dit)
    model.load_state_dict(torch.load(os.path.join(exp_dir, f"hybrid_dit-{config.train.exp_name}-{step}"), map_location="cpu", weights_only=True))
    model = model.to(device, dtype).eval()

    mmdit_step = 140000
    # exp_dir = "/data/phd/jinjiachun/experiment/mmdit/0817_sd3_256"
    exp_dir = "/data/phd/jinjiachun/experiment/mmdit/0813_sd3_1024"
    config_path = os.path.join(exp_dir, "config.yaml")
    config_decoder = OmegaConf.load(config_path)
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(config_decoder.sd3_5_path, subfolder="scheduler")
    mmdit = load_mmdit(config_decoder)
    ckpt_path = os.path.join(exp_dir, f"mmdit-mmdit-{mmdit_step}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    mmdit.load_state_dict(ckpt, strict=False)
    mmdit = mmdit.to(device, dtype).eval()

    # load vae
    vae = AutoencoderKL.from_pretrained(config_decoder.sd3_5_path, subfolder="vae")
    vae.requires_grad_(False)
    vae = vae.to(device, dtype).eval()

    # ---------- do autoregressive diffusion sampling ----------
    def sample_one_clip_block(model, prefix, y, cfg_scale=2.0, kv_caches=None):
        B = y.shape[0]
        x_t = torch.randn((B, 4, 1024), device=device, dtype=dtype)

        if cfg_scale > 1.0:
            x_t = x_t.repeat(2, 1, 1)
            prefix = prefix.repeat(2, 1, 1)
            y_null = torch.full_like(y, fill_value=1000, dtype=torch.int64, device=device)
            y_cfg = torch.cat([y, y_null], dim=0)
            # 如果有KV cache，也需要复制
            if kv_caches is not None:
                kv_caches_cfg = []
                for layer_cache in kv_caches:
                    if layer_cache is not None:
                        cached_k, cached_v = layer_cache
                        # 复制KV cache用于CFG
                        cached_k_cfg = cached_k.repeat(2, 1, 1, 1) if cached_k is not None else None
                        cached_v_cfg = cached_v.repeat(2, 1, 1, 1) if cached_v is not None else None
                        kv_caches_cfg.append((cached_k_cfg, cached_v_cfg))
                    else:
                        kv_caches_cfg.append(None)
            else:
                kv_caches_cfg = None
        else:
            y_cfg = y
            kv_caches_cfg = kv_caches

        # 在扩散步骤中维护KV cache
        current_kv_caches = kv_caches_cfg
        
        for t in scheduler.timesteps:
            x_t = scheduler.scale_model_input(x_t, t)
            with torch.no_grad():
                t_tensor = torch.as_tensor([t], device=device)
                
                if current_kv_caches is not None:
                    noise_pred, updated_kv_caches = model.forward_test(x_t, t_tensor, prefix, y_cfg, current_kv_caches)
                    current_kv_caches = updated_kv_caches
                else:
                    noise_pred = model.forward_test(x_t, t_tensor, prefix, y_cfg)

                if cfg_scale > 1.0:
                    noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2, dim=0)
                    noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
                    x_out = x_t[:B]
                    x_out = scheduler.step(noise_pred, t, x_out).prev_sample
                    x_t = torch.cat([x_out, x_out], dim=0)
                else:
                    x_t = scheduler.step(noise_pred, t, x_t).prev_sample

        # 如果使用了KV cache，需要返回更新后的cache用于下次调用
        if kv_caches is not None and current_kv_caches is not None:
            # 从CFG的cache中提取原始的cache（取前B个）
            final_kv_caches = []
            for layer_cache in current_kv_caches:
                if layer_cache is not None:
                    cached_k, cached_v = layer_cache
                    # 只保留前B个batch的cache
                    cached_k_final = cached_k[:B] if cached_k is not None else None
                    cached_v_final = cached_v[:B] if cached_v is not None else None
                    final_kv_caches.append((cached_k_final, cached_v_final))
                else:
                    final_kv_caches.append(None)
            return x_t[:B], final_kv_caches
        else:
            return x_t[:B], None

    B = 4
    cfg_scale = 1.0
    label = 1000
    y = torch.tensor([label]*B, dtype=torch.int64, device=device)

    x_clip = torch.empty((B, 0, 1024), device=device, dtype=dtype)
    kv_caches = None  # 初始化KV cache
    
    for i in trange(256):
        x_clip_block, kv_caches = sample_one_clip_block(model, x_clip, y, cfg_scale=cfg_scale, kv_caches=kv_caches)
        x_clip = torch.cat([x_clip, x_clip_block], dim=1)

    print(x_clip.shape)
    
    x_clip = block_sequence_to_row_major_tensor(x_clip, 32)

    torch.save(x_clip, f"asset/clip_dit/{exp_name}_{step}_dog_clip.pt")

    samples = sample_sd3_5(
        transformer         = mmdit,
        vae                 = vae,
        noise_scheduler     = noise_scheduler,
        device              = device,
        dtype               = dtype,
        context             = x_clip,
        batch_size          = x_clip.shape[0],
        height              = 448,
        width               = 448,
        num_inference_steps = 25,
        guidance_scale      = 1.0,
        seed                = 42
    )
    print(samples.shape)

    import torchvision.utils as vutils
    sample_path = f"asset/clip_dit/{exp_name}_{step}_{label}_{cfg_scale}.png"
    vutils.save_image(samples, sample_path, nrow=4, normalize=False)
    print(f"Samples saved to {sample_path}")   


if __name__ == "__main__":
    sample_imagenet()