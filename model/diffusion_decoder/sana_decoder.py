import math
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from diffusers import AutoencoderDC, FlowMatchEulerDiscreteScheduler, SanaTransformer2DModel
from diffusers.models.normalization import RMSNorm


class SanaDecoder(PreTrainedModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vae = AutoencoderDC.from_pretrained(config.sana0_6b_path, subfolder="vae")

        self.transformer = SanaTransformer2DModel.from_pretrained(config.sana0_6b_path, subfolder="transformer", torch_dtype=torch.bfloat16)

        self.connector = self._build_connector(config.clip_diffusion_connector)

        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(config.sana0_6b_path, subfolder="scheduler")
        self.vae.eval()
        self.vae.requires_grad_(False)

    def get_sigmas(self, timesteps, device, n_dim=4, dtype=torch.float32):
        sigmas = self.noise_scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.noise_scheduler.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    
    def load_transformer(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        m, u = self.transformer.load_state_dict(ckpt, strict=False)
        print(f"transformer missing keys: {m}")
        print(f"transformer unexpected keys: {u}")
        print(f"transformer loaded from {ckpt_path}")

    def load_connector(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        m, u = self.connector.load_state_dict(ckpt, strict=False)
        print(f"connector missing keys: {m}")
        print(f"connector unexpected keys: {u}")
        print(f"connector loaded from {ckpt_path}")
    
    def _build_connector(self, config):
        norm = RMSNorm(config.connector_out_dim, eps=1e-5, elementwise_affine=True)
        with torch.no_grad():
            norm.weight.fill_(math.sqrt(5.5))

        caption_channels = self.transformer.config.caption_channels
        connector = nn.Sequential(
            nn.Linear(config.clip_dim, caption_channels),
            nn.GELU(approximate="tanh"),
            nn.Linear(caption_channels, caption_channels),
            norm,
        )

        return connector