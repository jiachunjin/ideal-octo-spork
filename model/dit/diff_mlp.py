import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from einops import rearrange
from diffusers import DDPMScheduler

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq.to(self.mlp[0].weight.dtype))
        return t_emb

class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(
        self,
        channels
    ):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h

class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class SimpleMLPAdaLN(nn.Module):
    """
    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        grad_checkpointing=False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)

        self.input_proj = nn.Linear(in_channels, model_channels)

        # self.diffusion_pos_embed_learned = nn.Parameter(torch.zeros(1, 1024, 4096))
        # torch.nn.init.normal_(self.diffusion_pos_embed_learned, std=.02)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(
                model_channels,
            ))

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, c):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C] Tensor of outputs.
        """
        # c = rearrange(c, "(b n) d -> b n d", n=1024)
        # c += self.diffusion_pos_embed_learned
        # c = rearrange(c, "b n d -> (b n) d", n=1024)

        x = self.input_proj(x)
        t = self.time_embed(t)
        c = self.cond_embed(c)

        y = t + c
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.res_blocks:
                x = checkpoint(block, x, y)
        else:
            for block in self.res_blocks:
                x = block(x, y)
        return self.final_layer(x, y)

    def forward_with_cfg(self, x, t, c, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, c)
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


def equip_diffhead_query_with_janus(janus, config):
    diff_head = SimpleMLPAdaLN(
        in_channels    = config.diffhead.x_dim,
        model_channels = config.diffhead.hidden_size,
        out_channels   = config.diffhead.x_dim,
        z_channels     = config.diffhead.z_dim,
        num_res_blocks = config.diffhead.depth,
    )

    siglip16_aligner = nn.Sequential(
        nn.Linear(16, config.diffhead.z_dim),
        nn.GELU(),
        nn.Linear(config.diffhead.z_dim, config.diffhead.z_dim),
    )

    janus.requires_grad_(False)
    if config.tune_backbone:
        janus.language_model.model.requires_grad_(True)
    
    if getattr(config, "num_new_layers", None) is not None:
        janus.language_model.model, new_layer_indices = add_layers_with_initialization(
            janus.language_model.model,
            config.num_new_layers,
            initialization_method="copy_last",
        )
        for idx in new_layer_indices:
            layer = janus.language_model.model.layers[idx]
            layer.requires_grad_(True)

    janus.diff_head = diff_head
    janus.diff_head.requires_grad_(True)

    janus.siglip16_aligner = siglip16_aligner
    janus.siglip16_aligner.requires_grad_(True)

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

def add_diffhead_to_ar_model(ar_model, config):
    diff_head = SimpleMLPAdaLN(
        in_channels    = config.diffhead.x_dim,
        model_channels = config.diffhead.hidden_size,
        out_channels   = config.diffhead.x_dim,
        z_channels     = config.diffhead.z_dim,
        num_res_blocks = config.diffhead.depth,
    )

    clip_projector = nn.Sequential(
        nn.Linear(config.diffhead.x_dim, config.diffhead.z_dim),
        nn.GELU(),
        nn.Linear(config.diffhead.z_dim, config.diffhead.z_dim),
    )

    ar_model.requires_grad_(False)
    if config.tune_backbone:
        ar_model.language_model.model.requires_grad_(True)

    ar_model.diff_head = diff_head
    ar_model.diff_head.requires_grad_(True)

    ar_model.clip_projector = clip_projector
    ar_model.clip_projector.requires_grad_(True)

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

    return ar_model, train_scheduler

def add_diffhead_dit_to_ar_model(ar_model, config):
    from model.dit.lumina_next.nextdit import NextDiTCrossAttn, NextDiTCrossAttnConfig

    diff_head = SimpleMLPAdaLN(
        in_channels    = config.diffhead.x_dim,
        model_channels = config.diffhead.hidden_size,
        out_channels   = config.diffhead.x_dim,
        z_channels     = config.diffhead.z_dim,
        num_res_blocks = config.diffhead.depth,
    )
    num_parameters = sum(p.numel() for p in diff_head.parameters())
    print(f"diff_head has {num_parameters / 1e6} M parameters")

    clip_projector = nn.Sequential(
        nn.Linear(config.diffhead.x_dim, config.diffhead.z_dim),
        nn.GELU(),
        nn.Linear(config.diffhead.z_dim, config.diffhead.z_dim),
    )

    ar_model.requires_grad_(False)
    if config.tune_backbone:
        ar_model.language_model.model.requires_grad_(True)

    ar_model.diff_head = diff_head
    ar_model.diff_head.requires_grad_(True)

    ar_model.clip_projector = clip_projector
    ar_model.clip_projector.requires_grad_(True)

    dit_config = NextDiTCrossAttnConfig(**config.dit)
    dit = NextDiTCrossAttn(dit_config)
    num_parameters = sum(p.numel() for p in dit.parameters())
    print(f"dit has {num_parameters / 1e6} M parameters")

    ar_model.dit = dit
    ar_model.dit.requires_grad_(True)

    if hasattr(config, "num_hat"):
        from model.dit.dit_head import add_hat_to_intern
        ar_model.language_model.model = add_hat_to_intern(ar_model.language_model.model, config.num_hat)

        current_num_layers = len(ar_model.language_model.model.layers)
        new_layer_indices = range(current_num_layers - config.num_hat, current_num_layers)
        ar_model.requires_grad_(False)
        for idx in new_layer_indices:
            layer = ar_model.language_model.model.layers[idx]
            layer.requires_grad_(True)
        
        # number of trainable parameters in hat layers
        num_parameters = sum(p.numel() for p in ar_model.language_model.model.parameters() if p.requires_grad)
        print(f"number of trainable parameters in hat layers: {num_parameters / 1e6} M")


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

    return ar_model, train_scheduler

# ----------------------------------------------------------------------------------
# ------------------------ add new layers to janus backbone ------------------------
# ----------------------------------------------------------------------------------

import torch
import torch.nn as nn
from transformers import LlamaModel, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from typing import Optional


def add_layers_to_llama_model(
    model: LlamaModel, 
    num_new_layers: int, 
    config: Optional[LlamaConfig] = None,
    insert_position: Optional[int] = None
) -> LlamaModel:
    if config is None:
        config = model.config

    current_num_layers = len(model.layers)
    
    new_layers = []
    for i in range(num_new_layers):
        if insert_position is None:
            layer_idx = current_num_layers + i
        else:
            layer_idx = insert_position + i
        
        new_layer = LlamaDecoderLayer(config, layer_idx)
        new_layers.append(new_layer)
    
    if insert_position is None:
        model.layers.extend(new_layers)
    else:
        before_layers = model.layers[:insert_position]
        after_layers = model.layers[insert_position:]

        model.layers = nn.ModuleList(list(before_layers) + new_layers + list(after_layers))
    
    model.config.num_hidden_layers = len(model.layers)
    
    return model

def add_layers_with_initialization(
    model: LlamaModel,
    num_new_layers: int,
    initialization_method: str = "random",
    insert_position: Optional[int] = None
) -> LlamaModel:

    model = add_layers_to_llama_model(model, num_new_layers, insert_position=insert_position)
    
    current_num_layers = len(model.layers)
    if insert_position is None:
        new_layer_indices = range(current_num_layers - num_new_layers, current_num_layers)
    else:
        new_layer_indices = range(insert_position, insert_position + num_new_layers)
    
    if initialization_method == "random":
        pass
    
    elif initialization_method == "zeros":
        for idx in new_layer_indices:
            layer = model.layers[idx]
            for param in layer.parameters():
                if param.dim() > 1:  # weight
                    nn.init.zeros_(param)
                else:  # bias
                    nn.init.zeros_(param)
    
    elif initialization_method == "copy_last":
        if len(model.layers) > num_new_layers:
            last_layer_state = model.layers[-num_new_layers-1].state_dict()
            for idx in new_layer_indices:
                layer = model.layers[idx]
                layer.load_state_dict(last_layer_state)
    
    return model, new_layer_indices