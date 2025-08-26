import torch
import torch.nn as nn

from diffusers import DDPMScheduler
from transformers import Qwen2Model
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from model.dit.standard_dit import TimestepEmbedder, DiTBlock, FinalLayer, get_2d_sincos_pos_embed

def equip_internvl(internvl, config):
    # ----- add additional trainable LLM layers -----
    internvl.language_model.model = add_hat_to_intern(internvl.language_model.model, config.num_hat)
    
    current_num_layers = len(internvl.language_model.model.layers)
    new_layer_indices = range(current_num_layers - config.num_hat, current_num_layers)
    internvl.requires_grad_(False)
    for idx in new_layer_indices:
        layer = internvl.language_model.model.layers[idx]
        layer.requires_grad_(True)

    # ----- add diffusion head and scheduler -----
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

    internvl.diff_head = DiT_Head(config.dit_head)
    num_params = sum(p.numel() for p in internvl.diff_head.parameters())
    print(f"Head parameters: {num_params / 1e6:.2f}M")
    internvl.diff_head.requires_grad_(True)

    return internvl, train_scheduler

def add_hat_to_intern(model: Qwen2Model, num_hat: int):
    new_layers = []
    config = model.config
    current_num_layers = len(model.layers)

    for i in range(num_hat):
        layer_idx = current_num_layers + i
        new_layer = Qwen2DecoderLayer(config, layer_idx)
        new_layers.append(new_layer)

    model.layers.extend(new_layers)
    model.config.num_hidden_layers = len(model.layers)
    # model.config.layer_types.extend(["full_attention"] * num_hat)

    return model

class DiT_Head(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.x_embedder = nn.Linear(config.in_channels, config.hidden_size)
        self.y_embedder = nn.Linear(config.condition_channels, config.hidden_size)
        self.t_embedder = TimestepEmbedder(config.hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.num_tokens, config.hidden_size), requires_grad=False)
        self.blocks = nn.ModuleList([
            DiTBlock(config.hidden_size, config.num_heads, mlp_ratio=4) for _ in range(config.depth)
        ])
        self.final_layer = FinalLayer(config.hidden_size, config.out_channels)

        self.initialize_weights()

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
        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, y):
        x = self.x_embedder(x)
        t = self.t_embedder(t)
        y = self.y_embedder(y)
        c = t + y

        for block in self.blocks:
            x = block(x, c)

        x = self.final_layer(x, c)

        return x