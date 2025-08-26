import torch
import torch.nn as nn
from model.dit.standard_dit import TimestepEmbedder, DiTBlock, FinalLayer, get_2d_sincos_pos_embed

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