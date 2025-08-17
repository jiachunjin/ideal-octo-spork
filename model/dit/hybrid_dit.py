import torch
import torch.nn as nn
from model.dit.standard_dit import TimestepEmbedder

class HybridDiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.x_embedder = nn.Linear(1024, config.hidden_size)
        self.x_t_embedder = nn.Linear(1024, config.hidden_size)
        self.y_embedder = nn.Linear(config.intern_hidden_size, config.hidden_size)
        self.t_embedder = TimestepEmbedder(config.hidden_size)

        self.pos_embed = nn.Parameter(torch.randn(1, 2048, config.hidden_size))

    def forward(self, x, x_t, y, t):
        """
        only used for training
        x: (B, 1024, 1024), the clean clip tokens
        x_t: (B, 1024, 1024), the noisy clip tokens
        y: (B, 256, intern_hidden_size), the intern hidden states
        t: (B, 256), the timesteps, the timesteps are the same within one block
        """
        assert self.training
        x_embed = self.x_embedder(x) # (B, 1024, config.hidden_size)
        y_embed = self.y_embedder(y) # (B, 256, config.hidden_size)
        y_embed = y_embed.repeat_interleave(4, dim=1) # (B, 1024, config.hidden_size)
        t_embed = self.t_embedder(t, x.dtype)
        t_embed = t_embed.repeat_interleave(4, dim=1) # (B, 1024, config.hidden_size)
        if self.config.condition_injection == "beginning":
            x_t_embed = self.x_t_embedder(x_t) + y_embed + t_embed



        x = torch.cat([x, x_t], dim=1)

        for block in self.blocks:
            x = block(x)
        x = self.final_layer(x)



    
    def block_wise_noising(self, x):
        """
        add noise to x, the same noise for each block with size 4
        """
        ...

        

        
        

        # for block in self.blocks:
        #     x = block(x, c)
        # x = self.final_layer(x, c)
        # return x
        
        
        