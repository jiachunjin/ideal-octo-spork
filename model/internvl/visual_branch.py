import copy
import torch
import torch.nn as nn
from typing import Optional, Tuple, Callable
from transformers.processing_utils import Unpack
from transformers.utils import logging
from transformers.cache_utils import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, eager_attention_forward
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

logger = logging.get_logger(__name__)

class Qwen2Attention_Dual(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, pretrained_attn: Qwen2Attention):
        super().__init__()
        config = pretrained_attn.config
        self.config = pretrained_attn.config
        self.layer_idx = pretrained_attn.layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = pretrained_attn.q_proj
        self.k_proj = pretrained_attn.k_proj
        self.v_proj = pretrained_attn.v_proj
        self.o_proj = pretrained_attn.o_proj

        self.q_proj_visual = copy.deepcopy(pretrained_attn.q_proj).requires_grad_(True)
        self.k_proj_visual = copy.deepcopy(pretrained_attn.k_proj).requires_grad_(True)
        self.v_proj_visual = copy.deepcopy(pretrained_attn.v_proj).requires_grad_(True)
        self.o_proj_visual = copy.deepcopy(pretrained_attn.o_proj).requires_grad_(True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        print(f"hidden_shape.shape: {hidden_shape.shape}")
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        print(f"query_states.shape: {query_states.shape}")
        print(f"key_states.shape: {key_states.shape}")
        print(f"value_states.shape: {value_states.shape}")
        exit(0)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        sliding_window = None
        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=sliding_window,  # main diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights