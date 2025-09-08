import torch
import torch.nn as nn

from diffusers import DDPMScheduler
from transformers import Qwen2Model
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from model.dit.standard_dit import TimestepEmbedder, DiTBlock, FinalLayer, get_2d_sincos_pos_embed

import types
from functools import partial
from transformers.utils import logging
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

logger = logging.get_logger(__name__)


def equip_internvl_res_hat(internvl, config):
    internvl.requires_grad_(False)
    hidden_size = config.dit_head.condition_channels
    input_dim = len(config.stages) * hidden_size

    # add mlp2 to internvl.language_model.model: Qwen2Model
    mlp2 = nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, hidden_size),
        nn.GELU(),
        nn.Linear(hidden_size, hidden_size)
    )
    mlp2.requires_grad_(True)
    internvl.language_model.model.mlp2 = mlp2

    # add new decoder layers to internvl.language_model.model: Qwen2Model
    qwen2_config = internvl.language_model.model.config
    layer_idx_curr = qwen2_config.num_hidden_layers
    additional_layers = nn.ModuleList(
        [Qwen2DecoderLayer(qwen2_config, layer_idx_curr + layer_idx) for layer_idx in range(config.num_hat)]
    )
    additional_layers.requires_grad_(True)
    internvl.language_model.model.additional_layers = additional_layers
    internvl.language_model.model.norm2 = nn.LayerNorm(hidden_size)
    internvl.language_model.model.norm2.requires_grad_(True)
    internvl.language_model.model.stages = config.stages

    # modify the forward function of internvl.language_model.model: Qwen2Model
    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        position_ids = None,
        past_key_values = None,
        inputs_embeds = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        cache_position = None,
        **flash_attn_kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    partial(decoder_layer.__call__, **flash_attn_kwargs),
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # #########################################
        # -------------- newly added --------------
        # #########################################
        # print(f"len(all_hidden_states)", len(all_hidden_states))

        collected_hidden_states = []
        for stage in self.stages:
            collected_hidden_states.append(all_hidden_states[stage])
        collected_hidden_states = torch.cat(collected_hidden_states, dim=-1)
        # print(collected_hidden_states.shape)

        hat_input = self.mlp2(collected_hidden_states)
        # print(hat_input.shape)

        hidden_states = hat_input

        for decoder_layer in self.additional_layers:
            # print(decoder_layer.self_attn.layer_idx)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    partial(decoder_layer.__call__, **flash_attn_kwargs),
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

        hidden_states = self.norm2(hidden_states)


        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    # 使用types.MethodType来正确绑定方法
    internvl.language_model.model.forward = types.MethodType(forward, internvl.language_model.model)

    # add diffusion head to internvl.language_model.model: Qwen2Model
    internvl.diff_head = DiT_Head(config.dit_head)
    num_params = sum(p.numel() for p in internvl.diff_head.parameters())
    print(f"Head parameters: {num_params / 1e6:.2f}M")
    internvl.diff_head.requires_grad_(True)

    print("InternVL modified!")

    

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
    model.config.layer_types.extend(["full_attention"] * num_hat)

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
        t = self.t_embedder(t, x.dtype)
        y = self.y_embedder(y)
        c = t + y

        for block in self.blocks:
            x = block(x, c)

        x = self.final_layer(x, c)

        return x