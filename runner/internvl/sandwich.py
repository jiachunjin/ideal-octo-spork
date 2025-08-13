import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from transformers import Qwen2Model
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from model.internvl.modeling_internvl_chat import InternVLChatModel

def add_hat_to_intern(
    model: Qwen2Model,
    num_hat: int,
):
    config = model.config
    print(config)
    new_layers = []

    current_num_layers = len(model.layers)
    for i in range(num_hat):
        layer_idx = current_num_layers + i
        new_layer = Qwen2DecoderLayer(config, layer_idx)
        new_layers.append(new_layer)
    
    model.layers.extend(new_layers)
    model.config.num_hidden_layers = len(model.layers)
    model.config.layer_types.extend(["full_attention"] * num_hat)

    return model

def equip_internvl(ar_model, num_hat):
    ar_model.language_model.model = add_hat_to_intern(ar_model.language_model.model, num_hat)
    current_num_layers = len(ar_model.language_model.model.layers)
    new_layer_indices = range(current_num_layers - num_hat, current_num_layers)

    ar_model.requires_grad_(False)
    for idx in new_layer_indices:
        layer = ar_model.language_model.model.layers[idx]
        layer.requires_grad_(True)
    
    return ar_model

def dev():
    device = torch.device("cuda:0")
    intern_vl_1b_path = "/data/phd/jinjiachun/ckpt/OpenGVLab/InternVL3-1B"

    ar_model = InternVLChatModel.from_pretrained(intern_vl_1b_path)
    ar_model = equip_internvl(ar_model, 8)

    print(ar_model)



if __name__ == "__main__":
    dev()





