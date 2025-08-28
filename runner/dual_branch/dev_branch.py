import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from model.internvl.modeling_internvl_chat import InternVLChatModel
from model.internvl.visual_branch import Qwen2Attention_Dual
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention


internvl = InternVLChatModel.from_pretrained("/data/phd/jinjiachun/ckpt/OpenGVLab/InternVL3-8B")

# 把internvl中的Qwen2Attention替换为Qwen2Attention_Dual
def replace_qwen2_attention(model):
    """递归遍历模型，将所有的 Qwen2Attention 替换为 Qwen2Attention_Dual"""
    for name, module in model.named_children():
        if isinstance(module, Qwen2Attention):
            # 创建 Qwen2Attention_Dual 实例
            dual_attn = Qwen2Attention_Dual(module)
            # 替换原来的模块
            setattr(model, name, dual_attn)
            print(f"Replaced {name} with Qwen2Attention_Dual")
        else:
            # 递归处理子模块
            replace_qwen2_attention(module)


dtype = torch.bfloat16
device = torch.device("cuda:0")

internvl.requires_grad_(False)
replace_qwen2_attention(internvl)

num_para = sum(p.numel() for p in internvl.parameters() if p.requires_grad)
print(f"trainable num_para: {num_para}")

x = torch.randn(1, 3, 224, 224).to(device).to(dtype)



