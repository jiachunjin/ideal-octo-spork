import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from model.internvl.modeling_internvl_chat import InternVLChatModel
from transformers import AutoTokenizer


intern_vl_1b_path = "/data/phd/jinjiachun/ckpt/OpenGVLab/InternVL3-1B"

model = InternVLChatModel.from_pretrained(intern_vl_1b_path)
tokenizer = AutoTokenizer.from_pretrained(intern_vl_1b_path, trust_remote_code=True, use_fast=False)

generation_config = dict(max_new_tokens=1024, do_sample=True)

