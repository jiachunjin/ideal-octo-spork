import torch
from model.internvl.modeling_internvl_chat import InternVLChatModel

@torch.no_grad()
def understand_generation():
    device = torch.device("cuda:7")
    dtype = torch.float16


if __name__ == "__main__":
    understand_generation()