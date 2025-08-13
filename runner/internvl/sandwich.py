import torch
from model.internvl.modeling_internvl_chat import InternVLChatModel

def dev():
    device = torch.device("cuda:0")
    intern_vl_1b_path = "/data/phd/jinjiachun/ckpt/OpenGVLab/InternVL3-1B"

    ar_model = InternVLChatModel.from_pretrained("/data/phd/jinjiachun/model/internvl/intern_vl_1b")

    print(ar_model)


if __name__ == "__main__":
    dev()





