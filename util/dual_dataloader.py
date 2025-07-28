
# ------------------------------------------------------------------------------------------------
# ---------------------------- Understanding dataloader, LLaVAMix665K ----------------------------
# ------------------------------------------------------------------------------------------------
import os
import json
import torch
import random
from model.janus.models import VLChatProcessor
from PIL import Image
from torch.utils.data import DataLoader


class LLaVAMix665K(torch.utils.data.Dataset):
    def __init__(self, img_path, ann_path):
        self.img_path = img_path
        self.ann_path = ann_path
        self.data = json.load(open(ann_path, "r"))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        if "image" in data:
            # load image
            img_path = os.path.join(self.img_path, data["image"])
            # load Q&A pair
            num_qa_pair = len(data["conversations"]) // 2
            qa_index = random.randint(0, num_qa_pair - 1)
            assert data["conversations"][2*qa_index]["from"] == "human"
            assert data["conversations"][2*qa_index+1]["from"] == "gpt"
            question = data["conversations"][2*qa_index]["value"]
            answer = data["conversations"][2*qa_index+1]["value"]

            if "<image>\n" in question:
                question = question.replace("<image>\n", "")
            elif "\n<image>" in question:
                question = question.replace("\n<image>", "")

            item = {
                "question": question,
                "answer": answer,
                "image": img_path,
            }
            return item
        else:
            item = {
                "question": None,
                "answer": None,
            }
            return item

def create_labels_with_padding_handling(input_ids, sft_formats, tokenizer):
    """
    处理padding的精确方法
    """
    batch_size = input_ids.shape[0]
    labels = torch.full_like(input_ids, -100)  # 初始化为-100
    
    for i in range(batch_size):
        sft_format = sft_formats[i]
        
        # 分离prompt和answer
        assistant_start = sft_format.find("<|Assistant|>:")
        if assistant_start != -1:
            prompt = sft_format[:assistant_start + len("<|Assistant|>:")]
            answer = sft_format[assistant_start + len("<|Assistant|>:"):]
            
            # 编码prompt部分
            # prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
            
            # 编码answer部分
            answer_tokens = tokenizer.encode(answer, add_special_tokens=False)
            
            # 在input_ids中找到answer开始的位置
            input_tokens = input_ids[i].tolist()
            
            # 找到answer tokens的位置（考虑padding）
            answer_start_idx = None
            for j in range(len(input_tokens) - len(answer_tokens) + 1):
                # 检查是否匹配，同时确保不是padding
                if (input_tokens[j:j+len(answer_tokens)] == answer_tokens and 
                    input_tokens[j] != tokenizer.pad_token_id):
                    answer_start_idx = j
                    break
            
            if answer_start_idx is not None:
                # 设置labels，只对answer部分计算loss
                labels[i, answer_start_idx:answer_start_idx+len(answer_tokens)] = input_ids[i, answer_start_idx:answer_start_idx+len(answer_tokens)]
    
    return labels


def get_dataloader_und(config):
    vl_chat_processor = VLChatProcessor.from_pretrained(config.janus_1b_path)
    config = config.data
    img_path = config.understanding.img_path
    ann_path = config.understanding.ann_path

    def collate_fn_llava(batch):
        pixel_values = []
        conversations = []
        answers = []
        sft_formats = []
        for item in batch:
            if "image" not in item:
                continue
            # load image and convert to pixel values
            try:
                image = [Image.open(item["image"]).convert("RGB")]
                pixel_values.append(vl_chat_processor.image_processor(image).pixel_values)
            except Exception as e:
                print(f"Error loading image: {e}")
                continue

            # tokenize the conversation
            question = item["question"]
            answer = item["answer"]
            image = item["image"]
            conversation = [
                {
                    "role": "<|User|>",
                    "content": "<begin_of_image>" + "<image_placeholder>"*config.num_img_token + "<end_of_image>" + f"\n{question}",
                    "images": [image],
                },
                {"role": "<|Assistant|>", "content": ""},
            ]
            conversations.append(conversation)
            answers.append(answer)

        for conversation, answer in zip(conversations, answers):
            sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
                conversations = conversation,
                sft_format    = vl_chat_processor.sft_format,
                system_prompt = vl_chat_processor.system_prompt,
            )

            sft_format += answer
            sft_format += vl_chat_processor.tokenizer.eos_token
            sft_formats.append(sft_format)

        tokenizer_output = vl_chat_processor.tokenizer(
            sft_formats,
            return_tensors = "pt",
            padding        = "max_length",
            padding_side   = "right",
            truncation     = True,
            max_length     = config.max_seq_length,
        )
        input_ids = tokenizer_output["input_ids"]
        attention_mask = tokenizer_output["attention_mask"]
        labels = create_labels_with_padding_handling(input_ids, sft_formats, vl_chat_processor.tokenizer)

        pixel_values = torch.cat(pixel_values, dim=0)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    dataloader = DataLoader(
        LLaVAMix665K(img_path, ann_path),
        batch_size  = config.understanding.batch_size,
        shuffle     = True,
        num_workers = config.understanding.num_workers,
        pin_memory  = True,
        drop_last   = True,
        collate_fn  = collate_fn_llava,
    )

    return dataloader

# ------------------------------------------------------------------------------------------------
# -------------------------------- Generation dataloader, BLIP-3o --------------------------------
# ------------------------------------------------------------------------------------------------
import glob
import webdataset as wds
import torchvision.transforms as pth_transforms


def get_dataloader_gen(config):
    vl_chat_processor = VLChatProcessor.from_pretrained(config.janus_1b_path)
    config = config.data
    urls = []
    for path in config.generation.wds_path:
        urls.extend(glob.glob(os.path.join(path, "*.tar")))

    print(f"Found tar files: {len(urls)}")

    preprocess_gen = pth_transforms.Compose([
        pth_transforms.Resize(config.generation.img_size, max_size=None),
        pth_transforms.CenterCrop(config.generation.img_size),
        pth_transforms.ToTensor(),
        pth_transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
    ])

    # def process_sample(sample):
    #     image = sample["jpg"]
    #     prompt = sample["txt"]

    #     conversation = [
    #         {
    #             "role": "User",
    #             "content": prompt,
    #         },
    #         {"role": "Assistant", "content": ""},
    #     ]
    #     sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
    #         conversations=conversation,
    #         sft_format=vl_chat_processor.sft_format,
    #         system_prompt="",
    #     )
    #     prompt = sft_format + vl_chat_processor.image_start_tag

    #     tokenizer_output = vl_chat_processor.tokenizer(
    #         prompt,
    #         return_tensors = "pt",
    #         padding        = "max_length",
    #         padding_side   = "left",
    #         truncation     = True,
    #         max_length     = config.max_seq_length - config.num_img_token,
    #     )
    #     input_ids = tokenizer_output["input_ids"]
    #     attention_mask = tokenizer_output["attention_mask"]

    #     transformed_image = preprocess_gen(image)

    #     return {
    #         "pixel_value": transformed_image,
    #         "input_ids": input_ids,
    #         "attention_mask": attention_mask,
    #     }

    def process_img(image):
        try:
            if image is None:
                return None
            transformed_image = preprocess_gen(image)
            return {"pixel_value": transformed_image}
        except Exception as e:
            print(f"图像处理错误: {e}")
            return None

    def process_prompt(prompt):
        try:
            if prompt is None or not isinstance(prompt, str):
                return None
            conversation = [
                {
                    "role": "User",
                    "content": prompt,
                },
                {"role": "Assistant", "content": ""},
            ]
            sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format=vl_chat_processor.sft_format,
                system_prompt="",
            )
            prompt = sft_format + vl_chat_processor.image_start_tag

            tokenizer_output = vl_chat_processor.tokenizer(
                prompt,
                return_tensors = "pt",
                padding        = "max_length",
                padding_side   = "left",
                truncation     = True,
                max_length     = config.max_seq_length - config.num_img_token,
            )
            input_ids = tokenizer_output["input_ids"]
            attention_mask = tokenizer_output["attention_mask"]

            return {"input_ids": input_ids, "attention_mask": attention_mask}
        except Exception as e:
            print(f"文本处理错误: {e}")
            return None

    def collate_fn_gen(batch):
        """处理batch, 过滤掉None值"""
        valid_batches = []
        for item in batch:
            if item is not None and len(item) == 2:
                img_data, prompt_data = item
                if img_data is not None and prompt_data is not None:
                    valid_batches.append(item)
        
        if len(valid_batches) == 0:
            # 如果没有有效数据，返回空batch
            return {
                "pixel_value": torch.empty(0, 3, config.generation.img_size, config.generation.img_size),
                "input_ids": torch.empty(0, config.max_seq_length - config.num_img_token),
                "attention_mask": torch.empty(0, config.max_seq_length - config.num_img_token),
            }
        
        # 分离图像和文本数据
        img_batches = [item[0] for item in valid_batches]
        prompt_batches = [item[1] for item in valid_batches]
        
        # 堆叠数据
        pixel_values = torch.stack([item["pixel_value"] for item in img_batches])
        input_ids = torch.cat([item["input_ids"] for item in prompt_batches], dim=0)
        attention_mask = torch.cat([item["attention_mask"] for item in prompt_batches], dim=0)
        
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    gen_wds_dataset = (
        wds.WebDataset(urls, resampled=True, shardshuffle=True, nodesplitter=None)
        .shuffle(config.generation.buffer_size)
        .decode("pil", handler=wds.ignore_and_continue)
        .to_tuple("jpg", "txt")
        .map_tuple(process_img, process_prompt)
    )

    dataloader = DataLoader(
        gen_wds_dataset, 
        batch_size  = config.generation.batch_size,
        num_workers = config.generation.num_workers,
        collate_fn  = collate_fn_gen
    )

    return dataloader


class InfiniteIterator:
    """无限迭代器，自动重新初始化"""
    def __init__(self, dataloader, name):
        self.dataloader = dataloader
        self.name = name
        self.iterator = iter(dataloader)
        self.epoch_count = 0
    
    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.epoch_count += 1
            print(f"{self.name} 第 {self.epoch_count} 轮结束，重新初始化...")
            self.iterator = iter(self.dataloader)
            return next(self.iterator)
    
    def __iter__(self):
        return self