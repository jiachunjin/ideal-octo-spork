from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


qwen_vl_path = "/data/phd/jinjiachun/ckpt/Qwen/Qwen2.5-VL-7B-Instruct"

processor = AutoProcessor.from_pretrained(qwen_vl_path)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    qwen_vl_path, torch_dtype="auto", device_map="auto"
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "/data/phd/jinjiachun/codebase/connector/asset/kobe.png",
            },
            {"type": "text", "text": "Do you know the person in the image?"},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)