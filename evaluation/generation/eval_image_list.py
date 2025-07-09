import json, os
import torch.multiprocessing as mp
import threading
from tqdm import tqdm
import torch
import time
from PIL import Image
import torch.nn as nn
from transformers import AutoModel, AutoProcessor
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
import numpy as np
import hpsv2
import ImageReward as RM
import t2v_metrics
import pandas as pd
import typer
import numpy as np

app = typer.Typer()

# ------------------------ aesthetic score v2.5--------------------
@torch.no_grad()
def build_aesthetic_predictor_v2_5(gpu_id):
    model, preprocessor = convert_v2_5_from_siglip(
        low_cpu_mem_usage=False,
        trust_remote_code=True,
        cache_dir="/openseg_blob/v-junwenchen/huggingface_cache"
    )
    model = model.to(torch.bfloat16).to(f"cuda:{gpu_id}")
    return model, preprocessor

@torch.no_grad()
def predict_aesthetic_score(model, preprocessor, pil_image, gpu_id):
    pixel_values = (
        preprocessor(images=pil_image, return_tensors="pt")
        .pixel_values.to(torch.bfloat16)
        .to(f"cuda:{gpu_id}")
    )
    # predict aesthetic score
    with torch.inference_mode():
        score = model(pixel_values).logits.squeeze().float().cpu().numpy()
    return float(score)

# ------------------------ lion aesthetic score --------------------
class LaionAesthetic(nn.Module):
    def __init__(self, input_size=768):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

def build_aesthetic_predictor(gpu_id):
    laion_aesthetic = LaionAesthetic()
    laion_aesthetic.load_state_dict(torch.load('/openseg_blob/jhy/cache_ckpt/sac+logos+ava1-l14-linearMSE.pth', 'cpu'))
    laion_aesthetic.to(f"cuda:{gpu_id}")
    laion_aesthetic.eval()

    return laion_aesthetic

@torch.no_grad()
def laion_aesthetic_scorer(gpu_id, image, clip_processor, clip_model, laion_aesthetic):
    image_inputs = clip_processor(
        images=image,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(f"cuda:{gpu_id}")
    
    with torch.no_grad():
        image_embs = clip_model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
        return laion_aesthetic(image_embs).cpu().flatten().item()
    

# ---------------------- run --------------------------
def run_task(task_list, gpu_id, score_method):

    if score_method == "aes_v2_5": 
        # init aesthetic predictor
        print("Initializing Aesthetic Predictor")
        aes_v25_pipe, aes_v25_preprocessor = build_aesthetic_predictor_v2_5(gpu_id)
        
    if score_method == "hpsv2":
        # init hpsv2
        print("Initializing HPSv2")
    
    if score_method == "image_reward":
        # init image reward model
        print("Initializing ImageReward")
        rm_model = RM.load("ImageReward-v1.0", device=f"cuda:{gpu_id}")

    if score_method == "laion":
        # init laion aesthetic predictor
        print("Initializing LAION Aesthetic Predictor")
        clip_model_name = "openai/clip-vit-large-patch14"
        laion_pipe = build_aesthetic_predictor(gpu_id)
        laion_clip_model = AutoModel.from_pretrained(clip_model_name).eval().to(f"cuda:{gpu_id}")
        laion_clip_processor = AutoProcessor.from_pretrained(clip_model_name)
        
    if score_method == "vqascore":
        # init vqascore
        print("Initializing VQAScore")
        vqamodel = t2v_metrics.VQAScore(model='clip-flant5-xxl', device=f"cuda:{gpu_id}")


    # init result dict
    result_dict = {}

    for task in tqdm(task_list, total=len(task_list)):
        image_file, text_prompt = task

        if not os.path.exists(image_file):
            continue

        # load image
        try:
            image = Image.open(image_file).convert('RGB')
        except:
            print(f"Error loading image {image_file}")
            continue

        if score_method == "aes_v2_5":
            # get aesthetic score
            aes_score = predict_aesthetic_score(aes_v25_pipe, aes_v25_preprocessor, image, gpu_id)
            result_dict[image_file] = {"aesthetic_predictor_v2_5": aes_score}

        if score_method == "laion":
            # get laion aesthetic score
            laion_aesthetic_score = laion_aesthetic_scorer(gpu_id, image, laion_clip_processor, laion_clip_model, laion_pipe)
            laion_aesthetic_score = float(laion_aesthetic_score)
            result_dict[image_file] = {"laion_aesthetic_predictor": laion_aesthetic_score}
            
        if score_method == "hpsv2":
            # get hpsv2 score
            hpsv2_score = hpsv2.score(image, text_prompt, hps_version="v2.1")[0]
            hpsv2_score = float(hpsv2_score)
            result_dict[image_file] = {"hpsv2": hpsv2_score}
                
        if score_method == "image_reward":
            # get image reward score
            image_reward_score = rm_model.score(text_prompt, image)
            image_reward_score = float(image_reward_score)
            result_dict[image_file] = {"image_reward_score": image_reward_score}
            
        if score_method == "vqascore":
            # get vqascore
            vqascore = vqamodel(images=[image_file], texts=[text_prompt])
            vqascore = float(vqascore)
            result_dict[image_file] = {"vqa_score": vqascore}
    return result_dict

    
@app.command()
def main(csv_file, image_folder, gpu_id, score_method, output_name):
    csv_file = pd.read_csv(csv_file)
    task_list = [(os.path.join(image_folder, f'prompt_{i}_image_0.jpeg'), prompt) for i, prompt in enumerate(csv_file['partImagePrompt'])]
    result_dict = run_task(task_list, gpu_id, score_method)
    with open(output_name, "w") as f:
        json.dump(result_dict, f, indent=4)
    print(f"Saved results to {output_name}")

def debug():
    methods = ['aes_v2_5', 'laion', 'hpsv2', 'image_reward', 'vqascore']
    for m in methods:
        main(
            '/home/doch/exp/diffusion_rl_plan/evals-tools-main/data/unused/debug.csv',
            '/openseg_blob/doch/exp/dalle_vs_flux/images_dalle_en/',
            0,
            m,
            f'data/debug_dalle_{m}.json')

def get_mean(score_file):
    with open(score_file, 'r') as f:
        result_dict = json.load(f)
    scores = []
    for k in result_dict:
        for s in result_dict[k]:
            scores.append(result_dict[k][s])
    avg = np.array(scores).mean()
    return avg

def get_all_results():
    sets = ['neutrals'] # ['downs', 'ups', 'neutrals']
    metrics = ['aes_v2_5', 'laion', 'hpsv2', 'image_reward', 'vqascore']
    methods = ['fluxpro'] #['dalle', 'flux']
    for s in sets:
        print(s)
        for m in metrics:
            print(m, end=' ')
            for l in methods:
                f = f'data/scores/{s}_{l}_{m}.json'
                v = get_mean(f)
                print(v, end=' ')
            print('\n')

if __name__ == "__main__":
    # app()
    # get_mean(score_file = 'data/downs_dalle_hpsv2.json')
    # get_mean(score_file = 'data/downs_flux_hpsv2.json')
    get_all_results()
    