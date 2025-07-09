# eval_image_list.py
eval_image_list里面包括
1. aesthetic_predictor_v2_5/laion_aesthetic_predictor/hpsv2/vqa_score

# geneval
git clone https://github.com/djghosh13/geneval.git
cd geneval
conda create -n geneval python=3.9 -y
conda activate geneval
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1
pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html
pip install mmengine==0.7.3

pip install pandas
pip install numpy==1.23.1

pip install open-clip-torch
pip install clip-benchmark

git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection; git checkout 2.x
pip install -v -e .

这个环境很难装，对cuda有特定要求，我都是用docker的

# DPG
git clone https://github.com/TencentQQGYLab/ELLA.git
cd ELLA
cp ~/project/ReasonGen-R1/benchmark/requirements-for-dpg_bench.txt .
conda create -n dpg_test python=3.9 -y
conda activate dpg_test
conda install conda-forge::fairseq -y
pip install -r evaluation/generation/DPG/requirement.txt