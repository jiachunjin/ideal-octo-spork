accelerate launch \
--config_file config/ddp/deepspeed \
--main_process_port 30002 \
--num_processes 8 \
runner/vae_aligner/qwen_aligner.py \
--config config/vae_aligner/qwen_clip.yaml