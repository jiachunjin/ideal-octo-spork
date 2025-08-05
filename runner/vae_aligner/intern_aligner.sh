accelerate launch \
--config_file config/accelerate_config/deepspeed \
--main_process_port 30002 \
--num_processes 8 \
runner/vae_aligner/intern_aligner.py \
--config config/vae_aligner/intern_clip.yaml