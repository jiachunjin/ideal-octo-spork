accelerate launch \
--config_file config/accelerate_config/deepspeed \
--main_process_port 30002 \
--num_processes 8 \
runner/ar_backbone/train_diffhead.py \
--config config/ar_backbone/diff_head_and_ar_gen_vae.yaml