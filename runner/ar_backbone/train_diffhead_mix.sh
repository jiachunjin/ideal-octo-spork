accelerate launch \
--config_file config/accelerate_config/deepspeed \
--main_process_port 30002 \
--num_processes 2 \
runner/ar_backbone/train_diffhead_mix.py \
--config config/ar_backbone/diffhead.yaml