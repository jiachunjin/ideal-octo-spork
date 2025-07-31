accelerate launch \
--config_file config/accelerate_config/deepspeed \
--main_process_port 30002 \
--num_processes 8 \
runner/ar_backbone/hat_training.py \
--config config/ar_backbone/hat_training.yaml