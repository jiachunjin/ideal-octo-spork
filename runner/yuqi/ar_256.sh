accelerate launch \
--config_file config/accelerate_config/deepspeed \
--main_process_port 30002 \
--num_processes 8 \
runner/yuqi/ar_256.py \
--config config/yuqi/ar_256.yaml