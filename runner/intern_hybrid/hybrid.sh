accelerate launch \
--config_file config/accelerate_config/deepspeed \
--main_process_port 30002 \
--num_processes 8 \
runner/intern_hybrid/hybrid.py \
--config config/intern_hybrid/hybrid.yaml