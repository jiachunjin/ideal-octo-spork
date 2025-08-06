accelerate launch \
--config_file config/accelerate_config/deepspeed \
--main_process_port 30002 \
--num_processes 8 \
runner/internvl/gen_only.py \
--config config/internvl/gen_only.yaml