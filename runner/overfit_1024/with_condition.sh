accelerate launch \
--config_file config/accelerate_config/deepspeed \
--main_process_port 30002 \
--num_processes 8 \
runner/overfit_1024/with_condition.py \
--config config/overfit_1024/with_condition.yaml