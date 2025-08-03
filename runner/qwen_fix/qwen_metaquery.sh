accelerate launch \
--config_file config/accelerate_config/deepspeed \
--main_process_port 30002 \
--num_processes 8 \
runner/qwen_fix/qwen_metaquery.py \
--config config/qwen_fix/qwen_metaquery.yaml