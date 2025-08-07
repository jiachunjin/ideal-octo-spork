accelerate launch \
--config_file config/accelerate_config/deepspeed \
--main_process_port 30002 \
--num_processes 8 \
runner/mmdit/aligner_free_intern.py \
--config config/mmdit/aligner_free_intern.yaml