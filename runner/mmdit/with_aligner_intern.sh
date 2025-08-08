accelerate launch \
--config_file config/accelerate_config/deepspeed \
--main_process_port 30002 \
--num_processes 2 \
runner/mmdit/with_aligner_intern.py \
--config config/mmdit/with_aligner_intern.yaml