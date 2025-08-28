accelerate launch \
--config_file config/accelerate_config/deepspeed \
--main_process_port 30002 \
--num_processes 8 \
runner/clip_dit/res_hat.py \
--config config/clip_dit/res_hat.yaml