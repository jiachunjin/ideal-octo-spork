accelerate launch \
--config_file config/accelerate_config/deepspeed \
--main_process_port 30002 \
--num_processes 8 \
runner/diffusion_decoder/sd3_256.py \
--config config/diffusion_decoder/sd3_256.yaml