accelerate launch \
--config_file config/accelerate_config/deepspeed \
--main_process_port 30002 \
--num_processes 8 \
runner/yuqi/joint_hat.py \
--config config/yuqi/joint_hat.yaml