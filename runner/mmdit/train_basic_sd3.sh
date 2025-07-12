CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
--config_file config/ddp/deepspeed \
--main_process_port 30002 \
--num_processes 8 \
runner/mmdit/train_basic_sd3.py \
--config config/mmdit/basic_sd3.yaml