CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
--config_file config/accelerate_config/deepspeed \
--main_process_port 30002 \
--num_processes 8 \
runner/rec_align/rec_only.py \
--config config/rec_align/both_task.yaml