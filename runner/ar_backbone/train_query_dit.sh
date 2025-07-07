CUDA_VISIBLE_DEVICES=4,6 accelerate launch \
--config_file config/ddp/deepspeed \
--main_process_port 30002 \
--num_processes 2 \
runner/ar_backbone/train_query_dit.py \
--config config/ar_backbone/query_dit.yaml