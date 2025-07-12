CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
--config_file config/ddp/deepspeed \
--main_process_port 30002 \
--num_processes 8 \
runner/ar_backbone/train_query_dit.py \
--config config/ar_backbone/query_dit_vae_feature.yaml