CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
--config_file config/ddp/deepspeed \
--main_process_port 30002 \
--num_processes 8 \
runner/vae_aligner/train_vae_aligner.py \
--config config/vae_aligner/siglip_flux.yaml