#!/usr/bin/env bash
set -euo pipefail
export WANDB_DIR='/home/bamboofan/wandb'

accelerate launch --num_processes 1 examples/wanvideo/model_training/train_IND.py \
  --dataset_base_path /project/bamboofan/Adobe240/frame/train \
  --dataset_metadata_path /project/bamboofan/Adobe240/frame/train/metadata_vae_overfit.jsonl \
  --data_file_keys "image,video,frames" \
  --val_data_file_keys "image,video,frames" \
  --space_scale 4 \
  --time_scale 8 \
  --height 480 \
  --width 832 \
  --num_frames 33 \
  --batch_size 1 \
  --num_epochs 20000 \
  --learning_rate 5e-5 \
  --weight_decay 0.01 \
  --save_steps 500 \
  --validation_output_subdir validation \
  --output_path ./models/train/wan_ind_concat \
  --vae_path ./models/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.1_VAE.safetensors \
  --torch_dtype bfloat16 \
  --ind_query_bsize 65536 \
  --mse_weight 0.0 \
  --lpips_weight 0.0 \
  --l1_weight 1.0 \
  --use_gradient_checkpointing \
  --use_wandb \
  --wandb_project wan-video-ind \
  --wandb_run_name wan-ind-overfit-concat \
  --wandb_log_steps 8 \
  --wandb_video_steps 80 \
  --validation_steps 200 \
  --num_validation_samples 1 \
  --validation_fps 15 \
  --validation_quality 8 \
  --val_dataset_base_path /project/bamboofan/Adobe240/frame/train \
  --val_dataset_metadata_path /project/bamboofan/Adobe240/frame/train/metadata_vae_overfit.jsonl \
  # --resume_from_checkpoint models/train/wan_ind_concat/step-500.safetensors \

  
  
  
