#!/usr/bin/env bash
set -euo pipefail

accelerate launch --num_processes 1 examples/wanvideo/model_training/inference.py \
  --dataset_base_path /project/bamboofan/Adobe240/frame/train \
  --dataset_metadata_path /project/bamboofan/Adobe240/frame/train/metadata_overfit.jsonl \
  --data_file_keys "image,video,frames" \
  --space_scale 4 \
  --time_scale 8 \
  --rope_mode "3d" \
  --height 480 \
  --width 832 \
  --model_paths '[["models/train/overfit_3d/step-9000.safetensors"]]' \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-T2V-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-1.3B:Wan2.1_VAE.pth" \
  --output_path "./models/train/overfit_3d/inference" \
  --num_inference_steps 50 \
  --cfg_scale 1.0 \
  --sigma_shift 5.0 \
  --seed 0 \
  --fps 15 \
  --quality 5 \
  --dataset_repeat 1 \
  --tiled \
  --rope_method "base" \
  --rope_dype \
  --rope_base_grid_f 6 \
  --rope_base_grid_h 30 \
  --rope_base_grid_w 52 \
  # --dit_checkpoint "./models/train/overfit/step-6000.safetensors" \
  # --height 480 \
  # --width 832 \
