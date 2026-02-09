accelerate launch --num_processes 1 examples/wanvideo/model_training/train.py \
  --dataset_base_path /project/bamboofan/Adobe240/frame/train \
  --dataset_metadata_path /project/bamboofan/Adobe240/frame/train/metadata_1.jsonl \
  --space_scale 4 \
  --time_scale 8 \
  --height 480 \
  --width 832 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-1.3B:Wan2.1_VAE.pth" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.1-T2V-1.3B_full" \
  --trainable_models "dit" \
  --use_gradient_checkpointing \
  