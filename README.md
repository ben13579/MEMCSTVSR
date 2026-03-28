# MEMCSTVSR (Not finish)

MEMCSTVSR is built on top of [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio).

If you are not familiar with DiffSynth-Studio, please refer the [docs](https://github.com/modelscope/DiffSynth-Studio/tree/main/docs) of DiffSynth-Studio


## Installation

Clone the repository:

```bash
git clone https://github.com/ben13579/MEMCSTVSR.git
cd MEMCSTVSR
conda env create -f environment.yml
conda activate memcstvsr
pip install -r requirements.txt
pip install -e .
```

### Optional: FlashAttention

For NVIDIA CUDA environments, FlashAttention can improve speed and memory efficiency. Install it only after PyTorch and CUDA are already working correctly:

```bash
pip install flash-attn==2.8.3 \
  --no-build-isolation \
  --no-cache-dir
```

If the build fails, first verify that your local CUDA toolkit, compiler, and PyTorch CUDA version are compatible.

## Project Structure

The most important files and directories are:

- `examples/wanvideo/model_training/train.py`: STVSR DiT training entry point
- `examples/wanvideo/model_training/inference.py`: STVSR inference and evaluation entry point
- `examples/wanvideo/model_training/train_IND.py`: IND decoder training entry point
- `examples/wanvideo/model_training/full/Wan2.1-T2V-1.3B.sh`: STVSR DiT training command
- `examples/wanvideo/model_training/inference.sh`: example inference command
- `examples/wanvideo/model_training/train_IND.sh`: example IND training command
- `diffsynth/core/data/metadata_gennerator.py`: dataset metadata generator
- `diffsynth/core/data/gen_meta.sh`: example metadata generation shell script

## Dataset Preparation(目前的dataset是使用Adobe240,如果是在server使用可直接跳過這部分)

### Download dataset
Please refer [VideoINR](https://github.com/Picsart-AI-Research/VideoINR-Continuous-Space-Time-Super-Resolution) for Adobe240 dataset preperation.

### Expected Metadata Format

The dataset loader accepts metadata files in `json`, `jsonl`, or `csv` format. In practice, `jsonl` is the most convenient option for this project.

The relevant metadata keys are typically:

- `frames`: a list of relative frame paths
- `video`: a relative video path
- `prompt`: optional text prompt
- `start`: starting frame index for video-based clips
- `k`: number of frames to load from a video clip

Paths inside metadata are resolved relative to `--dataset_base_path`.

### Example: Frame-Sequence Metadata

```json
{"frames": ["clip_000/000.png", "clip_000/001.png", "clip_000/002.png", "clip_000/003.png", "clip_000/004.png", "clip_000/005.png", "clip_000/006.png", "clip_000/007.png", "clip_000/008.png"], "prompt": ""}
```

### Example: Video Metadata

```json
{"video": "videos/sample.mp4", "prompt": "", "start": 0, "k": 9}
```

### Generate Metadata Automatically

For frame folders:

```bash
python diffsynth/core/data/metadata_gennerator.py \
  --base_dir /path/to/dataset/train \
  --out_jsonl /path/to/dataset/train/metadata_vae.jsonl \
  --mode frames \
  --k 9 \
  --stride 8
```

For videos:

```bash
python diffsynth/core/data/metadata_gennerator.py \
  --base_dir /path/to/dataset/train \
  --out_jsonl /path/to/dataset/train/metadata_video.jsonl \
  --mode video \
  --k 9 \
  --stride 1
```

Notes:

- `--base_dir` is both the search root and the reference root used for relative paths.
- `frames` mode creates sliding windows over image sequences.
- `video` mode creates sliding windows over videos and writes `start` / `k` into the metadata.
- The helper script `diffsynth/core/data/gen_meta.sh` shows a ready-made pattern for generating train / valid / test metadata.

## Training Workflow

### 1. DiT Training

run

```bash
CUDA_VISIBLE_DEVICES=0 ./examples/wanvideo/model_training/full/Wan2.1-T2V-1.3B.sh
```

Important:

- The provided shell scripts contain hardcoded local dataset paths from the original development environment.
- Before running them, you should copy or edit the script and replace the dataset paths, metadata paths, and output paths with your own paths.

A minimal editable command looks like this:

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 \
  examples/wanvideo/model_training/train.py \
  --dataset_base_path /path/to/dataset/train \
  --dataset_metadata_path /path/to/dataset/train/metadata_overfit.jsonl \
  --val_dataset_base_path /path/to/dataset/valid \
  --val_dataset_metadata_path /path/to/dataset/valid/metadata_overfit.jsonl \
  --dataset_repeat 100 \
  --space_scale 4 \
  --time_scale 8 \
  --rope_mode 3d \
  --rope_method base \
  --height 480 \
  --width 832 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-1.3B:Wan2.1_VAE.pth" \
  --learning_rate 1e-5 \
  --num_epochs 400 \
  --batch_size 1 \
  --trainable_models dit \
  --remove_prefix_in_ckpt "pipe.dit." \
  --use_gradient_checkpointing \
  --save_steps 1000 \
  --validation_steps 1000 \
  --output_path ./models/train/overfit_3d
```

Key arguments:

- `--dataset_base_path`: root directory used to resolve file paths inside metadata
- `--dataset_metadata_path`: training metadata file
- `--val_dataset_base_path`: validation dataset root
- `--val_dataset_metadata_path`: validation metadata file
- `--space_scale`: spatial downsampling factor for the low-quality input
- `--time_scale`: temporal downsampling factor
- `--model_id_with_origin_paths`: upstream Wan model files to load
- `--trainable_models`: which module to train, typically `dit`
- `--output_path`: directory for checkpoints, training state, and validation outputs

Training outputs include:

- `step-*.safetensors`: model weights
- `training_state/step-*`: resumable Accelerate training states
- `validation/step-*/`: validation predictions and metrics

To resume training from the latest saved training state:

```bash
--resume_from_checkpoint model_path
```

### 2. Inference and Evaluation

run

```bash
CUDA_VISIBLE_DEVICES=0 bash examples/wanvideo/model_training/inference.sh
```

As with the training scripts, you must update the hardcoded local paths before running it.

A typical command looks like this:

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 \
  examples/wanvideo/model_training/inference.py \
  --dataset_base_path /path/to/dataset/valid \
  --dataset_metadata_path /path/to/dataset/valid/metadata_overfit.jsonl \
  --data_file_keys "image,video,frames" \
  --space_scale 4 \
  --time_scale 8 \
  --rope_mode 3d \
  --rope_method base \
  --height 480 \
  --width 832 \
  --model_paths '[["models/train/overfit_3d/step-9000.safetensors"]]' \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-T2V-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-1.3B:Wan2.1_VAE.pth" \
  --num_inference_steps 50 \
  --cfg_scale 1.0 \
  --sigma_shift 5.0 \
  --seed 0 \
  --fps 15 \
  --quality 5 \
  --tiled \
  --output_path ./models/train/overfit_3d/inference
```

The inference script writes:

- `sample-*/pred.mp4`: predicted videos
- `sample-*/gt.mp4`: ground-truth videos
- `sample-*/lq.mp4`: low-quality input videos
- `metrics.json`: aggregated PSNR / SSIM summary

### 3. IND Training

run

```bash
CUDA_VISIBLE_DEVICES=0 bash examples/wanvideo/model_training/train_IND.sh
```

Typical command:

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 \
  examples/wanvideo/model_training/train_IND.py \
  --dataset_base_path /path/to/dataset/train \
  --dataset_metadata_path /path/to/dataset/train/metadata_vae.jsonl \
  --val_dataset_base_path /path/to/dataset/valid \
  --val_dataset_metadata_path /path/to/dataset/valid/metadata_vae.jsonl \
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
  --vae_path ./models/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.1_VAE.safetensors \
  --torch_dtype bfloat16 \
  --ind_query_bsize 65536 \
  --mse_weight 0.0 \
  --lpips_weight 0.0 \
  --l1_weight 1.0 \
  --use_gradient_checkpointing \
  --save_steps 500 \
  --validation_steps 200 \
  --num_validation_samples 1 \
  --validation_fps 15 \
  --validation_quality 8 \
  --output_path ./models/train/wan_ind_concat
```

Optional experiment tracking with Weights & Biases:

```bash
--use_wandb \
--wandb_project wan-video-ind \
--wandb_run_name your-run-name
```

## Common Notes

- This repository is experimental STVSR work.


## Reference

- Upstream project: [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
- Additional Wan-oriented reference in this repository: [README_wan.md](README_wan.md)
