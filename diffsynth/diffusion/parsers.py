import argparse


def add_dataset_base_config(parser: argparse.ArgumentParser):
    parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--dataset_num_workers", type=int, default=0, help="Number of workers for data loading.")
    parser.add_argument("--data_file_keys", type=str, default="image,video,frames", help="Data file keys in the metadata. Comma-separated.")
    return parser

def add_stvsr_dataset_config(parser: argparse.ArgumentParser):
    parser.add_argument("--space_scale", type=float, default=4, help="Space scale for STVSR dataset, e.g., 4x. If provided, this will override the height and width config.")
    parser.add_argument("--time_scale", type=float, default=8, help="Time scale for STVSR dataset, e.g., 4x. If provided, this will override the num_frames config.")
    parser.add_argument("--downsample_indexes", type=str, default="0,1,-2,-1", help="Indexes of frames to be downsampled in the STVSR dataset. Comma-separated. Supports negative indexing.")
    parser.add_argument("--rope_mode", type=str, default="3d", choices=["3d", "4d"], help="ROPE mode for STVSR dataset. Supported options: '3d', '4d'.")
    parser.add_argument("--rope_method", type=str, default="base", choices=["base", "yarn"], help="RoPE frequency generation method for STVSR. Supported options: 'base' and 'yarn'.")
    parser.add_argument("--rope_dype", default=False, action=argparse.BooleanOptionalAction, help="Enable DyPE timestep-aware modulation for YaRN RoPE.")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="Base theta used when building dynamic 3D RoPE frequencies.")
    parser.add_argument("--rope_base_grid_f", type=int, default=None, help="Optional base temporal patch grid used by dynamic 3D RoPE.")
    parser.add_argument("--rope_base_grid_h", type=int, default=None, help="Optional base height patch grid used by dynamic 3D RoPE.")
    parser.add_argument("--rope_base_grid_w", type=int, default=None, help="Optional base width patch grid used by dynamic 3D RoPE.")
    return parser

def add_image_size_config(parser: argparse.ArgumentParser):
    parser.add_argument("--height", type=int, default=None, help="Height of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--max_pixels", type=int, default=1024*1024, help="Maximum number of pixels per frame, used for dynamic resolution.")
    return parser

def add_video_size_config(parser: argparse.ArgumentParser):
    parser.add_argument("--height", type=int, default=None, help="Height of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--max_pixels", type=int, default=1024*1024, help="Maximum number of pixels per frame, used for dynamic resolution.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames per video. Frames are sampled from the video prefix.")
    return parser

def add_model_config(parser: argparse.ArgumentParser):
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--fp8_models", default=None, help="Models with FP8 precision, comma-separated.")
    parser.add_argument("--offload_models", default=None, help="Models with offload, comma-separated. Only used in splited training.")
    return parser

def add_training_config(parser: argparse.ArgumentParser):
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size per process.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--task", type=str, default="sft", required=False, help="Task type.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for training.")
    return parser

def add_output_config(parser: argparse.ArgumentParser):
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--save_steps", type=int, default=None, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume training from a full training-state checkpoint directory, or use `latest`.")
    return parser

def add_lora_config(parser: argparse.ArgumentParser):
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.")
    parser.add_argument("--preset_lora_path", type=str, default=None, help="Path to the preset LoRA checkpoint. If provided, this LoRA will be fused to the base model.")
    parser.add_argument("--preset_lora_model", type=str, default=None, help="Which model the preset LoRA is fused to.")
    return parser

def add_gradient_config(parser: argparse.ArgumentParser):
    parser.add_argument("--use_gradient_checkpointing", default=False, action="store_true", help="Whether to use gradient checkpointing.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    return parser

def add_validation_config(parser: argparse.ArgumentParser):
    parser.add_argument("--val_dataset_base_path", type=str, default=None, help="Base path of the validation dataset.")
    parser.add_argument("--val_dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the validation dataset.")
    parser.add_argument("--val_data_file_keys", type=str, default="image,video,frames", help="Validation data file keys in metadata. Comma-separated.")
    parser.add_argument("--validation_steps", type=int, default=500, help="Run validation every N training steps.")
    parser.add_argument("--num_validation_samples", type=int, default=2, help="Max number of validation samples per validation run.")
    parser.add_argument("--validation_num_inference_steps", type=int, default=50, help="Inference steps used during validation generation.")
    parser.add_argument("--validation_seed", type=int, default=0, help="Random seed used during validation generation.")
    parser.add_argument("--validation_fps", type=int, default=15, help="FPS for saved validation videos.")
    parser.add_argument("--validation_quality", type=int, default=8, help="Quality for saved validation videos.")
    parser.add_argument("--validation_output_subdir", type=str, default="validation", help="Validation output folder under output_path.")
    parser.add_argument("--validation_cfg_scale", type=float, default=1.0, help="CFG scale used during validation generation.")
    parser.add_argument("--validation_sigma_shift", type=float, default=5.0, help="Sigma shift used during validation generation.")
    parser.add_argument("--validation_tiled", default=False, action="store_true", help="Whether to enable tiled VAE decode during validation.")
    return parser

def add_general_config(parser: argparse.ArgumentParser):
    parser = add_dataset_base_config(parser)
    parser = add_stvsr_dataset_config(parser)
    parser = add_model_config(parser)
    parser = add_training_config(parser)
    parser = add_output_config(parser)
    parser = add_lora_config(parser)
    parser = add_gradient_config(parser)
    parser = add_validation_config(parser)
    return parser
