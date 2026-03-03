import argparse
import json
import os
import traceback

import numpy as np
import torch
from tqdm import tqdm

from diffsynth.core import STVSRDataset, load_state_dict
from diffsynth.core.data.operators import LoadAudio, LoadVideo, ImageCropAndResize, ToAbsolutePath
from diffsynth.diffusion import DiffusionTrainingModule
from diffsynth.diffusion.parsers import add_general_config, add_video_size_config
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.utils.data import save_video

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def wan_inference_parser():
    parser = argparse.ArgumentParser(
        description="STVSR inference script for WanVideoPipeline.",
        conflict_handler="resolve",
    )
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)

    # Model load
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--audio_processor_path", type=str, default=None, help="Path to audio processor.")
    parser.add_argument("--dit_checkpoint", type=str, required=True, help="Path to full DiT checkpoint.")
    parser.add_argument("--dit_checkpoint_strict", action="store_true", help="Load DiT checkpoint with strict=True.")
    parser.add_argument("--device", type=str, default=None, help="Inference device. Default: cuda if available else cpu.")
    parser.add_argument(
        "--torch_dtype",
        type=str,
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Torch dtype for pipeline.",
    )

    # Inference hyperparameters
    parser.add_argument("--num_inference_samples", type=int, default=None, help="Number of samples to infer. Default: all.")
    parser.add_argument("--start_index", type=int, default=0, help="Starting sample index.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--seed_stride", type=int, default=1, help="Seed increment per sample.")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps.")
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="CFG scale.")
    parser.add_argument("--sigma_shift", type=float, default=5.0, help="Scheduler sigma shift.")
    parser.add_argument("--denoising_strength", type=float, default=1.0, help="Denoising strength for video-to-video.")
    parser.add_argument("--tiled", action="store_true", help="Enable tiled VAE encode/decode.")
    parser.add_argument("--tile_size_h", type=int, default=30, help="Tile size H.")
    parser.add_argument("--tile_size_w", type=int, default=52, help="Tile size W.")
    parser.add_argument("--tile_stride_h", type=int, default=15, help="Tile stride H.")
    parser.add_argument("--tile_stride_w", type=int, default=26, help="Tile stride W.")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt for all samples.")
    parser.add_argument(
        "--drop_first_frame",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop first frame when pred has one extra frame.",
    )

    # Output
    parser.add_argument("--output_path", type=str, required=True, help="Output directory.")
    parser.add_argument("--fps", type=int, default=15, help="FPS for saved videos.")
    parser.add_argument("--quality", type=int, default=5, help="Quality for saved videos.")
    parser.add_argument(
        "--save_gt_lq",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save GT/LQ videos alongside prediction.",
    )
    parser.add_argument(
        "--save_metrics_json",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save aggregated metrics as JSON.",
    )
    return parser


def build_stvsr_dataset_for_inference(
    base_path,
    metadata_path,
    data_file_keys,
    args,
    repeat=1,
):
    return STVSRDataset(
        base_path=base_path,
        metadata_path=metadata_path,
        repeat=repeat,
        data_file_keys=data_file_keys.split(","),
        space_scale=args.space_scale,
        time_scale=args.time_scale,
        main_data_operator=STVSRDataset.load_clip_operators(
            base_path=base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
        ),
    )


def frame_to_float_np(frame):
    return np.asarray(frame).astype(np.float64) / 255.0


def compute_psnr(pred_frame, gt_frame):
    mse = np.mean((pred_frame - gt_frame) ** 2)
    if mse <= 1e-12:
        return 100.0
    return float(10.0 * np.log10(1.0 / mse))


def compute_ssim(pred_frame, gt_frame):
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_x = pred_frame.mean()
    mu_y = gt_frame.mean()
    sigma_x = ((pred_frame - mu_x) ** 2).mean()
    sigma_y = ((gt_frame - mu_y) ** 2).mean()
    sigma_xy = ((pred_frame - mu_x) * (gt_frame - mu_y)).mean()
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    if abs(denominator) <= 1e-12:
        return 1.0
    return float(numerator / denominator)


def compute_video_metrics(pred_video, gt_video):
    num_frames = min(len(pred_video), len(gt_video))
    if num_frames == 0:
        return None, None
    psnr_sum, ssim_sum = 0.0, 0.0
    for frame_id in range(num_frames):
        pred_frame = frame_to_float_np(pred_video[frame_id])
        gt_frame = frame_to_float_np(gt_video[frame_id])
        psnr_sum += compute_psnr(pred_frame, gt_frame)
        ssim_sum += compute_ssim(pred_frame, gt_frame)
    return psnr_sum / num_frames, ssim_sum / num_frames


def parse_torch_dtype(dtype_name: str):
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return dtype_map[dtype_name]


def resolve_device(device_arg: str | None):
    if device_arg is not None:
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_model_configs_for_inference(model_paths, model_id_with_origin_paths, fp8_models=None, offload_models=None, device="cpu"):
    parser_helper = DiffusionTrainingModule()
    return parser_helper.parse_model_configs(
        model_paths=model_paths,
        model_id_with_origin_paths=model_id_with_origin_paths,
        fp8_models=fp8_models,
        offload_models=offload_models,
        device=device,
    )


def build_pipeline(args):
    model_configs = parse_model_configs_for_inference(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        device=args.device,
    )
    tokenizer_config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/")
    if args.tokenizer_path is not None:
        tokenizer_config = ModelConfig(args.tokenizer_path)

    audio_processor_config = ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="wav2vec2-large-xlsr-53-english/")
    if args.audio_processor_path is not None:
        audio_processor_config = ModelConfig(args.audio_processor_path)

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=parse_torch_dtype(args.torch_dtype),
        device=args.device,
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
        audio_processor_config=audio_processor_config,
        rope_mode=args.rope_mode,
    )

    if not os.path.exists(args.dit_checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.dit_checkpoint}")
    state_dict = load_state_dict(args.dit_checkpoint, device="cpu")
    load_result = pipe.dit.load_state_dict(state_dict, strict=args.dit_checkpoint_strict)
    missing_keys, unexpected_keys = load_result
    print(
        "[Checkpoint] loaded:",
        args.dit_checkpoint,
        f"strict={args.dit_checkpoint_strict}, missing={len(missing_keys)}, unexpected={len(unexpected_keys)}",
    )
    if missing_keys:
        print("[Checkpoint] missing_keys (first 20):", missing_keys[:20])
    if unexpected_keys:
        print("[Checkpoint] unexpected_keys (first 20):", unexpected_keys[:20])
    return pipe


def build_metrics_summary(sample_records):
    valid_psnr = [s["psnr"] for s in sample_records if s.get("psnr") is not None]
    valid_ssim = [s["ssim"] for s in sample_records if s.get("ssim") is not None]
    successful = [s for s in sample_records if s.get("error") is None]

    return {
        "num_samples": len(sample_records),
        "num_successful": len(successful),
        "num_failed": len(sample_records) - len(successful),
        "mean_psnr": (sum(valid_psnr) / len(valid_psnr)) if valid_psnr else None,
        "mean_ssim": (sum(valid_ssim) / len(valid_ssim)) if valid_ssim else None,
        "samples": sample_records,
    }


def run_inference_loop(args, dataset, pipe):
    os.makedirs(args.output_path, exist_ok=True)

    dataset_len = len(dataset)
    start_index = max(0, args.start_index)
    if start_index >= dataset_len:
        raise ValueError(f"start_index={start_index} is out of range for dataset length={dataset_len}.")

    if args.num_inference_samples is None:
        end_index = dataset_len
    else:
        end_index = min(dataset_len, start_index + max(0, args.num_inference_samples))

    tile_size = (args.tile_size_h, args.tile_size_w)
    tile_stride = (args.tile_stride_h, args.tile_stride_w)

    sample_records = []
    sample_ids = list(range(start_index, end_index))
    print(f"[Inference] running samples: {len(sample_ids)} (start={start_index}, end={end_index})")

    with torch.no_grad():
        for idx in tqdm(sample_ids, desc="STVSR inference"):
            record = {
                "sample_id": idx,
                "prompt": "",
                "seed": args.seed + (idx - start_index) * args.seed_stride,
                "pred_path": None,
                "gt_path": None,
                "lq_path": None,
                "pred_num_frames": None,
                "gt_num_frames": None,
                "lq_num_frames": None,
                "psnr": None,
                "ssim": None,
                "error": None,
            }
            sample_output_dir = os.path.join(args.output_path, f"sample-{idx}")
            os.makedirs(sample_output_dir, exist_ok=True)
            try:
                data = dataset[idx]
                gt_video = data["GT"]
                lq_video = data["LQ"]
                prompt = data.get("prompt", "")
                if prompt is None:
                    prompt = ""
                record["prompt"] = str(prompt)

                pred_video = pipe(
                    prompt=record["prompt"],
                    negative_prompt=args.negative_prompt,
                    input_video=gt_video,
                    LQ_video=lq_video,
                    denoising_strength=args.denoising_strength,
                    seed=record["seed"],
                    rand_device=pipe.device,
                    height=gt_video[0].size[1],
                    width=gt_video[0].size[0],
                    num_frames=len(gt_video),
                    cfg_scale=args.cfg_scale,
                    num_inference_steps=args.num_inference_steps,
                    sigma_shift=args.sigma_shift,
                    tiled=args.tiled,
                    tile_size=tile_size,
                    tile_stride=tile_stride,
                    progress_bar_cmd=lambda x: x,
                    output_type="quantized",
                )

                if args.drop_first_frame and len(pred_video) > len(gt_video):
                    pred_video = pred_video[1:]

                pred_path = os.path.join(sample_output_dir, "pred.mp4")
                save_video(pred_video, pred_path, fps=args.fps, quality=args.quality)
                record["pred_path"] = pred_path

                if args.save_gt_lq:
                    gt_path = os.path.join(sample_output_dir, "gt.mp4")
                    lq_path = os.path.join(sample_output_dir, "lq.mp4")
                    save_video(gt_video, gt_path, fps=args.fps, quality=args.quality)
                    save_video(lq_video, lq_path, fps=args.fps, quality=args.quality)
                    record["gt_path"] = gt_path
                    record["lq_path"] = lq_path

                psnr_value, ssim_value = compute_video_metrics(pred_video, gt_video)
                record["psnr"] = psnr_value
                record["ssim"] = ssim_value
                record["pred_num_frames"] = len(pred_video)
                record["gt_num_frames"] = len(gt_video)
                record["lq_num_frames"] = len(lq_video)
            except Exception as error:
                record["error"] = f"{type(error).__name__}: {error}"
                print(f"[Inference][Error] sample={idx} -> {record['error']}")
                print(traceback.format_exc())
            sample_records.append(record)

    summary = build_metrics_summary(sample_records)
    if args.save_metrics_json:
        metrics_path = os.path.join(args.output_path, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[Metrics] saved: {metrics_path}")

    print(
        "[Summary]",
        f"num_samples={summary['num_samples']},",
        f"num_successful={summary['num_successful']},",
        f"num_failed={summary['num_failed']},",
        f"mean_psnr={summary['mean_psnr']},",
        f"mean_ssim={summary['mean_ssim']}",
    )
    return summary


def main():
    parser = wan_inference_parser()
    args = parser.parse_args()

    args.device = resolve_device(args.device)
    print(f"[Config] device={args.device}, dtype={args.torch_dtype}")

    dataset = build_stvsr_dataset_for_inference(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        data_file_keys=args.data_file_keys,
        args=args,
        repeat=args.dataset_repeat,
    )
    pipe = build_pipeline(args)
    run_inference_loop(args, dataset, pipe)


if __name__ == "__main__":
    main()
