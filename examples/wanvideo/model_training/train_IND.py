import argparse
import json
import os
import random
import warnings
from typing import Optional

import accelerate
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from einops import rearrange
from tqdm import tqdm

from diffsynth.core import load_model, load_state_dict
from diffsynth.core.data import INDDataset
from diffsynth.core.data.operators import ImageCropAndResize, LoadAudio, LoadVideo, ToAbsolutePath
from diffsynth.diffusion import DiffusionTrainingModule, ModelLogger
from diffsynth.diffusion.parsers import (
    add_dataset_base_config,
    add_gradient_config,
    add_output_config,
    add_stvsr_dataset_config,
    add_training_config,
    add_validation_config,
    add_video_size_config,
)
from diffsynth.diffusion.base_pipeline import BasePipeline
from diffsynth.models.wan_video_ind import WanVideoIND
from diffsynth.models.wan_video_vae import WanVideoVAE
from diffsynth.utils.state_dict_converters.wan_video_vae import WanVideoVAEStateDictConverter
from diffsynth.utils.data import save_video

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TrainingProgress:
    def __init__(self):
        self.global_step = 0
        self.epoch_id = 0

    def state_dict(self):
        return {
            "global_step": self.global_step,
            "epoch_id": self.epoch_id,
        }

    def load_state_dict(self, state_dict):
        self.global_step = int(state_dict.get("global_step", 0))
        self.epoch_id = int(state_dict.get("epoch_id", 0))


def ind_parser():
    parser = argparse.ArgumentParser(
        description="Train WanVideoIND as a reconstruction decoder on top of frozen WanVideoVAE latents."
    )
    parser = add_dataset_base_config(parser)
    parser = add_stvsr_dataset_config(parser)
    parser = add_video_size_config(parser)
    parser = add_training_config(parser)
    parser = add_output_config(parser)
    parser.set_defaults(remove_prefix_in_ckpt="INDecoder.")
    parser = add_gradient_config(parser)
    parser = add_validation_config(parser)

    parser.add_argument(
        "--vae_path",
        type=str,
        default="models/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.1_VAE.safetensors",
        help="Path to Wan 2.1 VAE safetensors.",
    )
    parser.add_argument("--ind_checkpoint", type=str, default=None, help="Optional IND checkpoint for initialization.")
    parser.add_argument("--decoder_config", type=str, default=None, help="JSON string for WanVideoIND decoder_config.")
    parser.add_argument("--liif_config", type=str, default=None, help="JSON string for WanVideoIND liif_config.")
    parser.add_argument(
        "--torch_dtype",
        type=str,
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Torch dtype for the frozen VAE and trainable IND.",
    )
    parser.add_argument("--mse_weight", type=float, default=1.0, help="Weight for MSE reconstruction loss.")
    parser.add_argument("--l1_weight", type=float, default=0.0, help="Weight for L1 reconstruction loss.")
    parser.add_argument("--lpips_weight", type=float, default=1.0, help="Weight for LPIPS reconstruction loss.")
    parser.add_argument("--tiled", action="store_true", help="Enable tiled VAE encode.")
    parser.add_argument("--tile_size_h", type=int, default=30, help="Tile size H for VAE encode.")
    parser.add_argument("--tile_size_w", type=int, default=52, help="Tile size W for VAE encode.")
    parser.add_argument("--tile_stride_h", type=int, default=15, help="Tile stride H for VAE encode.")
    parser.add_argument("--tile_stride_w", type=int, default=26, help="Tile stride W for VAE encode.")
    parser.add_argument("--ind_query_bsize", type=int, default=8192, help="Chunk size for LIIF3D query in both training and inference.")
    parser.add_argument("--use_wandb", action="store_true", help="Enable wandb logging through accelerate trackers.")
    parser.add_argument("--wandb_project", type=str, default="wan-video-ind", help="wandb project name.")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="wandb run name.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="wandb entity.")
    parser.add_argument("--wandb_log_steps", type=int, default=10, help="Log scalar metrics every N steps.")
    parser.add_argument("--wandb_video_steps", type=int, default=200, help="Log visualizations every N steps.")
    parser.add_argument("--num_visualization_frames", type=int, default=3, help="Representative frames to log per sample.")
    parser.add_argument("--initialize_model_on_cpu", action="store_true", help="Initialize frozen VAE and IND on CPU first.")
    return parser


def parse_torch_dtype(dtype_name: str):
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return dtype_map[dtype_name]


def parse_optional_json(json_text: Optional[str]):
    if json_text is None or json_text == "":
        return None
    return json.loads(json_text)


def build_ind_dataset(base_path, metadata_path, data_file_keys, args, repeat=1):
    return INDDataset(
        base_path=base_path,
        metadata_path=metadata_path,
        repeat=repeat,
        data_file_keys=data_file_keys.split(","),
        space_scale=args.space_scale,
        time_scale=args.time_scale,
        main_data_operator=INDDataset.load_clip_operators(
            base_path=base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
        ),
    )


def parse_checkpoint_name(path):
    name = os.path.splitext(os.path.basename(path))[0]
    if name.startswith("step-"):
        return ("step", int(name.split("-", 1)[1]))
    if name.startswith("epoch-"):
        return ("epoch", int(name.split("-", 1)[1]))
    return None


def resolve_weight_checkpoint_path(output_path, checkpoint_arg):
    if checkpoint_arg is None:
        return None
    if checkpoint_arg != "latest":
        if not os.path.exists(checkpoint_arg):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_arg}")
        return checkpoint_arg

    if not os.path.isdir(output_path):
        raise FileNotFoundError(f"Output path does not exist: {output_path}")

    candidates = []
    for name in os.listdir(output_path):
        if not name.endswith(".safetensors"):
            continue
        path = os.path.join(output_path, name)
        parsed = parse_checkpoint_name(path)
        if parsed is None:
            continue
        kind, value = parsed
        kind_order = 1 if kind == "step" else 0
        candidates.append((value, kind_order, path))

    if not candidates:
        raise FileNotFoundError(f"No weights checkpoint found under: {output_path}")

    candidates.sort()
    return candidates[-1][2]


def strip_prefix_from_state_dict(state_dict, prefix):
    if prefix == "":
        return dict(state_dict)
    return {
        (key[len(prefix) :] if key.startswith(prefix) else key): value
        for key, value in state_dict.items()
    }


def normalize_ind_state_dict_for_load(state_dict, target_keys):
    candidate_prefixes = (
        "",
        "INDecoder.",
        "module.INDecoder.",
        "model.INDecoder.",
        "module.model.INDecoder.",
        "module.",
        "model.",
    )

    best_state_dict = dict(state_dict)
    best_overlap = len(set(best_state_dict.keys()) & target_keys)
    best_exact = sum(1 for key in best_state_dict if key in target_keys)

    for prefix in candidate_prefixes:
        candidate = strip_prefix_from_state_dict(state_dict, prefix)
        overlap = len(set(candidate.keys()) & target_keys)
        exact = sum(1 for key in candidate if key in target_keys)
        if overlap > best_overlap or (overlap == best_overlap and exact > best_exact):
            best_state_dict = candidate
            best_overlap = overlap
            best_exact = exact

    return best_state_dict


def load_ind_weights_if_needed(accelerator, model, args):
    resume_path = resolve_weight_checkpoint_path(args.output_path, args.resume_from_checkpoint)
    init_path = resume_path if resume_path is not None else args.ind_checkpoint
    if init_path is None:
        return

    state_dict = load_state_dict(init_path, device="cpu")
    target_state_dict = accelerator.unwrap_model(model).INDecoder.state_dict()
    state_dict = normalize_ind_state_dict_for_load(state_dict, set(target_state_dict.keys()))
    load_result = accelerator.unwrap_model(model).INDecoder.load_state_dict(state_dict, strict=False)
    if accelerator.is_main_process:
        source = "resume" if resume_path is not None else "init"
        print(
            f"[Checkpoint] loaded {source} weights from {init_path}; "
            "optimizer/scheduler/global_step restart from 0"
        )
        if load_result.missing_keys:
            print(f"[Checkpoint] missing_keys (first 20): {load_result.missing_keys[:20]}")
        if load_result.unexpected_keys:
            print(f"[Checkpoint] unexpected_keys (first 20): {load_result.unexpected_keys[:20]}")


def tensor_to_uint8_video(video):
    video = video.detach().float().clamp(-1, 1)
    video = ((video + 1.0) * 127.5).round().to(torch.uint8)
    return rearrange(video, "b c t h w -> b t h w c")


def build_visualization_grid(gt_video, pred_video, lq_video, num_frames):
    gt_uint8 = tensor_to_uint8_video(gt_video)[0]
    pred_uint8 = tensor_to_uint8_video(pred_video)[0]
    lq_frames = rearrange(lq_video[0], "c t h w -> t c h w")
    lq_frames = F.interpolate(
        lq_frames.float(),
        size=gt_video.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    lq_up = rearrange(lq_frames, "t c h w -> 1 c t h w")
    lq_uint8 = tensor_to_uint8_video(lq_up)[0]

    total_frames = gt_uint8.shape[0]
    frame_ids = torch.linspace(0, total_frames - 1, steps=min(total_frames, num_frames)).round().long().tolist()
    rows = []
    for frame_id in frame_ids:
        row = torch.cat(
            [lq_uint8[frame_id], pred_uint8[frame_id], gt_uint8[frame_id]],
            dim=1,
        )
        rows.append(row)
    grid = torch.cat(rows, dim=0).cpu().numpy()
    return grid


def validation_is_enabled(args):
    return args.val_dataset_base_path is not None and args.val_dataset_metadata_path is not None


def tensor_video_to_frame_list(video):
    frames = tensor_to_uint8_video(video)[0].cpu().numpy()
    return [frame for frame in frames]


class INDTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        vae_path,
        decoder_config=None,
        liif_config=None,
        torch_dtype=torch.bfloat16,
        device="cpu",
        tiled=False,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        ind_query_bsize=8192,
        mse_weight=1.0,
        l1_weight=1.0,
        lpips_weight=1.0,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
    ):
        super().__init__()
        try:
            import lpips
        except ImportError as error:
            raise ImportError("lpips is required for train_IND.py. Install it before running training.") from error

        if use_gradient_checkpointing_offload and not use_gradient_checkpointing:
            warnings.warn(
                "`--use_gradient_checkpointing_offload` implies gradient checkpointing; "
                "enabling `use_gradient_checkpointing` for WanVideoIND."
            )
            use_gradient_checkpointing = True

        self.runtime_device = device
        self.runtime_dtype = torch_dtype
        self.tiled = tiled
        self.tile_size = tile_size
        self.tile_stride = tile_stride
        self.ind_query_bsize = ind_query_bsize
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.lpips_weight = lpips_weight
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload

        self.preprocessor = BasePipeline(
            device=device,
            torch_dtype=torch_dtype,
            height_division_factor=16,
            width_division_factor=16,
            time_division_factor=4,
            time_division_remainder=1,
        )
        self.vae = load_model(
            WanVideoVAE,
            vae_path,
            torch_dtype=torch_dtype,
            device=device,
            state_dict_converter=WanVideoVAEStateDictConverter,
        )
        self.vae.requires_grad_(False)
        self.vae.eval()

        self.INDecoder = WanVideoIND(
            decoder_config=decoder_config,
            liif_config=liif_config,
        )
        self.lpips = lpips.LPIPS(net="alex")
        self.lpips.requires_grad_(False)
        self.lpips.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.vae.eval()
        self.lpips.eval()
        return self

    def preprocess_video(self, video):
        return self.preprocessor.preprocess_video(
            video,
            torch_dtype=self.runtime_dtype,
            device=self.runtime_device,
        )

    def encode_lq(self, lq_tensor):
        with torch.no_grad():
            latents = self.vae.encode(
                lq_tensor,
                device=self.runtime_device,
                tiled=self.tiled,
                tile_size=self.tile_size,
                tile_stride=self.tile_stride,
            ).to(dtype=self.runtime_dtype, device=self.runtime_device)
        return latents

    def compute_losses(self, pred_video, gt_video):
        loss_mse = F.mse_loss(pred_video.float(), gt_video.float())
        loss_l1 = F.l1_loss(pred_video.float(), gt_video.float())

        pred_frames = rearrange(pred_video, "b c t h w -> (b t) c h w")
        gt_frames = rearrange(gt_video, "b c t h w -> (b t) c h w")
        loss_lpips = self.lpips(pred_frames.float(), gt_frames.float()).mean()

        total_loss = self.mse_weight * loss_mse + self.l1_weight * loss_l1 + self.lpips_weight * loss_lpips
        return {
            "loss": total_loss,
            "loss_mse": loss_mse,
            "loss_l1": loss_l1,
            "loss_lpips": loss_lpips,
        }

    def forward(self, data, inputs=None, return_outputs=False):
        sample = data if inputs is None else inputs
        gt_video = self.preprocess_video(sample["GT"])
        lq_video = self.preprocess_video(sample["LQ"])
        output_size = (gt_video.shape[2], gt_video.shape[3], gt_video.shape[4])

        latents = self.encode_lq(lq_video)
        pred_video = self.INDecoder(
            latents,
            output_size=output_size,
            return_img=True,
            bsize=self.ind_query_bsize,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload,
        )
        losses = self.compute_losses(pred_video, gt_video)

        if return_outputs:
            losses["pred_video"] = pred_video.detach()
            losses["gt_video"] = gt_video.detach()
            losses["lq_video"] = lq_video.detach()
            losses["latents"] = latents.detach()
        return losses


def init_wandb_if_needed(accelerator, args, decoder_config, liif_config):
    if not args.use_wandb:
        return
    try:
        import wandb  # noqa: F401
    except ImportError as error:
        raise ImportError("wandb is not installed. Install it or disable --use_wandb.") from error

    accelerator.init_trackers(
        project_name=args.wandb_project,
        config={
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "dataset_base_path": args.dataset_base_path,
            "dataset_metadata_path": args.dataset_metadata_path,
            "vae_path": args.vae_path,
            "mse_weight": args.mse_weight,
            "l1_weight": args.l1_weight,
            "lpips_weight": args.lpips_weight,
            "space_scale": args.space_scale,
            "time_scale": args.time_scale,
            "torch_dtype": args.torch_dtype,
            "use_gradient_checkpointing": args.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": args.use_gradient_checkpointing_offload,
            "decoder_config": decoder_config,
            "liif_config": liif_config,
        },
        init_kwargs={
            "wandb": {
                "name": args.wandb_run_name,
                "entity": args.wandb_entity,
            }
        },
    )


def maybe_log_visualization(accelerator, model, sample, global_step, args):
    if not args.use_wandb or args.wandb_video_steps <= 0 or global_step % args.wandb_video_steps != 0:
        return
    if not accelerator.is_main_process:
        return

    import wandb

    unwrapped_model = accelerator.unwrap_model(model)
    was_training = unwrapped_model.training
    unwrapped_model.eval()
    try:
        with torch.no_grad():
            outputs = unwrapped_model(sample, return_outputs=True)
        grid = build_visualization_grid(
            outputs["gt_video"],
            outputs["pred_video"],
            outputs["lq_video"],
            args.num_visualization_frames,
        )
        tracker = accelerator.get_tracker("wandb", unwrap=True)
        tracker.log(
            {
                "train/sample_grid": wandb.Image(
                    grid,
                    caption="left: LQ (upsampled), middle: pred, right: GT",
                )
            },
            step=global_step,
        )
    finally:
        if was_training:
            unwrapped_model.train()


def run_ind_validation(accelerator, model, val_dataset, global_step, args):
    if val_dataset is None or args.validation_steps <= 0 or args.num_validation_samples <= 0:
        return

    world_size = accelerator.num_processes
    process_index = accelerator.process_index
    num_samples = min(len(val_dataset), args.num_validation_samples)
    if num_samples <= 0:
        return

    was_training = model.training
    model.eval()
    unwrapped_model = accelerator.unwrap_model(model)
    step_output_dir = os.path.join(args.output_path, args.validation_output_subdir, f"step-{global_step}")
    local_loss_sum = 0.0
    local_mse_sum = 0.0
    local_l1_sum = 0.0
    local_lpips_sum = 0.0
    local_count = 0
    local_psnr_sum = 0

    try:
        with torch.no_grad():
            for sample_id in range(process_index, num_samples, world_size):
                sample = val_dataset[sample_id]
                sample_output_dir = os.path.join(step_output_dir, f"sample-{sample_id}")
                os.makedirs(sample_output_dir, exist_ok=True)

                try:
                    outputs = unwrapped_model(sample, return_outputs=True)
                    pred_frames = tensor_video_to_frame_list(outputs["pred_video"])
                    save_video(
                        pred_frames,
                        os.path.join(sample_output_dir, "pred.mp4"),
                        fps=args.validation_fps,
                        quality=args.validation_quality,
                    )
                    save_video(
                        sample["GT"],
                        os.path.join(sample_output_dir, "gt.mp4"),
                        fps=args.validation_fps,
                        quality=args.validation_quality,
                    )
                    save_video(
                        sample["LQ"],
                        os.path.join(sample_output_dir, "lq.mp4"),
                        fps=args.validation_fps,
                        quality=args.validation_quality,
                    )
                    psnr = 10 * torch.log10(1 / outputs["loss_mse"]).item() if outputs["loss_mse"].item() > 0 else float("inf")
                    local_psnr_sum += psnr
                    local_loss_sum += outputs["loss"].item()
                    local_mse_sum += outputs["loss_mse"].item()
                    local_lpips_sum += outputs["loss_lpips"].item()
                    local_l1_sum += outputs["loss_l1"].item()
                    local_count += 1
                except Exception as error:
                    print(f"[Validation] step={global_step}, sample={sample_id}, process={process_index}, error={error}")

        local_stats = torch.tensor(
            [local_loss_sum, local_mse_sum, local_lpips_sum, float(local_count), local_psnr_sum, local_l1_sum],
            dtype=torch.float64,
            device=accelerator.device,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(local_stats, op=dist.ReduceOp.SUM)

        total_loss = local_stats[0].item()
        total_mse = local_stats[1].item()
        total_lpips = local_stats[2].item()
        total_count = int(round(local_stats[3].item()))
        total_psnr = local_stats[4].item()
        total_l1 = local_stats[5].item()
        mean_loss = (total_loss / total_count) if total_count > 0 else None
        mean_mse = (total_mse / total_count) if total_count > 0 else None
        mean_lpips = (total_lpips / total_count) if total_count > 0 else None
        mean_psnr = (total_psnr / total_count) if total_count > 0 else None
        mean_l1 = (total_l1 / total_count) if total_count > 0 else None

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            os.makedirs(step_output_dir, exist_ok=True)
            metrics = {
                "step": global_step,
                "num_samples": total_count,
                "mean_loss": mean_loss,
                "mean_mse": mean_mse,
                "mean_l1": mean_l1,
                "mean_lpips": mean_lpips,
                "mean_psnr": mean_psnr,
            }
            with open(os.path.join(step_output_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            print(
                f"[Validation] step={global_step}, samples={total_count}, "
                f"mean_loss={mean_loss}, mean_mse={mean_mse}, mean_l1={mean_l1}, mean_lpips={mean_lpips}, mean_psnr={mean_psnr}"
            )
            if args.use_wandb:
                accelerator.log(
                    {
                        "val/loss": mean_loss,
                        "val/loss_mse": mean_mse,
                        "val/loss_l1": mean_l1,
                        "val/loss_lpips": mean_lpips,
                        "val/global_step": global_step,
                        "val/psnr": mean_psnr,
                    },
                    step=global_step,
                )
    finally:
        if was_training:
            model.train()


def launch_ind_training(
    accelerator: Accelerator,
    dataset,
    model: INDTrainingModule,
    model_logger: ModelLogger,
    val_dataset,
    args,
):
    g = torch.Generator()
    if args.seed is not None:
        g.manual_seed(args.seed)

    def seed_worker(worker_id):
        if args.seed is not None:
            worker_seed = (args.seed + worker_id) % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            torch.manual_seed(worker_seed)

    optimizer = torch.optim.AdamW(
        model.trainable_modules(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: x,
        num_workers=args.dataset_num_workers,
        worker_init_fn=seed_worker,
        generator=g,
    )
    total_training_steps = max(1, args.num_epochs * len(dataloader)*2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_training_steps,
    )
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.runtime_device = accelerator.device
    unwrapped_model.preprocessor.device = accelerator.device
    unwrapped_model.preprocessor.torch_dtype = unwrapped_model.runtime_dtype

    load_ind_weights_if_needed(accelerator, model, args)

    global_step = 0 # Set to 500 to test loading from checkpoint. Change to 0 for normal training.
    for epoch_id in range(args.num_epochs):
        for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                batch_outputs = []
                for sample in batch:
                    batch_outputs.append(model(sample))

                loss = torch.stack([output["loss"] for output in batch_outputs]).mean()
                loss_mse = torch.stack([output["loss_mse"] for output in batch_outputs]).mean()
                loss_l1 = torch.stack([output["loss_l1"] for output in batch_outputs]).mean()
                loss_lpips = torch.stack([output["loss_lpips"] for output in batch_outputs]).mean()

                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()

                global_step += 1
                model_logger.num_steps = global_step

                if accelerator.is_main_process and global_step % 100 == 0:
                    print(
                        f"[Training] epoch={epoch_id}, step={global_step}, "
                        f"loss={loss.item():.6f}, mse={loss_mse.item():.6f}, l1={loss_l1.item():.6f}, lpips={loss_lpips.item():.6f}"
                    )

                if args.save_steps is not None and global_step % args.save_steps == 0:
                    model_logger.save_model(accelerator, model, f"step-{global_step}.safetensors")

                if args.use_wandb and args.wandb_log_steps > 0 and global_step % args.wandb_log_steps == 0:
                    accelerator.log(
                        {
                            "train/loss": loss.detach().item(),
                            "train/loss_mse": loss_mse.detach().item(),
                            "train/loss_l1": loss_l1.detach().item(),
                            "train/loss_lpips": loss_lpips.detach().item(),
                            "train/lr": scheduler.get_last_lr()[0],
                            "train/global_step": global_step,
                        },
                        step=global_step,
                    )

                if batch:
                    maybe_log_visualization(accelerator, model, batch[0], global_step, args)

                if (
                    val_dataset is not None
                    and args.validation_steps > 0
                    and global_step % args.validation_steps == 0
                ):
                    run_ind_validation(accelerator, model, val_dataset, global_step, args)

        if args.save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)

    model_logger.on_training_end(accelerator, model, args.save_steps)
    accelerator.end_training()


if __name__ == "__main__":
    args = ind_parser().parse_args()
    if args.seed is not None:
        set_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    decoder_config = parse_optional_json(args.decoder_config)
    liif_config = parse_optional_json(args.liif_config)
    torch_dtype = parse_torch_dtype(args.torch_dtype)

    accelerator = accelerate.Accelerator(
        log_with="wandb" if args.use_wandb else None,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    init_wandb_if_needed(accelerator, args, decoder_config, liif_config)

    dataset = build_ind_dataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        data_file_keys=args.data_file_keys,
        args=args,
        repeat=args.dataset_repeat,
    )
    val_dataset = None
    if validation_is_enabled(args):
        val_dataset = build_ind_dataset(
            base_path=args.val_dataset_base_path,
            metadata_path=args.val_dataset_metadata_path,
            data_file_keys=args.val_data_file_keys,
            args=args,
            repeat=1,
        )
    model = INDTrainingModule(
        vae_path=args.vae_path,
        decoder_config=decoder_config,
        liif_config=liif_config,
        torch_dtype=torch_dtype,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        tiled=args.tiled,
        tile_size=(args.tile_size_h, args.tile_size_w),
        tile_stride=(args.tile_stride_h, args.tile_stride_w),
        ind_query_bsize=args.ind_query_bsize,
        mse_weight=args.mse_weight,
        l1_weight=args.l1_weight,
        lpips_weight=args.lpips_weight,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    launch_ind_training(
        accelerator=accelerator,
        dataset=dataset,
        model=model,
        model_logger=model_logger,
        val_dataset=val_dataset,
        args=args,
    )
