import json
import torch, os, argparse, accelerate, warnings
import numpy as np
from tqdm import tqdm
from diffsynth.core import STVSRDataset
from diffsynth.core.data.operators import LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.diffusion import *
from diffsynth.utils.data import save_video
from accelerate import Accelerator
import random
from accelerate.utils import set_seed
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, audio_processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        rope_mode="3d",
    ):
        super().__init__()
        # Warning
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing is detected as disabled. To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing.")
            use_gradient_checkpointing = True
        
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        audio_processor_config = ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="wav2vec2-large-xlsr-53-english/") if audio_processor_path is None else ModelConfig(audio_processor_path)
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config, audio_processor_config=audio_processor_config, rope_mode=rope_mode)
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)
        
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        
    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        return inputs_shared
    
    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["GT"],
            "LQ_video": data["LQ"],
            "height": data["GT"][0].size[1],
            "width": data["GT"][0].size[0],
            "num_frames": len(data["GT"]),
            "rope_mode": self.pipe.rope_mode if hasattr(self.pipe, "rope_mode") else "3d",
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        if not self.pipe.scheduler.training:
            self.pipe.scheduler.set_timesteps(1000, training=True)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        # print(self.pipe.units)
        for unit in self.pipe.units:
            # print(unit)
            # if isinstance(unit, WanVideoUnit_ImageEmbedderFused):
            #     print(unit.take_over)
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        if "input_latents" not in inputs[0]:
            raise RuntimeError(
                "Missing `input_latents` before loss computation. "
                "Scheduler may be in inference mode; expected training mode with scheduler.training=True."
            )
        # print("[DBG] input keys after pipe:", inputs[0].keys())
        # print("[DBG] after forward inputs video shape:", inputs[0]["latents"].shape)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--audio_processor_path", type=str, default=None, help="Path to the audio processor. If provided, the processor will be used for Wan2.2-S2V model.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    return parser


def build_stvsr_dataset(
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
        special_operator_map={
            "animate_face_video": ToAbsolutePath(base_path) >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16)),
            "input_audio": ToAbsolutePath(base_path) >> LoadAudio(sr=16000),
        }
    )


def validation_is_enabled(args):
    return args.val_dataset_base_path is not None and args.val_dataset_metadata_path is not None


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


def run_stvsr_validation(
    accelerator,
    model,
    val_dataset,
    global_step,
    args,
):
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
    pipe = unwrapped_model.pipe
    original_scheduler_training = pipe.scheduler.training
    local_psnr_sum, local_ssim_sum, local_count = 0.0, 0.0, 0
    step_output_dir = os.path.join(args.output_path, args.validation_output_subdir, f"step-{global_step}")

    try:
        with torch.no_grad():
            for sample_id in range(process_index, num_samples, world_size):
                data = val_dataset[sample_id]
                gt_video = data["GT"]
                lq_video = data["LQ"]
                prompt = data.get("prompt", "")
                sample_output_dir = os.path.join(step_output_dir, f"sample-{sample_id}")
                os.makedirs(sample_output_dir, exist_ok=True)

                try:
                    pred_video = pipe(
                        prompt=prompt,
                        negative_prompt="",
                        input_video=gt_video,
                        LQ_video=lq_video,
                        # seed=args.validation_seed + sample_id,
                        seed=args.validation_seed,
                        rand_device=pipe.device,
                        height=gt_video[0].size[1],
                        width=gt_video[0].size[0],
                        num_frames=len(gt_video),
                        cfg_scale=args.validation_cfg_scale,
                        num_inference_steps=args.validation_num_inference_steps,
                        sigma_shift=args.validation_sigma_shift,
                        tiled=args.validation_tiled,
                        progress_bar_cmd=lambda x: x,
                        output_type="quantized",
                    )

                    save_video(pred_video, os.path.join(sample_output_dir, "pred.mp4"), fps=args.validation_fps, quality=args.validation_quality)
                    save_video(gt_video, os.path.join(sample_output_dir, "gt.mp4"), fps=args.validation_fps, quality=args.validation_quality)
                    save_video(lq_video, os.path.join(sample_output_dir, "lq.mp4"), fps=args.validation_fps, quality=args.validation_quality)

                    psnr_value, ssim_value = compute_video_metrics(pred_video, gt_video)
                    if psnr_value is not None and ssim_value is not None:
                        local_psnr_sum += psnr_value
                        local_ssim_sum += ssim_value
                        local_count += 1
                except Exception as error:
                    print(f"[Validation] step={global_step}, sample={sample_id}, process={process_index}, error={error}")

        local_stats = torch.tensor([local_psnr_sum, local_ssim_sum, float(local_count)], dtype=torch.float64, device=accelerator.device)
        gathered = accelerator.gather_for_metrics(local_stats) if hasattr(accelerator, "gather_for_metrics") else accelerator.gather(local_stats)
        if gathered.ndim == 1:
            gathered = gathered.unsqueeze(0)

        total_psnr = gathered[:, 0].sum().item()
        total_ssim = gathered[:, 1].sum().item()
        total_count = int(round(gathered[:, 2].sum().item()))
        mean_psnr = (total_psnr / total_count) if total_count > 0 else None
        mean_ssim = (total_ssim / total_count) if total_count > 0 else None

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            os.makedirs(step_output_dir, exist_ok=True)
            metrics = {
                "step": global_step,
                "num_samples": total_count,
                "mean_psnr": mean_psnr,
                "mean_ssim": mean_ssim,
            }
            with open(os.path.join(step_output_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            print(f"[Validation] step={global_step}, samples={total_count}, mean_psnr={mean_psnr}, mean_ssim={mean_ssim}")
    finally:
        pipe.scheduler.set_timesteps(1000, training=True)
        if original_scheduler_training and not pipe.scheduler.training:
            pipe.scheduler.set_timesteps(1000, training=True)
        if was_training:
            model.train()


def launch_training_task_with_validation(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    val_dataset: torch.utils.data.Dataset=None,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
):
    batch_size = 1
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        batch_size = args.batch_size
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs

    g = torch.Generator()
    g.manual_seed(args.seed)
    def seed_worker(worker_id):
        worker_seed = (args.seed + worker_id) % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: x,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
    )
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    global_step = 0
    best_loss = float("inf")
    for epoch_id in range(num_epochs):
        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                losses = []
                for sample in data:
                    if dataset.load_from_cache:
                        losses.append(model({}, inputs=sample))
                    else:
                        losses.append(model(sample))
                loss = torch.stack(losses).mean()
                if global_step % 100 == 0:
                    print(f"[Training] epoch={epoch_id}, step={global_step}, loss={loss.item()}")
                accelerator.backward(loss)
                optimizer.step()
                global_step += 1
                model_logger.on_step_end(accelerator, model, save_steps, loss=loss)
                scheduler.step()

                if (
                    val_dataset is not None
                    and args is not None
                    and args.validation_steps > 0
                    and global_step % args.validation_steps == 0
                ):
                    run_stvsr_validation(accelerator, model, val_dataset, global_step, args)

                # if loss.item() < best_loss:
                #     best_loss = loss.item()
                #     if accelerator.is_main_process:
                #         model_logger.save_model(accelerator, model, "best_loss.safetensors")

        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    set_seed(args.seed)                 # 會處理 random / numpy / torch (含分散式)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # 需要更強可重現（會變慢）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    dataset = build_stvsr_dataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        data_file_keys=args.data_file_keys,
        args=args,
        repeat=args.dataset_repeat,
    )
    val_dataset = None
    if validation_is_enabled(args):
        val_dataset = build_stvsr_dataset(
            base_path=args.val_dataset_base_path,
            metadata_path=args.val_dataset_metadata_path,
            data_file_keys=args.val_data_file_keys,
            args=args,
            repeat=1,
        )
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        rope_mode=args.rope_mode,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        # "sft": launch_training_task,
        "sft": launch_training_task_with_validation,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args,val_dataset=val_dataset)
