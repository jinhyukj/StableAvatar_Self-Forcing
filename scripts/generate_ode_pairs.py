
"""Generate ODE pair data for CausalKD training (Stage 1 of the 3-stage pipeline).

Runs the teacher's ODE solver (x_T → x_0) and captures intermediate latent
states at the target t_list timesteps. This produces the actual ODE trajectory
rather than synthetic forward-process reconstructions.

Supports:
- T2W (text-to-world): text prompt → teacher sample → ODE path
- V2W (video/image-to-world): text prompt + source image → teacher sample → ODE path

Output structure (local directory per sample):
    output_dir/
        000000/
            latent.pth       # clean teacher output x_0 [C, T, H, W]
            path.pth         # ODE trajectory intermediates [num_steps, C, T, H, W]
            prompt.txt       # raw text prompt
            cond_latent.pth  # (V2W only) conditioning latents
            video.mp4        # decoded video for visual verification
        000001/
            ...

Examples:

    # T2W mode
    PYTHONPATH=$(pwd) torchrun --nproc_per_node=1 --standalone \\
        scripts/generate_ode_pairs.py \\
        --config fastgen/configs/experiments/CosmosPredict2/config_dmd2.py \\
        --prompt_file scripts/inference/prompts/validation_aug_qwen_2_5_14b_seed42.txt \\
        --output_dir ODE_PAIRS/cosmos_t2w \\
        --num_steps 35 --guidance_scale 3.0 \\
        - trainer.ddp=True model.guidance_scale=3.0

    # V2W mode
    PYTHONPATH=$(pwd) torchrun --nproc_per_node=1 --standalone \\
        scripts/generate_ode_pairs.py \\
        --config fastgen/configs/experiments/CosmosPredict2/config_dmd2.py \\
        --prompt_file scripts/inference/prompts/validation_aug_qwen_2_5_14b_seed42.txt \\
        --input_image_file scripts/inference/prompts/source_image_paths.txt \\
        --num_conditioning_frames 1 \\
        --output_dir ODE_PAIRS/cosmos_v2w \\
        --num_steps 35 --guidance_scale 3.0 \\
        - trainer.ddp=True model.guidance_scale=3.0 \\
          model.net.is_video2world=True
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.distributed as dist

import fastgen.utils.basic_utils as basic_utils
import fastgen.utils.logging_utils as logger
from scripts.inference.inference_utils import (
    add_common_args,
    cleanup_unused_modules,
    expand_path,
    init_checkpointer,
    init_model,
    load_checkpoint,
    load_prompts,
    setup_inference_modules,
)
from scripts.inference.video_model_inference import (
    load_conditioning_image,
    prepare_cosmos_v2w_condition,
)


def sample_with_trajectory(
    teacher,
    noise: torch.Tensor,
    target_timesteps: list[float],
    precision_amp: torch.dtype,
    **sample_kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run teacher ODE solver and capture intermediate states at target timesteps.

    Instead of generating x_0 and then reconstructing noisy states via forward process,
    this captures the actual ODE trajectory — the real intermediate latent states that
    the solver passes through on its way from x_T to x_0.

    Args:
        teacher: Teacher network with sample() method.
        noise: Initial noise tensor [B, C, T, H, W].
        target_timesteps: Timesteps at which to capture intermediates (e.g., [0.999, 0.937, 0.833, 0.624]).
            Must be in descending order and > 0.
        precision_amp: AMP precision for inference.
        **sample_kwargs: Additional kwargs passed to teacher's ODE loop
            (condition, neg_condition, num_steps, fps, etc.)

    Returns:
        Tuple of (x_0, path) where:
            x_0: Clean output [B, C, T, H, W]
            path: Intermediate states [len(target_timesteps), C, T, H, W] (batch dim squeezed)
    """
    from diffusers import UniPCMultistepScheduler

    assert teacher.schedule_type == "rf", f"{teacher.schedule_type} is not supported"

    was_training = teacher.training
    teacher.eval()

    # Extract kwargs
    condition = sample_kwargs.get("condition")
    neg_condition = sample_kwargs.get("neg_condition")
    guidance_scale = sample_kwargs.get("guidance_scale", 5.0)
    num_steps = sample_kwargs.get("num_steps", 35)
    shift = sample_kwargs.get("shift", 5.0)
    fps = sample_kwargs.get("fps")

    # Set up scheduler (same as teacher.sample())
    if teacher.sample_scheduler is None:
        teacher.sample_scheduler = UniPCMultistepScheduler(
            num_train_timesteps=1000,
            prediction_type="flow_prediction",
            use_flow_sigmas=True,
            flow_shift=shift,
        )
    else:
        teacher.sample_scheduler.config.flow_shift = shift
    teacher.sample_scheduler.set_timesteps(num_inference_steps=num_steps, device=noise.device)
    timesteps = teacher.sample_scheduler.timesteps

    # Initialize latents
    t_init = timesteps[0] / teacher.sample_scheduler.config.num_train_timesteps
    latents = teacher.noise_scheduler.latents(noise=noise, t_init=t_init)

    if fps is None:
        fps = torch.full((latents.shape[0],), teacher.fps, device=latents.device)

    # Handle dict-style condition (same as teacher.sample())
    conditioning_latents = None
    if isinstance(condition, dict):
        if "conditioning_latents" in condition:
            conditioning_latents = condition["conditioning_latents"]
        num_conditioning_frames = 1
        if "condition_mask" in condition:
            mask = condition["condition_mask"]
            if conditioning_latents is not None:
                num_conditioning_frames = int(mask[:, :, :, 0, 0].sum(dim=2).max().item())
        condition = condition["text_embeds"]
    if isinstance(neg_condition, dict):
        neg_condition = neg_condition["text_embeds"]

    # Video2world setup
    video2world_mode = conditioning_latents is not None
    conditioning_latents_full = None
    condition_mask = None
    condition_mask_C = None
    initial_noise = None

    if video2world_mode:
        B, C, T, H, W = latents.shape
        condition_mask = torch.zeros(B, 1, T, H, W, device=latents.device, dtype=latents.dtype)
        condition_mask[:, :, :num_conditioning_frames, :, :] = 1.0
        v2w_condition = {"conditioning_latents": conditioning_latents, "condition_mask": condition_mask}
        conditioning_latents_full, condition_mask, condition_mask_C = teacher._get_conditioning_tensors(
            latents, v2w_condition
        )
        initial_noise = latents.clone()

    # Prepare trajectory capture: find nearest scheduler step for each target timestep
    target_set = sorted(target_timesteps, reverse=True)
    captured = {}

    with basic_utils.inference_mode(teacher, precision_amp=precision_amp):
        for idx, timestep in enumerate(timesteps):
            # Normalized t in [0, 1]
            t = (timestep / teacher.sample_scheduler.config.num_train_timesteps).expand(latents.shape[0])
            t = teacher.noise_scheduler.safe_clamp(
                t, min=teacher.noise_scheduler.min_t, max=teacher.noise_scheduler.max_t
            ).to(latents.dtype)
            t_val = t[0].item()

            # Capture: for each target, save the latents at the closest step BEFORE the solver moves past it
            for target_t in target_set:
                if target_t not in captured and t_val <= target_t:
                    captured[target_t] = latents.clone()

            # Forward pass (same as teacher.sample())
            if video2world_mode:
                v2w_cond = {"conditioning_latents": conditioning_latents, "condition_mask": condition_mask}
                model_input = teacher.preserve_conditioning(latents, v2w_cond)
                B, _, T, _, _ = latents.shape
                t_expanded = t.unsqueeze(1).expand(B, T)
                mask_B_T = condition_mask[:, 0, :, 0, 0]
                t_per_frame = 0.0 * mask_B_T + t_expanded * (1 - mask_B_T)
                cond_with_mask = {"text_embeds": condition, "condition_mask": condition_mask}
                neg_cond_with_mask = {"text_embeds": neg_condition, "condition_mask": condition_mask}
            else:
                model_input = latents
                t_per_frame = t
                cond_with_mask = condition
                neg_cond_with_mask = neg_condition

            velocity_pred = teacher(model_input, t_per_frame, cond_with_mask, fps=fps)

            if guidance_scale > 1.0:
                velocity_uncond = teacher(model_input, t_per_frame, neg_cond_with_mask, fps=fps)
                velocity_pred = velocity_uncond + guidance_scale * (velocity_pred - velocity_uncond)

            if video2world_mode:
                gt_velocity = initial_noise - conditioning_latents_full
                velocity_pred = gt_velocity * condition_mask_C + velocity_pred * (1 - condition_mask_C)

            sample_input = model_input if video2world_mode else latents
            latents = teacher.sample_scheduler.step(velocity_pred, timestep, sample_input, return_dict=False)[0]

            if video2world_mode:
                v2w_cond = {"conditioning_latents": conditioning_latents, "condition_mask": condition_mask}
                latents = teacher.preserve_conditioning(latents, v2w_cond)

    # Capture any remaining targets (should be caught by t approaching 0)
    for target_t in target_set:
        if target_t not in captured:
            logger.warning(f"Target t={target_t} was not captured during ODE solve, using final latents")
            captured[target_t] = latents.clone()

    teacher.train(was_training)

    # Build path tensor in descending t order
    x_0 = latents  # [B, C, T, H, W]
    path_list = [captured[t].squeeze(0) for t in target_set]  # each [C, T, H, W]
    path = torch.stack(path_list, dim=0)  # [num_targets, C, T, H, W]

    return x_0, path


def generate_ode_pairs(args, config):
    """Generate ODE pair data using the teacher model."""

    latent_shape = config.model.input_shape  # [C, T, H, W]
    t_list = config.model.sample_t_cfg.t_list  # e.g., [0.999, 0.937, 0.833, 0.624, 0.0]

    # Load prompts
    pos_prompt_set = load_prompts(args.prompt_file, relative_to="cwd")
    logger.info(f"Loaded {len(pos_prompt_set)} prompts")

    # Load input images for V2W mode
    input_image_paths = None
    if args.input_image_file is not None:
        input_image_file_path = expand_path(args.input_image_file, relative_to="cwd")
        if input_image_file_path.is_file():
            with input_image_file_path.open("r") as f:
                input_image_paths = [line.strip() for line in f.readlines() if line.strip()]
            # Align with prompts
            num_prompts = len(pos_prompt_set)
            num_images = len(input_image_paths)
            if num_images < num_prompts:
                input_image_paths.extend([input_image_paths[-1]] * (num_prompts - num_images))
            elif num_images > num_prompts:
                input_image_paths = input_image_paths[:num_prompts]
            logger.info(f"V2W mode: {len(input_image_paths)} input images")
        else:
            raise FileNotFoundError(f"input_image_file not found: {input_image_file_path}")

    # Load negative prompts
    neg_prompt_file = getattr(args, "neg_prompt_file", None)
    neg_condition = None

    # Set seed
    basic_utils.set_random_seed(config.trainer.seed, by_rank=True)

    # Initialize model
    model = init_model(config)
    checkpointer = init_checkpointer(config)
    load_checkpoint(checkpointer, model, args.ckpt_path, config)

    # Clean up unused modules (discriminator, fake_score)
    cleanup_unused_modules(model, do_teacher_sampling=True)

    # Set up teacher and VAE
    teacher, _, vae = setup_inference_modules(
        model, config, do_teacher_sampling=True, do_student_sampling=False, precision=model.precision
    )
    ctx = {"dtype": model.precision, "device": model.device}

    assert teacher is not None and hasattr(teacher, "sample"), "Teacher must be available for ODE pair generation"

    # Encode negative prompt
    if neg_prompt_file is not None:
        neg_prompts = load_prompts(neg_prompt_file, relative_to="cwd")
        if len(neg_prompts) > 0:
            neg_condition = neg_prompts[:1]
            if hasattr(model.net, "text_encoder"):
                with basic_utils.inference_mode(
                    model.net.text_encoder, precision_amp=model.precision_amp_enc, device_type=model.device.type
                ):
                    neg_condition = basic_utils.to(model.net.text_encoder.encode(neg_condition), **ctx)

    # Timesteps for path construction (exclude the final 0.0)
    path_timesteps = [t for t in t_list if t > 0]
    logger.info(f"Path timesteps: {path_timesteps}")
    logger.info(f"Latent shape: {latent_shape}")

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Distributed: split prompts across ranks
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    prompts_per_rank = list(range(rank, len(pos_prompt_set), world_size))
    logger.info(f"Rank {rank}: processing {len(prompts_per_rank)} samples out of {len(pos_prompt_set)}")

    num_generated = 0
    for idx in prompts_per_rank:
        prompt = pos_prompt_set[idx]
        sample_dir = output_dir / f"{idx:06d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Skip if already generated
        if (sample_dir / "latent.pth").exists() and (sample_dir / "path.pth").exists():
            logger.info(f"[{idx}] Already exists, skipping")
            num_generated += 1
            continue

        logger.info(f"[{idx}/{len(pos_prompt_set)}] Generating: {prompt[:80]}...")

        # Encode text prompt
        condition = [prompt]
        if hasattr(model.net, "text_encoder"):
            with basic_utils.inference_mode(
                model.net.text_encoder, precision_amp=model.precision_amp_enc, device_type=model.device.type
            ):
                condition = basic_utils.to(model.net.text_encoder.encode(condition), **ctx)

        # Prepare V2W conditioning if applicable
        neg_condition_sample = neg_condition
        conditioning_latents = None
        is_v2w = getattr(model.net, "is_video2world", False) and input_image_paths is not None

        if is_v2w:
            vae_spatial_factor = 16 if latent_shape[0] == 48 else 8
            height = latent_shape[2] * vae_spatial_factor
            width = latent_shape[3] * vae_spatial_factor
            num_cond_frames = args.num_conditioning_frames

            conditioning_frames = load_conditioning_image(
                input_image_paths[idx],
                height=height,
                width=width,
                num_latent_frames=num_cond_frames,
            )
            if conditioning_frames is None:
                logger.warning(f"[{idx}] Failed to load conditioning image, skipping")
                continue

            conditioning_frames = conditioning_frames.to(**ctx)
            with basic_utils.inference_mode(
                vae, precision_amp=model.precision_amp_infer, device_type=model.device.type
            ):
                conditioning_latents = vae.encode(conditioning_frames)
            logger.info(f"[{idx}] Encoded conditioning image to {conditioning_latents.shape}")

            condition, neg_condition_sample, _ = prepare_cosmos_v2w_condition(
                conditioning_latents=conditioning_latents,
                condition=condition,
                neg_condition=neg_condition,
                latent_shape=latent_shape,
                num_conditioning_frames=num_cond_frames,
            )

        # Generate noise
        noise = torch.randn([1, *latent_shape], **ctx)

        # Teacher ODE sampling with trajectory capture
        x_0, path = sample_with_trajectory(
            teacher=teacher,
            noise=noise,
            target_timesteps=path_timesteps,
            precision_amp=model.precision_amp_infer,
            condition=condition,
            neg_condition=neg_condition_sample,
            guidance_scale=config.model.guidance_scale,
            num_steps=args.num_steps,
            fps=torch.full((noise.shape[0],), float(args.fps), device=noise.device),
        )  # x_0: [1, C, T, H, W], path: [num_steps, C, T, H, W]

        # Save tensors to local directory
        x_0_save = x_0.squeeze(0).cpu()  # [C, T, H, W]
        path_save = path.cpu()  # [num_steps, C, T, H, W]

        torch.save(x_0_save, sample_dir / "latent.pth")
        torch.save(path_save, sample_dir / "path.pth")
        (sample_dir / "prompt.txt").write_text(prompt)

        # Save conditioning latents for V2W mode
        if conditioning_latents is not None:
            torch.save(conditioning_latents.cpu(), sample_dir / "cond_latent.pth")

        # Decode and save video for visual verification
        if vae is not None:
            try:
                basic_utils.save_media(
                    x_0,
                    str(sample_dir / "video.mp4"),
                    vae=vae,
                    precision_amp=model.precision_amp_infer,
                    fps=args.fps,
                    save_as_gif=False,
                )
                logger.info(f"[{idx}] Saved decoded video to {sample_dir / 'video.mp4'}")
            except Exception as e:
                logger.warning(f"[{idx}] Failed to save decoded video: {e}")

        num_generated += 1
        logger.info(f"[{idx}] Saved ODE pair to {sample_dir} (latent: {x_0_save.shape}, path: {path_save.shape})")

        if args.num_samples > 0 and num_generated >= args.num_samples:
            logger.info(f"Reached requested {args.num_samples} samples, stopping")
            break

    logger.success(f"Rank {rank}: Generated {num_generated} ODE pairs in {output_dir}")


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ODE pair data for CausalKD training")

    # Common inference args
    add_common_args(parser)

    # ODE pair generation specific args
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for ODE pair data",
    )
    parser.add_argument(
        "--prompt_file",
        default="scripts/inference/prompts/validation_aug_qwen_2_5_14b_seed42.txt",
        type=str,
        help="File containing prompts (one per line)",
    )
    parser.add_argument(
        "--neg_prompt_file",
        default="scripts/inference/prompts/negative_prompt_cosmos.txt",
        type=str,
        help="File containing the negative prompt for CFG",
    )
    parser.add_argument(
        "--num_steps",
        default=35,
        type=int,
        help="Number of ODE steps for teacher sampling",
    )
    parser.add_argument(
        "--fps",
        default=24,
        type=int,
        help="Frames per second for video generation and temporal encoding",
    )
    parser.add_argument(
        "--guidance_scale",
        default=3.0,
        type=float,
        help="Classifier-free guidance scale for teacher sampling",
    )
    parser.add_argument(
        "--num_samples",
        default=0,
        type=int,
        help="Max number of samples to generate (0 = all prompts)",
    )
    parser.add_argument(
        "--input_image_file",
        default=None,
        type=str,
        help="File containing source image paths for V2W mode (one per line)",
    )
    parser.add_argument(
        "--num_conditioning_frames",
        default=1,
        type=int,
        help="Number of conditioning frames for V2W mode",
    )

    from fastgen.utils.scripts import parse_args, setup
    from fastgen.utils.distributed import clean_up

    args = parse_args(parser)
    config = setup(args, evaluation=True)

    # Override guidance scale if specified via CLI
    if args.guidance_scale is not None:
        config.model.guidance_scale = args.guidance_scale

    generate_ode_pairs(args, config)

    clean_up()
