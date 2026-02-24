"""Generate ODE pair data for StableAvatar CausalKD training (Stage 1).

Runs the bidirectional StableAvatar teacher's ODE solver (x_T → x_0) and
captures intermediate latent states at the target t_list timesteps.

Input: precomputed StableAvatar training data directories containing:
    vae_latents.pt, audio_emb.pt, text_emb.pt, ref_latents.pt, [clip_fea.pt]

Output: adds to each sample directory:
    latent.pth   # clean teacher output x_0 [C, T, H, W]
    path.pth     # ODE trajectory intermediates [num_steps, C, T, H, W]

Example:
    PYTHONPATH=$(pwd) python scripts/generate_stableavatar_ode_pairs.py \
        --data_dir /home/work/stableavatar_data/v2v_training_data \
        --checkpoint /home/work/.local/StableAvatar/checkpoints/StableAvatar-1.3B/transformer3d-square.pt \
        --output_dir /home/work/stableavatar_data/ode_pairs \
        --num_steps 40 --guidance_scale 5.0
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

import fastgen.utils.basic_utils as basic_utils
import fastgen.utils.logging_utils as logger
from fastgen.networks.StableAvatar.network import StableAvatar
from fastgen.networks.noise_schedule import NoiseScheduler


def build_teacher(checkpoint_path: str, device: torch.device, dtype: torch.dtype) -> StableAvatar:
    """Build and load the bidirectional StableAvatar teacher."""
    teacher = StableAvatar(
        checkpoint_path=checkpoint_path,
        dim=2048,
        num_layers=32,
        num_heads=16,
        net_pred_type="flow",
        schedule_type="rf",
        video_sample_n_frames=81,
        load_pretrained=True,
    )
    teacher = teacher.to(device=device, dtype=dtype)
    teacher.eval()
    return teacher


def sample_with_trajectory(
    teacher: StableAvatar,
    noise: torch.Tensor,
    condition: dict,
    neg_condition: dict,
    target_timesteps: list[float],
    guidance_scale: float = 5.0,
    num_steps: int = 40,
    shift: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run teacher ODE solver and capture intermediate states at target timesteps.

    Args:
        teacher: Bidirectional StableAvatar teacher.
        noise: Initial noise [B, C, T, H, W].
        condition: Dict with text_embeds, first_frame_cond, vocal_embeddings, clip_fea.
        neg_condition: Dict with negative conditions.
        target_timesteps: Timesteps to capture (descending order, > 0).
        guidance_scale: CFG scale.
        num_steps: Number of ODE steps.
        shift: Flow shift for scheduler.

    Returns:
        (x_0, path): x_0 is [B, C, T, H, W], path is [num_targets, C, T, H, W].
    """
    from diffusers import UniPCMultistepScheduler

    first_frame_cond = condition.get("first_frame_cond")

    scheduler = UniPCMultistepScheduler()
    scheduler.config.flow_shift = shift
    scheduler.set_timesteps(num_inference_steps=num_steps, device=noise.device)
    timesteps = scheduler.timesteps

    t_init = timesteps[0] / scheduler.config.num_train_timesteps
    latents = teacher.noise_scheduler.latents(noise=noise, t_init=t_init)

    target_set = sorted(target_timesteps, reverse=True)
    captured = {}

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for idx, timestep in enumerate(timesteps):
            t = (timestep / scheduler.config.num_train_timesteps).expand(latents.shape[0])
            t = teacher.noise_scheduler.safe_clamp(
                t, min=teacher.noise_scheduler.min_t, max=teacher.noise_scheduler.max_t,
            ).to(latents.dtype)
            t_val = t[0].item()

            # Capture intermediates
            for target_t in target_set:
                if target_t not in captured and t_val <= target_t:
                    captured[target_t] = latents.clone()

            # Forward pass with CFG
            flow_pred = teacher(latents, t, condition=condition)

            if guidance_scale is not None and guidance_scale > 1.0:
                flow_uncond = teacher(latents, t, condition=neg_condition)
                flow_pred = flow_uncond + guidance_scale * (flow_pred - flow_uncond)

            # Keep first frame clean
            if first_frame_cond is not None:
                flow_pred = flow_pred.clone()
                flow_pred[:, :, 0] = 0.0

            latents = scheduler.step(flow_pred, timestep, latents, return_dict=False)[0]

            if first_frame_cond is not None:
                latents = latents.clone()
                latents[:, :, 0] = first_frame_cond[:, :, 0]

    # Capture any remaining targets
    for target_t in target_set:
        if target_t not in captured:
            logger.warning(f"Target t={target_t} was not captured, using final latents")
            captured[target_t] = latents.clone()

    x_0 = latents
    path_list = [captured[t].squeeze(0) for t in target_set]
    path = torch.stack(path_list, dim=0)

    return x_0, path


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    # Build teacher
    logger.info(f"Loading teacher from {args.checkpoint}")
    teacher = build_teacher(args.checkpoint, device, dtype)

    # Discover sample directories
    data_dir = Path(args.data_dir)
    if args.manifest_file:
        manifest = Path(args.manifest_file)
        with open(manifest) as f:
            sample_names = [line.strip() for line in f if line.strip()]
        sample_dirs = [data_dir / name for name in sample_names]
        sample_dirs = [d for d in sample_dirs if d.exists()]
    else:
        sample_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir())

    # Filter to valid samples
    sample_dirs = [d for d in sample_dirs if (d / "vae_latents.pt").exists()]
    logger.info(f"Found {len(sample_dirs)} valid samples in {data_dir}")

    # Load negative condition if provided
    neg_text_emb = None
    if args.neg_condition_path and Path(args.neg_condition_path).exists():
        neg_text_emb = torch.load(args.neg_condition_path, map_location="cpu", weights_only=True)

    # Target timesteps (exclude final 0.0)
    t_list = [0.999, 0.937, 0.833, 0.624, 0.0]
    path_timesteps = [t for t in t_list if t > 0]
    logger.info(f"Path timesteps: {path_timesteps}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_generated = 0
    for idx, sample_dir in enumerate(sample_dirs):
        out_dir = output_dir / sample_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Skip if already generated
        if (out_dir / "latent.pth").exists() and (out_dir / "path.pth").exists():
            logger.info(f"[{idx}] Already exists, skipping: {sample_dir.name}")
            num_generated += 1
            continue

        logger.info(f"[{idx}/{len(sample_dirs)}] Processing: {sample_dir.name}")

        # Load precomputed data
        vae_latents = torch.load(sample_dir / "vae_latents.pt", map_location="cpu", weights_only=True)
        audio_emb = torch.load(sample_dir / "audio_emb.pt", map_location="cpu", weights_only=True)
        text_emb = torch.load(sample_dir / "text_emb.pt", map_location="cpu", weights_only=True)

        ref_path = sample_dir / "ref_latents.pt"
        if ref_path.exists():
            ref_latents = torch.load(ref_path, map_location="cpu", weights_only=True)
            if ref_latents.ndim == 4 and ref_latents.shape[1] > 1:
                ref_latents = ref_latents[:, :1]
        else:
            ref_latents = vae_latents[:, :1]

        clip_fea = None
        clip_path = sample_dir / "clip_fea.pt"
        if clip_path.exists():
            clip_fea = torch.load(clip_path, map_location="cpu", weights_only=True)

        # Build condition dicts - add batch dim
        ctx = {"device": device, "dtype": dtype}
        condition = {
            "text_embeds": text_emb.unsqueeze(0).to(**ctx),
            "first_frame_cond": ref_latents.unsqueeze(0).to(**ctx),
            "vocal_embeddings": audio_emb.unsqueeze(0).to(**ctx),
            "clip_fea": clip_fea.unsqueeze(0).to(**ctx) if clip_fea is not None else None,
        }

        if neg_text_emb is not None:
            neg_text = neg_text_emb.unsqueeze(0).to(**ctx) if neg_text_emb.ndim == 2 else neg_text_emb.to(**ctx)
        else:
            neg_text = torch.zeros_like(condition["text_embeds"])

        neg_condition = {
            "text_embeds": neg_text,
            "first_frame_cond": condition["first_frame_cond"],
            "vocal_embeddings": condition["vocal_embeddings"],
            "clip_fea": condition["clip_fea"],
        }

        # Generate noise matching latent shape
        noise = torch.randn_like(vae_latents.unsqueeze(0).to(**ctx))

        # Run teacher ODE with trajectory capture
        x_0, path = sample_with_trajectory(
            teacher=teacher,
            noise=noise,
            condition=condition,
            neg_condition=neg_condition,
            target_timesteps=path_timesteps,
            guidance_scale=args.guidance_scale,
            num_steps=args.num_steps,
            shift=args.shift,
        )

        # Save ODE pair data
        x_0_save = x_0.squeeze(0).cpu()
        path_save = path.cpu()
        torch.save(x_0_save, out_dir / "latent.pth")
        torch.save(path_save, out_dir / "path.pth")

        # Copy condition files to output dir (for the data loader)
        for fname in ["vae_latents.pt", "audio_emb.pt", "text_emb.pt", "ref_latents.pt", "clip_fea.pt", "prompt.txt"]:
            src = sample_dir / fname
            if src.exists():
                shutil.copy2(src, out_dir / fname)

        num_generated += 1
        logger.info(f"[{idx}] Saved: latent={x_0_save.shape}, path={path_save.shape}")

        if args.num_samples > 0 and num_generated >= args.num_samples:
            logger.info(f"Reached {args.num_samples} samples, stopping")
            break

    logger.success(f"Generated {num_generated} ODE pairs in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ODE pairs for StableAvatar CausalKD")
    parser.add_argument("--data_dir", type=str, required=True, help="Input data directory with sample subdirs")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to StableAvatar .pt checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for ODE pairs")
    parser.add_argument("--manifest_file", type=str, default=None, help="Manifest file listing sample dirs")
    parser.add_argument("--neg_condition_path", type=str, default=None, help="Negative condition .pt file")
    parser.add_argument("--num_steps", type=int, default=40, help="Number of ODE steps")
    parser.add_argument("--guidance_scale", type=float, default=5.0, help="CFG scale")
    parser.add_argument("--shift", type=float, default=3.0, help="Flow shift for scheduler")
    parser.add_argument("--num_samples", type=int, default=0, help="Max samples (0=all)")

    args = parser.parse_args()
    main(args)
