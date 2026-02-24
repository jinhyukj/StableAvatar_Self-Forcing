"""Precompute CLIP features for StableAvatar training data.

For each training sample, computes CLIP image features from the reference frame
and saves as clip_fea.pt [257, 1280]. This is a one-time preprocessing step.

Requires a CLIP model (e.g., openai/clip-vit-large-patch14-336).

Example:
    PYTHONPATH=$(pwd) python scripts/precompute_stableavatar_clip.py \
        --data_dir /home/work/stableavatar_data/v2v_training_data \
        --clip_model openai/clip-vit-large-patch14-336 \
        --vae_model Wan-AI/Wan2.1-T2V-1.3B-Diffusers
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import fastgen.utils.logging_utils as logger


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CLIP model
    logger.info(f"Loading CLIP model from {args.clip_model}")
    from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

    clip_model = CLIPVisionModelWithProjection.from_pretrained(
        args.clip_model, torch_dtype=torch.float16,
    ).to(device).eval()
    clip_processor = CLIPImageProcessor.from_pretrained(args.clip_model)

    # Optionally load VAE for decoding latents to pixels
    vae = None
    if args.vae_model:
        logger.info(f"Loading VAE from {args.vae_model} for decoding ref latents")
        from diffusers import AutoencoderKLWan
        vae = AutoencoderKLWan.from_pretrained(
            args.vae_model, subfolder="vae", torch_dtype=torch.float32,
        ).to(device).eval()

    # Discover sample directories
    data_dir = Path(args.data_dir)
    if args.manifest_file:
        with open(args.manifest_file) as f:
            sample_names = [line.strip() for line in f if line.strip()]
        sample_dirs = [data_dir / name for name in sample_names if (data_dir / name).exists()]
    else:
        sample_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir())

    sample_dirs = [d for d in sample_dirs if (d / "vae_latents.pt").exists()]
    logger.info(f"Found {len(sample_dirs)} samples")

    num_processed = 0
    num_skipped = 0
    for sample_dir in tqdm(sample_dirs, desc="Computing CLIP features"):
        clip_out_path = sample_dir / "clip_fea.pt"

        # Skip if already computed
        if clip_out_path.exists() and not args.overwrite:
            num_skipped += 1
            continue

        # Load reference latent
        ref_path = sample_dir / "ref_latents.pt"
        if not ref_path.exists():
            # Fall back to first frame of vae_latents
            vae_latents = torch.load(sample_dir / "vae_latents.pt", map_location="cpu", weights_only=True)
            ref_latent = vae_latents[:, :1]  # [C, 1, H, W]
        else:
            ref_latent = torch.load(ref_path, map_location="cpu", weights_only=True)
            if ref_latent.ndim == 4 and ref_latent.shape[1] > 1:
                ref_latent = ref_latent[:, :1]

        # Decode latent to pixel space if VAE is available
        if vae is not None:
            with torch.no_grad():
                ref_latent_input = ref_latent.unsqueeze(0).to(device=device, dtype=torch.float32)
                # VAE decode: [B, C, T, H, W] -> [B, 3, T, H, W]
                ref_pixels = vae.decode(ref_latent_input).sample
                # Take first frame: [B, 3, H, W]
                ref_image = ref_pixels[:, :, 0]
                # Normalize to [0, 1]
                ref_image = (ref_image + 1.0) / 2.0
                ref_image = ref_image.clamp(0, 1)
        else:
            # If no VAE, try to use ref latents directly (won't work well, but avoids crash)
            logger.warning(f"No VAE available, using raw latent for CLIP (results may be poor)")
            ref_image = ref_latent[:3, 0].unsqueeze(0).to(device=device, dtype=torch.float16)
            ref_image = (ref_image - ref_image.min()) / (ref_image.max() - ref_image.min() + 1e-8)

        # Process through CLIP
        with torch.no_grad():
            # Resize to CLIP input size
            ref_image_resized = F.interpolate(
                ref_image.float(), size=(336, 336), mode="bilinear", align_corners=False,
            )
            # Normalize with CLIP stats
            mean = torch.tensor(clip_processor.image_mean, device=device).view(1, 3, 1, 1)
            std = torch.tensor(clip_processor.image_std, device=device).view(1, 3, 1, 1)
            ref_image_normalized = (ref_image_resized - mean) / std

            # Get CLIP features (last hidden state includes CLS + patch tokens)
            clip_output = clip_model(
                pixel_values=ref_image_normalized.to(torch.float16),
                output_hidden_states=True,
            )
            # Use last hidden state: [B, 257, 1280] for ViT-L/14@336
            clip_features = clip_output.hidden_states[-1].squeeze(0)  # [257, 1280]

        # Save
        torch.save(clip_features.cpu(), clip_out_path)
        num_processed += 1

    logger.success(f"Done: {num_processed} computed, {num_skipped} skipped (already exist)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute CLIP features for StableAvatar data")
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory with sample subdirs")
    parser.add_argument("--manifest_file", type=str, default=None, help="Manifest file listing sample dirs")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-large-patch14-336", help="CLIP model ID")
    parser.add_argument("--vae_model", type=str, default=None, help="VAE model ID for decoding latents to pixels")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing clip_fea.pt files")

    args = parser.parse_args()
    main(args)
