# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SyncNet evaluation callback for StableAvatar training.

Generates videos from the student model during validation, runs SyncNet
evaluation via subprocess in the latentsync conda env, and logs Sync-C /
Sync-D scores + generated video samples to wandb.
"""

from __future__ import annotations

import gc
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import torch
import wandb

from fastgen.callbacks.callback import Callback
from fastgen.utils import basic_utils
from fastgen.utils.distributed import get_rank, synchronize
import fastgen.utils.logging_utils as logger

if TYPE_CHECKING:
    from fastgen.methods import FastGenModel


class SyncNetEvalCallback(Callback):
    """Evaluate lip-sync quality (Sync-C, Sync-D) during validation.

    For each validation set, the callback:
    1. Loads precomputed conditions from .pt files
    2. Runs the causal student model to generate latent video
    3. Decodes latents to pixels via the Wan VAE
    4. Saves video + audio as MP4
    5. Runs SyncNet evaluation via subprocess (latentsync conda env)
    6. Logs Sync-C / Sync-D scores and generated videos to wandb
    """

    def __init__(
        self,
        val_sets: Dict[str, str] | None = None,
        max_samples_per_set: int = 8,
        latentsync_dir: str = "/home/work/.local/latentsync-metrics",
        conda_env: str = "latentsync",
        syncnet_model: str = "checkpoints/auxiliary/syncnet_v2.model",
        vae_model_path: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        video_fps: int = 16,
        guidance_scale: float = 5.0,
        sample_steps: int = 50,
    ):
        super().__init__()
        self.val_sets = val_sets or {}
        self.max_samples_per_set = max_samples_per_set
        self.latentsync_dir = latentsync_dir
        self.conda_env = conda_env
        self.syncnet_model = syncnet_model
        self.vae_model_path = vae_model_path
        self.video_fps = video_fps
        self.guidance_scale = guidance_scale
        self.sample_steps = sample_steps
        self._vae = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _discover_samples(self, val_dir: str) -> List[Path]:
        """Find sample directories that have both audio.wav and vae_latents.pt."""
        val_path = Path(val_dir)
        samples = []
        for d in sorted(val_path.iterdir()):
            if d.is_dir() and (d / "audio.wav").exists() and (d / "vae_latents.pt").exists():
                samples.append(d)
        return samples[: self.max_samples_per_set]

    def _load_condition(self, sample_dir: Path, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        """Load precomputed condition tensors from a sample directory."""

        def _load(name: str) -> torch.Tensor:
            return torch.load(sample_dir / name, map_location="cpu", weights_only=True).unsqueeze(0).to(device, dtype)

        cond: Dict[str, torch.Tensor] = {}
        cond["text_embeds"] = _load("text_emb.pt")
        cond["vocal_embeddings"] = _load("audio_emb.pt")

        ref = torch.load(sample_dir / "ref_latents.pt", map_location="cpu", weights_only=True)
        if ref.ndim == 4 and ref.shape[1] > 1:
            ref = ref[:, :1]
        cond["first_frame_cond"] = ref.unsqueeze(0).to(device, dtype)

        clip_path = sample_dir / "clip_fea.pt"
        if clip_path.exists():
            cond["clip_fea"] = _load("clip_fea.pt")
        else:
            cond["clip_fea"] = None

        return cond

    def _get_neg_condition(self, cond: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Build a negative condition dict (zeros for text, keep audio/ref)."""
        return {
            "text_embeds": torch.zeros_like(cond["text_embeds"]),
            "first_frame_cond": cond["first_frame_cond"],
            "vocal_embeddings": cond["vocal_embeddings"],
            "clip_fea": cond["clip_fea"],
        }

    def _get_vae(self, device: torch.device):
        """Lazy-load and cache the Wan VAE decoder."""
        if self._vae is None:
            from fastgen.networks.Wan.network import WanVideoEncoder

            logger.info("SyncNetEvalCallback: loading Wan VAE for decoding...")
            self._vae = WanVideoEncoder(model_id_or_local_path=self.vae_model_path)
        self._vae.to(device=device, dtype=torch.float32)
        return self._vae

    def _free_vae(self):
        """Move VAE to CPU and free GPU memory."""
        if self._vae is not None:
            self._vae.to(device="cpu")
        gc.collect()
        torch.cuda.empty_cache()

    def _run_syncnet(self, videos_dir: str) -> tuple[Optional[float], Optional[float]]:
        """Run SyncNet evaluation subprocess and parse Sync-D / Sync-C."""
        eval_script = os.path.join(self.latentsync_dir, "eval", "eval_sync.py")
        model_path = os.path.join(self.latentsync_dir, self.syncnet_model)

        cmd = [
            "conda", "run", "-n", self.conda_env, "--no-capture-output",
            "python", eval_script,
            "--videos_dir", videos_dir,
            "--initial_model", model_path,
        ]
        logger.info(f"SyncNetEvalCallback: running {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600,
                cwd=self.latentsync_dir,
            )
            stdout = result.stdout
            stderr = result.stderr
            if result.returncode != 0:
                logger.warning(f"SyncNet subprocess failed (rc={result.returncode}):\nstdout: {stdout}\nstderr: {stderr}")
                return None, None

            logger.info(f"SyncNet stdout:\n{stdout}")

            sync_d = None
            sync_c = None
            m = re.search(r"Mean SyncNet Min Distance \(Sync-D\):\s*([\d.]+)", stdout)
            if m:
                sync_d = float(m.group(1))
            m = re.search(r"Mean SyncNet Confidence \(Sync-C\):\s*([\d.]+)", stdout)
            if m:
                sync_c = float(m.group(1))
            return sync_d, sync_c
        except subprocess.TimeoutExpired:
            logger.warning("SyncNet subprocess timed out (600s)")
            return None, None
        except Exception as e:
            logger.warning(f"SyncNet subprocess error: {e}")
            return None, None

    # ------------------------------------------------------------------
    # Callback hook
    # ------------------------------------------------------------------

    def on_validation_end(self, model: FastGenModel, iteration: int = 0, idx: int = 0) -> None:
        if get_rank() != 0:
            return
        if not self.val_sets:
            return

        for set_name, val_dir in self.val_sets.items():
            logger.info(f"SyncNetEvalCallback: evaluating set '{set_name}' at iteration {iteration}")
            try:
                self._evaluate_set(model, set_name, val_dir, iteration)
            except Exception as e:
                logger.warning(f"SyncNetEvalCallback: error evaluating '{set_name}': {e}")

        synchronize()

    def _evaluate_set(self, model: FastGenModel, set_name: str, val_dir: str, iteration: int) -> None:
        samples = self._discover_samples(val_dir)
        if not samples:
            logger.warning(f"SyncNetEvalCallback: no valid samples in {val_dir}")
            return

        device = model.device
        dtype = model.precision

        tmp_dir = tempfile.mkdtemp(prefix=f"syncnet_{set_name}_")
        try:
            wandb_videos = []
            wandb_gt_videos = []

            for i, sample_dir in enumerate(samples):
                logger.info(f"  [{i+1}/{len(samples)}] generating from {sample_dir.name}")

                # Load conditions
                cond = self._load_condition(sample_dir, device, dtype)
                neg_cond = self._get_neg_condition(cond)

                # Determine latent shape from vae_latents
                vae_latents = torch.load(
                    sample_dir / "vae_latents.pt", map_location="cpu", weights_only=True,
                )
                latent_shape = vae_latents.shape  # [C, T, H, W]

                # Generate noise and run student inference
                noise = torch.randn(1, *latent_shape, device=device, dtype=dtype)

                with basic_utils.inference_mode(
                    model.net, precision_amp=model.precision_amp_infer,
                ):
                    gen_latents = model.net.sample(
                        noise,
                        condition=cond,
                        neg_condition=neg_cond,
                        guidance_scale=self.guidance_scale,
                        sample_steps=self.sample_steps,
                    )  # [1, C, T, H, W]

                # Decode latents -> pixel video
                vae = self._get_vae(device)
                with basic_utils.inference_mode(precision_amp=None, device_type=device.type):
                    pixel_video = vae.decode(gen_latents)  # [1, C, T, H, W] in [-1, 1]

                # Save video-only MP4
                video_only_path = os.path.join(tmp_dir, f"{sample_dir.name}_video.mp4")
                basic_utils.save_video(
                    pixel_video[0],  # [C, T, H, W]
                    video_only_path,
                    save_as_gif=False,
                    fps=self.video_fps,
                )

                # Combine video + audio via ffmpeg
                audio_path = str(sample_dir / "audio.wav")
                final_path = os.path.join(tmp_dir, f"{sample_dir.name}.mp4")
                ffmpeg_cmd = [
                    "ffmpeg", "-y",
                    "-i", video_only_path,
                    "-i", audio_path,
                    "-c:v", "copy", "-c:a", "aac",
                    "-shortest",
                    "-loglevel", "quiet",
                    final_path,
                ]
                subprocess.run(ffmpeg_cmd, check=True, timeout=60)

                # Collect wandb video
                if wandb.run and os.path.exists(final_path):
                    wandb_videos.append(
                        wandb.Video(final_path, fps=self.video_fps, caption=sample_dir.name)
                    )

                # Ground truth video
                gt_path = sample_dir / "sub_clip.mp4"
                if wandb.run and gt_path.exists():
                    wandb_gt_videos.append(
                        wandb.Video(str(gt_path), fps=self.video_fps, caption=sample_dir.name)
                    )

                # Free intermediate tensors
                del gen_latents, pixel_video, noise, cond, neg_cond, vae_latents
                torch.cuda.empty_cache()

            self._free_vae()

            # Run SyncNet evaluation
            sync_d, sync_c = self._run_syncnet(tmp_dir)

            # Log to wandb
            if wandb.run:
                log_dict = {}
                if sync_d is not None:
                    log_dict[f"val/sync_d_{set_name}"] = sync_d
                    logger.info(f"  Sync-D ({set_name}): {sync_d:.2f}")
                if sync_c is not None:
                    log_dict[f"val/sync_c_{set_name}"] = sync_c
                    logger.info(f"  Sync-C ({set_name}): {sync_c:.2f}")

                # Log generated videos
                if wandb_videos:
                    log_dict[f"val_media/generation_{set_name}"] = wandb_videos
                if wandb_gt_videos:
                    log_dict[f"val_media/ground_truth_{set_name}"] = wandb_gt_videos

                if log_dict:
                    wandb.log(log_dict, step=iteration)

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
