# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""StableAvatar directory-based data loader for precomputed training data.

Each sample is a directory containing precomputed .pt files:
    sample_dir/
        vae_latents.pt  -> [C, T, H, W] VAE-encoded video latents
        audio_emb.pt    -> [T_audio, 768] audio embeddings
        text_emb.pt     -> [L, 4096] T5 text embeddings
        ref_latents.pt  -> [C, T, H, W] reference frame latents
        clip_fea.pt     -> [257, 1280] CLIP features (optional)
        prompt.txt      -> text prompt (metadata only)

Uses a manifest file to list sample directories.
"""

from typing import Dict, Any, Optional
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

import fastgen.utils.logging_utils as logger


class StableAvatarDirectoryLoader:
    """Dataloader for StableAvatar precomputed training data.

    Args:
        data_dir: Root directory containing sample subdirectories.
        manifest_file: Path to manifest file listing sample directory names (one per line).
            If None, all subdirectories in data_dir are used.
        neg_condition_path: Path to a shared negative condition .pt file.
        batch_size: Batch size.
        shuffle: Whether to shuffle samples.
        num_workers: Number of dataloader workers.
        repeat: Whether to repeat indefinitely (for training).
        load_clip_fea: Whether to load clip_fea.pt if present.
        load_ode_pairs: Whether to load ODE pair data (path.pth, latent.pth).
    """

    def __init__(
        self,
        data_dir: str,
        manifest_file: Optional[str] = None,
        neg_condition_path: Optional[str] = None,
        batch_size: int = 1,
        shuffle: bool = True,
        num_workers: int = 4,
        repeat: bool = True,
        load_clip_fea: bool = True,
        load_ode_pairs: bool = False,
    ):
        self.batch_size = batch_size
        data_path = Path(data_dir)

        # Discover sample directories
        if manifest_file is not None:
            manifest = Path(manifest_file)
            with open(manifest) as f:
                sample_names = [line.strip() for line in f if line.strip()]
            sample_dirs = [data_path / name for name in sample_names]
            sample_dirs = [d for d in sample_dirs if d.exists()]
        else:
            sample_dirs = sorted(d for d in data_path.iterdir() if d.is_dir())

        # Filter to valid samples (must have vae_latents.pt)
        sample_dirs = [d for d in sample_dirs if (d / "vae_latents.pt").exists()]

        if len(sample_dirs) == 0:
            raise RuntimeError(f"No valid StableAvatar samples found in {data_dir}")
        logger.info(f"StableAvatarDirectoryLoader: found {len(sample_dirs)} samples in {data_dir}")

        # Load shared negative condition
        neg_condition = None
        if neg_condition_path is not None and Path(neg_condition_path).exists():
            neg_condition = torch.load(neg_condition_path, map_location="cpu", weights_only=True)
            logger.info(f"Loaded negative condition from {neg_condition_path}")

        dataset = _StableAvatarDataset(
            sample_dirs, neg_condition=neg_condition,
            load_clip_fea=load_clip_fea, load_ode_pairs=load_ode_pairs,
        )

        if repeat:
            self.loader = _InfiniteLoader(dataset, batch_size, shuffle, num_workers)
        else:
            self.loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle,
                num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                drop_last=True,
            )

    def __iter__(self):
        return iter(self.loader)


class _StableAvatarDataset(Dataset):
    def __init__(self, dirs, neg_condition=None, load_clip_fea=True, load_ode_pairs=False):
        self.dirs = dirs
        self.neg_condition = neg_condition
        self.load_clip_fea = load_clip_fea
        self.load_ode_pairs = load_ode_pairs

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx):
        d = self.dirs[idx]
        out = {}

        # Core data
        out["real"] = torch.load(d / "vae_latents.pt", map_location="cpu", weights_only=True)
        out["vocal_embeddings"] = torch.load(d / "audio_emb.pt", map_location="cpu", weights_only=True)
        out["condition"] = torch.load(d / "text_emb.pt", map_location="cpu", weights_only=True)

        # Reference latents (first frame conditioning)
        ref_path = d / "ref_latents.pt"
        if ref_path.exists():
            ref = torch.load(ref_path, map_location="cpu", weights_only=True)
            # Ensure it's just the first frame: [C, 1, H, W]
            if ref.ndim == 4 and ref.shape[1] > 1:
                ref = ref[:, :1]
            out["first_frame_cond"] = ref
        else:
            # Use first frame of vae_latents as fallback
            out["first_frame_cond"] = out["real"][:, :1]

        # CLIP features (optional)
        if self.load_clip_fea:
            clip_path = d / "clip_fea.pt"
            if clip_path.exists():
                out["clip_fea"] = torch.load(clip_path, map_location="cpu", weights_only=True)

        # Negative condition
        if self.neg_condition is not None:
            out["neg_condition"] = self.neg_condition.clone()
        else:
            # Use zeros as default negative condition
            out["neg_condition"] = torch.zeros_like(out["condition"])

        # ODE pair data (for Stage 2 KD)
        if self.load_ode_pairs:
            path_file = d / "path.pth"
            latent_file = d / "latent.pth"
            if path_file.exists():
                out["path"] = torch.load(path_file, map_location="cpu", weights_only=True)
            if latent_file.exists():
                out["real"] = torch.load(latent_file, map_location="cpu", weights_only=True)

        return out


class _InfiniteLoader:
    """Infinite dataloader that reshuffles each epoch."""

    def __init__(self, ds, batch_size, shuffle, num_workers):
        self.ds = ds
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def __iter__(self):
        while True:
            loader = DataLoader(
                self.ds,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
                pin_memory=torch.cuda.is_available(),
                drop_last=True,
            )
            yield from loader
