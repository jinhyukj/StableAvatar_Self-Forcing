# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bidirectional StableAvatar network wrapper for Self-Forcing framework.

StableAvatar is a Wan2.1-based model with custom audio conditioning via
FantasyTalkingVocalCondition1BModel. It is NOT a diffusers model — uses
a custom WanTransformer3DFantasyModel loaded from a .pt checkpoint.
"""

from typing import Optional, List, Set, Union, Tuple, Dict
import os

import torch
from tqdm.auto import tqdm
from diffusers import UniPCMultistepScheduler

from fastgen.networks.network import FastGenNetwork
from fastgen.networks.noise_schedule import NET_PRED_TYPES
from fastgen.networks.StableAvatar.models.wan_fantasy_transformer3d_1B import (
    WanTransformer3DFantasyModel,
)
import fastgen.utils.logging_utils as logger


class StableAvatar(FastGenNetwork):
    """Bidirectional StableAvatar teacher model for Self-Forcing distillation.

    This wraps the custom WanTransformer3DFantasyModel (not a diffusers model)
    and follows the pattern of WanI2V but with vocal/audio conditioning.
    """

    def __init__(
        self,
        checkpoint_path: str = "",
        dim: int = 2048,
        num_layers: int = 32,
        num_heads: int = 16,
        net_pred_type: str = "flow",
        schedule_type: str = "rf",
        video_sample_n_frames: int = 81,
        load_pretrained: bool = True,
        **model_kwargs,
    ):
        """StableAvatar model constructor.

        Args:
            checkpoint_path: Path to the .pt checkpoint file for the transformer.
            dim: Hidden dimension of the transformer.
            num_layers: Number of transformer blocks.
            num_heads: Number of attention heads.
            net_pred_type: Prediction type. Defaults to "flow".
            schedule_type: Schedule type. Defaults to "rf".
            video_sample_n_frames: Number of video frames (pixel space). Defaults to 81.
            load_pretrained: Whether to load pretrained weights. Defaults to True.
        """
        super().__init__(
            net_pred_type=net_pred_type,
            schedule_type=schedule_type,
            **model_kwargs,
        )

        self.is_i2v = True
        self.is_stableavatar = True
        self.concat_mask = False
        self.expand_timesteps = True
        self.checkpoint_path = checkpoint_path
        self.video_sample_n_frames = video_sample_n_frames

        # Initialize the transformer
        self.transformer = WanTransformer3DFantasyModel(
            model_type='i2v',
            patch_size=(1, 2, 2),
            text_len=512,
            in_dim=16,
            dim=dim,
            ffn_dim=dim * 4,
            freq_dim=256,
            text_dim=4096,
            out_dim=16,
            num_heads=num_heads,
            num_layers=num_layers,
            window_size=(-1, -1),
            qk_norm=True,
            cross_attn_norm=True,
            eps=1e-6,
        )

        # Load weights if requested and path exists
        if load_pretrained and checkpoint_path and os.path.isfile(checkpoint_path):
            self._load_checkpoint(checkpoint_path)

        # UniPC scheduler for teacher sampling
        self.unipc_scheduler = UniPCMultistepScheduler()

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load transformer weights from a .pt checkpoint file."""
        logger.info(f"Loading StableAvatar transformer from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        # Handle 0-dim tensors (FSDP2 compat)
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor) and v.ndim == 0:
                state_dict[k] = v.unsqueeze(0)

        load_info = self.transformer.load_state_dict(state_dict, strict=False)
        logger.success(f"StableAvatar checkpoint loaded. Info: {load_info}")

    def _compute_timestep_inputs(
        self,
        timestep: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute timestep input for the transformer.

        For StableAvatar (Wan 2.2 TI2V style), expand timesteps to [B, T]
        and zero out the first frame if mask indicates it's the conditioning frame.

        Args:
            timestep: [B] or [B, T] tensor of timesteps.
            mask: [B, T, H, W] optional mask for first frame zeroing.
        """
        timestep = self.noise_scheduler.rescale_t(timestep)
        if timestep.ndim == 1:
            timestep = timestep.view(-1, 1)
        if mask is not None:
            p_t = self.transformer.patch_size[0]
            timestep = mask[:, ::p_t, 0, 0] * timestep
        return timestep

    def _replace_first_frame(
        self,
        first_frame_cond: torch.Tensor,
        latents: torch.Tensor,
        return_mask: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Replace the first latent frame with the clean conditioning frame.

        Args:
            first_frame_cond: Clean first frame [B, C, 1, H, W].
            latents: Noisy latents [B, C, T, H, W].
            return_mask: Whether to return the mask.
        """
        bsz, _, num_latent_frames, latent_height, latent_width = latents.shape
        first_frame_mask = torch.ones(
            1, 1, num_latent_frames, latent_height, latent_width,
            dtype=latents.dtype, device=latents.device,
        )
        first_frame_mask[:, :, 0] = 0
        latents = (1 - first_frame_mask) * first_frame_cond + first_frame_mask * latents
        if return_mask:
            return latents, first_frame_mask
        return latents

    def preserve_conditioning(
        self, x: torch.Tensor, condition: Optional[Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Preserve conditioning frames for I2V models during student sampling."""
        if not isinstance(condition, dict) or "first_frame_cond" not in condition:
            return x
        first_frame_cond = condition["first_frame_cond"]
        x = x.clone()
        x[:, :, 0] = first_frame_cond[:, :, 0]
        return x

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[Dict[str, torch.Tensor]] = None,
        r: Optional[torch.Tensor] = None,
        return_features_early: bool = False,
        feature_indices: Optional[Set[int]] = None,
        return_logvar: bool = False,
        fwd_pred_type: Optional[str] = None,
        **fwd_kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of the StableAvatar model.

        Args:
            x_t: Noisy latent video [B, C, T, H, W].
            t: Timestep [B] or [B, T].
            condition: Dict with text_embeds, first_frame_cond, vocal_embeddings, clip_fea.
            r: Not used (for interface compat).
            return_features_early: If True, return intermediate features.
            feature_indices: Block indices for feature extraction.
            return_logvar: Not supported for StableAvatar.
            fwd_pred_type: Override prediction type.
        """
        assert isinstance(condition, dict), "condition must be a dict"
        assert "text_embeds" in condition, "condition must contain 'text_embeds'"
        assert "first_frame_cond" in condition, "condition must contain 'first_frame_cond'"

        if feature_indices is None:
            feature_indices = {}
        if return_features_early and len(feature_indices) == 0:
            return []

        if fwd_pred_type is None:
            fwd_pred_type = self.net_pred_type
        else:
            assert fwd_pred_type in NET_PRED_TYPES, f"{fwd_pred_type} is not supported"

        text_embeds = condition["text_embeds"]
        first_frame_cond = condition["first_frame_cond"]
        vocal_embeddings = condition.get("vocal_embeddings")
        clip_fea = condition.get("clip_fea")

        # Replace first frame with clean conditioning (Wan 2.2 TI2V style)
        latent_model_input, first_frame_mask = self._replace_first_frame(
            first_frame_cond, x_t, return_mask=True,
        )

        # Expand timestep with zero for first frame
        timestep_mask = first_frame_mask[:, 0]
        timestep = self._compute_timestep_inputs(t, timestep_mask)

        # Prepare inputs for transformer
        # Convert [B, C, T, H, W] -> list of [C, T, H, W]
        x_list = [latent_model_input[i] for i in range(latent_model_input.shape[0])]
        # Convert [B, L, D] -> list of [L, D]
        if isinstance(text_embeds, torch.Tensor):
            context_list = [text_embeds[i] for i in range(text_embeds.shape[0])]
        else:
            context_list = list(text_embeds)

        # Reference conditioning (y): first_frame_cond per sample
        y_list = [first_frame_cond[i] for i in range(first_frame_cond.shape[0])]

        # Compute sequence length
        bsz, C, T, H, W = latent_model_input.shape
        p_t, p_h, p_w = self.transformer.patch_size
        f_patches = T // p_t
        h_patches = H // p_h
        w_patches = W // p_w
        seq_len = f_patches * h_patches * w_patches * 2  # x2 for concat with y

        # Call transformer
        out = self.transformer(
            x=x_list,
            t=timestep,
            context=context_list,
            seq_len=seq_len,
            clip_fea=clip_fea,
            y=y_list,
            vocal_embeddings=vocal_embeddings,
            video_sample_n_frames=self.video_sample_n_frames,
        )

        # Convert model output
        out = self.noise_scheduler.convert_model_output(
            x_t, out, t, src_pred_type=self.net_pred_type, target_pred_type=fwd_pred_type,
        )

        # Replace first frame in output
        out = self._replace_first_frame(first_frame_cond, out)

        if len(feature_indices) > 0:
            return [out, []]

        return out

    def sample(
        self,
        noise: torch.Tensor,
        condition: Optional[Dict[str, torch.Tensor]] = None,
        neg_condition: Optional[Dict[str, torch.Tensor]] = None,
        guidance_scale: Optional[float] = 5.0,
        num_steps: int = 40,
        shift: float = 3.0,
        **kwargs,
    ) -> torch.Tensor:
        """Sample from the bidirectional StableAvatar teacher with CFG."""
        assert self.schedule_type == "rf", f"{self.schedule_type} is not supported"

        first_frame_cond = None
        if isinstance(condition, dict) and "first_frame_cond" in condition:
            first_frame_cond = condition["first_frame_cond"]

        self.unipc_scheduler.config.flow_shift = shift
        self.unipc_scheduler.set_timesteps(num_inference_steps=num_steps, device=noise.device)
        timesteps = self.unipc_scheduler.timesteps

        t_init = timesteps[0] / self.unipc_scheduler.config.num_train_timesteps
        latents = self.noise_scheduler.latents(noise=noise, t_init=t_init)

        for idx, timestep in tqdm(enumerate(timesteps), total=num_steps - 1):
            t = (timestep / self.unipc_scheduler.config.num_train_timesteps).expand(latents.shape[0])
            t = self.noise_scheduler.safe_clamp(
                t, min=self.noise_scheduler.min_t, max=self.noise_scheduler.max_t,
            ).to(latents.dtype)

            flow_pred = self(latents, t, condition=condition)

            if guidance_scale is not None:
                flow_uncond = self(latents, t, condition=neg_condition)
                flow_pred = flow_uncond + guidance_scale * (flow_pred - flow_uncond)

            # Keep first frame clean
            if first_frame_cond is not None:
                flow_pred = flow_pred.clone()
                flow_pred[:, :, 0] = 0.0

            latents = self.unipc_scheduler.step(flow_pred, timestep, latents, return_dict=False)[0]

            if first_frame_cond is not None:
                latents = latents.clone()
                latents[:, :, 0] = first_frame_cond[:, :, 0]

        return latents
