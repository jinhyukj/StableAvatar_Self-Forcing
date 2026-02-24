# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CausalKD config for StableAvatar-1.3B model (Stage 2 of 3-stage pipeline).

Trains a causal student network on pre-computed ODE pairs (from Stage 1)
using MSE regression with inhomogeneous timesteps.

StableAvatar uses precomputed latent data (vae_latents, audio_emb, text_emb,
ref_latents), so no VAE/text encoder is needed at training time.
"""

import os

import fastgen.configs.methods.config_kd_causal as config_kd_causal_default
from fastgen.configs.data import StableAvatarODEPairLoaderConfig
from fastgen.configs.net import (
    CausalStableAvatar_1B_Config,
    STABLEAVATAR_CKPT,
)


def create_config():
    config = config_kd_causal_default.create_config()

    config.trainer.max_iter = 10000
    config.trainer.logging_iter = 100
    config.trainer.save_ckpt_iter = 1000

    # Optimizer settings
    config.model.net_optimizer.lr = 1e-5

    config.model.precision = "bfloat16"

    # Latent shape: [C, T_latent, H_latent, W_latent]
    # 512x512 resolution -> 64x64 latent (VAE spatial /8), 81 frames -> 21 latent frames
    config.model.input_shape = [16, 21, 64, 64]

    # Student: causal StableAvatar
    config.model.net = CausalStableAvatar_1B_Config
    config.model.net.total_num_frames = config.model.input_shape[1]

    # Pretrained model path (bidirectional teacher weights, loaded into student)
    config.model.pretrained_model_path = STABLEAVATAR_CKPT

    # Timestep sampling
    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    # 4-step training
    config.model.student_sample_steps = 4
    config.model.sample_t_cfg.t_list = [0.999, 0.937, 0.833, 0.624, 0.0]

    # Preprocessors not needed (data is pre-encoded latents)
    config.model.enable_preprocessors = False

    # Dataloader: pre-computed ODE pairs with audio/text/ref data
    config.dataloader_train = StableAvatarODEPairLoaderConfig
    config.dataloader_train.batch_size = 1

    config.log_config.group = "stableavatar_kd_causal"
    config.log_config.project = os.getenv("WANDB_PROJECT", "stableavatar")

    return config
