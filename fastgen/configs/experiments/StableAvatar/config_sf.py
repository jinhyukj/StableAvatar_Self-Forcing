# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-Forcing config for StableAvatar-1.3B model (Stage 3 of 3-stage pipeline).

Uses audio-conditioned causal student with bidirectional teacher for
adversarial Self-Forcing training. Data is fully precomputed (latents, audio
embeddings, text embeddings, reference frames).
"""

import os

from fastgen.configs.discriminator import Discriminator_StableAvatar_1B_Config
import fastgen.configs.methods.config_self_forcing as config_self_forcing_default
from fastgen.configs.data import StableAvatarLoaderConfig, StableAvatarValLoaderConfig
from fastgen.configs.callbacks import SyncNetEval_CALLBACK
from fastgen.configs.net import (
    CausalStableAvatar_1B_Config,
    StableAvatar_1B_Config,
    STABLEAVATAR_CKPT,
    CKPT_ROOT_DIR,
)


def create_config():
    config = config_self_forcing_default.create_config()

    config.trainer.max_iter = 10000
    config.trainer.logging_iter = 100
    config.trainer.save_ckpt_iter = 1000

    # Optimizer settings
    config.model.net_optimizer.lr = 5e-6
    config.model.discriminator_optimizer.lr = 5e-6
    config.model.fake_score_optimizer.lr = 5e-6

    config.model.precision = "bfloat16"

    # Latent shape: [C, T_latent, H_latent, W_latent]
    # 512x512 resolution -> 64x64 latent (VAE spatial /8), 81 frames -> 21 latent frames
    config.model.input_shape = [16, 21, 64, 64]

    # Student: causal StableAvatar
    config.model.net = CausalStableAvatar_1B_Config
    config.model.net.total_num_frames = config.model.input_shape[1]

    # Teacher: bidirectional StableAvatar
    config.model.teacher = StableAvatar_1B_Config

    # Discriminator: 32-layer model, features at evenly spaced blocks
    config.model.discriminator = Discriminator_StableAvatar_1B_Config
    config.model.discriminator.disc_type = "multiscale_down_mlp_large"
    config.model.discriminator.feature_indices = [10, 20, 30]

    # GAN settings
    config.model.gan_loss_weight_gen = 0.003
    config.model.gan_use_same_t_noise = True
    config.model.student_sample_type = "ode"
    config.model.fake_score_pred_type = "x0"
    config.model.guidance_scale = 5.0

    # Pretrained model path (bidirectional teacher weights)
    config.model.pretrained_model_path = STABLEAVATAR_CKPT

    # Pretrained student net weights from CausalKD (Stage 2 output)
    # Generate via: python scripts/extract_net_weights.py --input <kd_ckpt>.pth --output ode_init.pt
    config.model.pretrained_student_net_path = f"{CKPT_ROOT_DIR}/stableavatar/ode_init.pt"

    # Timestep sampling
    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    # 4-step training
    config.model.student_sample_steps = 4
    config.model.sample_t_cfg.t_list = [0.999, 0.937, 0.833, 0.624, 0.0]

    # Preprocessors not needed (data is pre-encoded latents)
    config.model.enable_preprocessors = False

    # Dataloader: precomputed StableAvatar data
    config.dataloader_train = StableAvatarLoaderConfig
    config.dataloader_train.batch_size = 1

    config.log_config.group = "stableavatar_sf"
    config.log_config.project = os.getenv("WANDB_PROJECT", "stableavatar")

    # Validation
    config.trainer.validation_iter = 500
    config.dataloader_val = StableAvatarValLoaderConfig

    # SyncNet evaluation callback
    config.trainer.callbacks.update(SyncNetEval_CALLBACK)
    config.trainer.callbacks["syncnet_eval"].val_sets = {
        "recon": "/home/work/stableavatar_data/v2v_validation_data/recon",
        "mixed": "/home/work/stableavatar_data/v2v_validation_data/mixed",
    }

    return config
