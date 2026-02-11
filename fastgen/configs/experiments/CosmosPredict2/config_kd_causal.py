# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CausalKD config for Cosmos-Predict2.5-2B model (Stage 2 of 3-stage pipeline).

Trains a causal student network on pre-computed ODE pairs (from Stage 1)
using MSE regression with inhomogeneous timesteps.
"""

import fastgen.configs.methods.config_kd_causal as config_kd_causal_default
from fastgen.configs.data import PathLoaderConfig
from fastgen.configs.net import (
    CausalCosmosPredict2_2B_Config,
    CKPT_ROOT_DIR,
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
    # 480p (832x480) resolution, 93 frames -> 24 latent frames
    config.model.input_shape = [16, 24, 60, 104]  # cthw - 480p, 93 frames

    # Student: causal (SAC disabled for KV cache compat, use FSDP checkpointing instead)
    config.model.net = CausalCosmosPredict2_2B_Config
    config.model.net.total_num_frames = config.model.input_shape[1]
    config.model.net.use_fsdp_checkpoint = True

    # Pretrained model path (bidirectional teacher weights, loaded into both teacher and student)
    config.model.pretrained_model_path = f"{CKPT_ROOT_DIR}/cosmos_predict2/Cosmos-Predict2.5-2B/base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt"

    # Timestep sampling - must match path data timesteps
    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    # 4-step training
    config.model.student_sample_steps = 4
    config.model.sample_t_cfg.t_list = [0.999, 0.937, 0.833, 0.624, 0.0]

    # Preprocessors not needed (data is pre-encoded latents)
    config.model.enable_preprocessors = False

    # Dataloader: pre-computed ODE pairs
    config.dataloader_train = PathLoaderConfig
    config.dataloader_train.batch_size = 2

    config.log_config.group = "cosmos_predict2_kd_causal"

    return config
