# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-Forcing config for Cosmos-Predict2.5-2B model."""

from fastgen.configs.discriminator import Discriminator_CosmosPredict2_2B_Config
import fastgen.configs.methods.config_self_forcing as config_self_forcing_default
from fastgen.configs.data import VideoLoaderConfig
from fastgen.configs.net import (
    CausalCosmosPredict2_2B_Config,
    CosmosPredict2_2B_Aggressive_Config,
    CKPT_ROOT_DIR,
)


def create_config():
    config = config_self_forcing_default.create_config()

    config.trainer.max_iter = 10000
    config.trainer.logging_iter = 100
    config.trainer.save_ckpt_iter = 1000

    # Optimizer settings
    config.model.net_optimizer.lr = 1e-5
    config.model.discriminator_optimizer.lr = 1e-5
    config.model.fake_score_optimizer.lr = 1e-5

    config.model.precision = "bfloat16"

    # Latent shape: [C, T_latent, H_latent, W_latent]
    # 480p (832x480) resolution, 93 frames -> 24 latent frames
    config.model.input_shape = [16, 24, 60, 104]  # cthw - 480p, 93 frames

    # Student: causal (SAC disabled for KV cache compat, use FSDP checkpointing instead)
    config.model.net = CausalCosmosPredict2_2B_Config
    config.model.net.total_num_frames = config.model.input_shape[1]
    config.model.net.use_fsdp_checkpoint = True

    # Teacher: non-causal with AGGRESSIVE SAC for memory savings
    config.model.teacher = CosmosPredict2_2B_Aggressive_Config

    # Discriminator
    config.model.discriminator = Discriminator_CosmosPredict2_2B_Config
    config.model.discriminator.disc_type = "multiscale_down_mlp_large"
    config.model.discriminator.feature_indices = [13, 20, 27]

    # GAN settings
    config.model.gan_loss_weight_gen = 0.003
    config.model.gan_use_same_t_noise = True
    config.model.student_sample_type = "ode"
    config.model.fake_score_pred_type = "x0"
    config.model.guidance_scale = 3.0

    # Pretrained model path (bidirectional teacher weights)
    config.model.pretrained_model_path = f"{CKPT_ROOT_DIR}/cosmos_predict2/Cosmos-Predict2.5-2B/base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt"

    # Pretrained student net weights from CausalKD (Stage 2 output)
    # Generate via: python scripts/extract_net_weights.py --input <kd_ckpt>.pth --output ode_init.pt
    config.model.pretrained_student_net_path = f"{CKPT_ROOT_DIR}/cosmos_predict2/ode_init.pt"

    # Timestep sampling
    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    # 4-step training
    config.model.student_sample_steps = 4
    config.model.sample_t_cfg.t_list = [0.999, 0.937, 0.833, 0.624, 0.0]

    # Dataloader settings
    config.dataloader_train = VideoLoaderConfig
    config.dataloader_train.batch_size = 1
    config.dataloader_train.img_size = (config.model.input_shape[-1] * 8, config.model.input_shape[-2] * 8)
    config.dataloader_train.sequence_length = (config.model.input_shape[1] - 1) * 4 + 1

    config.log_config.group = "cosmos_predict2_sf"

    return config
