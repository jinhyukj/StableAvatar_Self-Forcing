# Modified from StableAvatar vocal_projector_fantasy.py
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F


class VocalProjModel(nn.Module):
    def __init__(self, audio_in_dim=1024, cross_attention_dim=1024):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.proj = torch.nn.Linear(audio_in_dim, cross_attention_dim, bias=False)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, audio_embeds):
        context_tokens = self.proj(audio_embeds)
        context_tokens = self.norm(context_tokens)
        return context_tokens  # [B,L,C]


class FantasyTalkingVocalConditionModel(nn.Module):
    def __init__(self, audio_in_dim: int, audio_proj_dim: int):
        super().__init__()
        self.audio_in_dim = audio_in_dim
        self.audio_proj_dim = audio_proj_dim
        self.proj_model = self.init_proj(self.audio_proj_dim)

    def init_proj(self, cross_attention_dim=5120):
        proj_model = VocalProjModel(
            audio_in_dim=self.audio_in_dim, cross_attention_dim=cross_attention_dim
        )
        return proj_model

    def forward(self, audio_fea=None):
        return self.proj_model(audio_fea) if audio_fea is not None else None


def split_audio_sequence(audio_proj_length, num_frames=81):
    """
    Map the audio feature sequence to corresponding latent frame slices.

    Args:
        audio_proj_length (int): The total length of the audio feature sequence.
        num_frames (int): The number of video frames in the training data (default: 81).

    Returns:
        list: A list of [start_idx, end_idx] pairs for each latent frame.
    """
    tokens_per_frame = audio_proj_length / num_frames
    tokens_per_latent_frame = tokens_per_frame * 4
    half_tokens = int(tokens_per_latent_frame / 2)

    pos_indices = []
    for i in range(int((num_frames - 1) / 4) + 1):
        if i == 0:
            pos_indices.append(0)
        else:
            start_token = tokens_per_frame * ((i - 1) * 4 + 1)
            end_token = tokens_per_frame * (i * 4 + 1)
            center_token = int((start_token + end_token) / 2) - 1
            pos_indices.append(center_token)

    pos_idx_ranges = [[idx - half_tokens, idx + half_tokens] for idx in pos_indices]
    pos_idx_ranges[0] = [
        -(half_tokens * 2 - pos_idx_ranges[1][0]),
        pos_idx_ranges[1][0],
    ]

    return pos_idx_ranges


def split_tensor_with_padding(input_tensor, pos_idx_ranges, expand_length=0):
    """
    Split the input tensor into subsequences based on index ranges, and apply
    right-side zero-padding if the range exceeds the input boundaries.

    Args:
        input_tensor (Tensor): Input audio tensor of shape [1, L, 768].
        pos_idx_ranges (list): A list of index ranges.
        expand_length (int): Number of tokens to expand on both sides.

    Returns:
        sub_sequences (Tensor): Shape [1, F, L, 768].
        k_lens (Tensor): Shape [F], actual (unpadded) length of each subsequence.
    """
    pos_idx_ranges = [
        [idx[0] - expand_length, idx[1] + expand_length] for idx in pos_idx_ranges
    ]
    sub_sequences = []
    seq_len = input_tensor.size(1)
    max_valid_idx = seq_len - 1
    k_lens_list = []
    for start, end in pos_idx_ranges:
        pad_front = max(-start, 0)
        pad_back = max(end - max_valid_idx, 0)
        valid_start = max(start, 0)
        valid_end = min(end, max_valid_idx)

        if valid_start <= valid_end:
            valid_part = input_tensor[:, valid_start: valid_end + 1, :]
        else:
            valid_part = input_tensor.new_zeros((1, 0, input_tensor.size(2)))

        padded_subseq = F.pad(
            valid_part,
            (0, 0, 0, pad_back + pad_front, 0, 0),
            mode="constant",
            value=0,
        )
        k_lens_list.append(padded_subseq.size(-2) - pad_back - pad_front)
        sub_sequences.append(padded_subseq)

    return torch.stack(sub_sequences, dim=1), torch.tensor(
        k_lens_list, dtype=torch.long
    )
