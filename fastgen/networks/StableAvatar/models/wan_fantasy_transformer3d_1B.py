# Modified from StableAvatar wan_fantasy_transformer3d_1B.py
# Original: https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modifications for Self-Forcing integration:
# - Stripped sequence-parallel code (not needed for training)
# - Stripped TeaCache code (not needed for training)
# - Fixed imports to use local modules
# - Added attention_kwargs passthrough for causal KV caching

import math
import warnings
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fastgen.networks.StableAvatar.models.vocal_projector_fantasy_1B import (
    FantasyTalkingVocalCondition1BModel,
)

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False


def flash_attention(
    q, k, v,
    q_lens=None, k_lens=None,
    dropout_p=0., softmax_scale=None, q_scale=None,
    causal=False, window_size=(-1, -1), deterministic=False,
    dtype=torch.bfloat16, version=None,
):
    """
    q: [B, Lq, Nq, C1], k: [B, Lk, Nk, C1], v: [B, Lk, Nk, C2].
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor([lq] * b, dtype=torch.int32).to(device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor([lk] * b, dtype=torch.int32).to(device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)
    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn('Flash attention 3 is not available, use flash attention 2 instead.')

    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q, k=k, v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None, seqused_k=None,
            max_seqlen_q=lq, max_seqlen_k=lk,
            softmax_scale=softmax_scale, causal=causal, deterministic=deterministic,
        )
        if isinstance(x, tuple):
            x = x[0]
        x = x.unflatten(0, (b, lq))
    elif FLASH_ATTN_2_AVAILABLE:
        x = flash_attn.flash_attn_varlen_func(
            q=q, k=k, v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq, max_seqlen_k=lk,
            dropout_p=dropout_p, softmax_scale=softmax_scale,
            causal=causal, window_size=window_size, deterministic=deterministic,
        ).unflatten(0, (b, lq))
    else:
        # Fallback to scaled_dot_product_attention
        q_r = q.unflatten(0, (b, lq)).transpose(1, 2)
        k_r = k.unflatten(0, (b, lk)).transpose(1, 2)
        v_r = v.unflatten(0, (b, lk)).transpose(1, 2)
        x = F.scaled_dot_product_attention(q_r, k_r, v_r, is_causal=causal, dropout_p=dropout_p)
        x = x.transpose(1, 2).contiguous()
        return x.type(out_dtype)

    return x.type(out_dtype)


def attention(
    q, k, v,
    q_lens=None, k_lens=None,
    dropout_p=0., softmax_scale=None, q_scale=None,
    causal=False, window_size=(-1, -1), deterministic=False,
    dtype=torch.bfloat16, fa_version=None,
):
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q, k=k, v=v, q_lens=q_lens, k_lens=k_lens,
            dropout_p=dropout_p, softmax_scale=softmax_scale,
            q_scale=q_scale, causal=causal, window_size=window_size,
            deterministic=deterministic, dtype=dtype, version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn('Padding mask is disabled when using scaled_dot_product_attention.')
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal, dropout_p=dropout_p)
        out = out.transpose(1, 2).contiguous()
        return out


def sinusoidal_embedding_1d(dim, position):
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@amp.autocast('cuda', enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@amp.autocast('cuda', enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float32).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(seq_len, 1, -1)

        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).float()


class WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs, dtype):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        def qkv_fn(x):
            q = self.norm_q(self.q(x.to(dtype))).view(b, s, n, d)
            k = self.norm_k(self.k(x.to(dtype))).view(b, s, n, d)
            v = self.v(x.to(dtype)).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        x = attention(
            q=rope_apply(q, grid_sizes, freqs).to(dtype),
            k=rope_apply(k, grid_sizes, freqs).to(dtype),
            v=v.to(dtype),
            k_lens=seq_lens,
            window_size=self.window_size,
        )
        x = x.to(dtype)
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):
    def forward(self, x, context, context_lens, dtype):
        b, n, d = x.size(0), self.num_heads, self.head_dim
        q = self.norm_q(self.q(x.to(dtype))).view(b, -1, n, d)
        k = self.norm_k(self.k(context.to(dtype))).view(b, -1, n, d)
        v = self.v(context.to(dtype)).view(b, -1, n, d)

        x = attention(q.to(dtype), k.to(dtype), v.to(dtype), k_lens=context_lens)
        x = x.to(dtype)
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)
        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens, dtype):
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        q = self.norm_q(self.q(x.to(dtype))).view(b, -1, n, d)
        k = self.norm_k(self.k(context.to(dtype))).view(b, -1, n, d)
        v = self.v(context.to(dtype)).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img.to(dtype))).view(b, -1, n, d)
        v_img = self.v_img(context_img.to(dtype)).view(b, -1, n, d)

        img_x = attention(q.to(dtype), k_img.to(dtype), v_img.to(dtype), k_lens=None)
        img_x = img_x.to(dtype)
        x = attention(q.to(dtype), k.to(dtype), v.to(dtype), k_lens=context_lens)
        x = x.to(dtype)

        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


class WanI2VTalkingCrossAttention(WanSelfAttention):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6,
                 audio_context_dim=1024):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        self.k_vocal = nn.Linear(dim, dim)
        self.v_vocal = nn.Linear(dim, dim)

        nn.init.zeros_(self.k_vocal.weight)
        nn.init.zeros_(self.v_vocal.weight)
        if self.k_vocal.bias is not None:
            nn.init.zeros_(self.k_vocal.bias)
        if self.v_vocal.bias is not None:
            nn.init.zeros_(self.v_vocal.bias)

    def forward(self, x, context, context_lens, dtype,
                vocal_context=None, vocal_context_lens=None, latents_num_frames=None):
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        q = self.norm_q(self.q(x.to(dtype))).view(b, -1, n, d)
        k = self.norm_k(self.k(context.to(dtype))).view(b, -1, n, d)
        v = self.v(context.to(dtype)).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img.to(dtype))).view(b, -1, n, d)
        v_img = self.v_img(context_img.to(dtype)).view(b, -1, n, d)

        img_x = attention(q.to(dtype), k_img.to(dtype), v_img.to(dtype), k_lens=None)
        img_x = img_x.to(dtype)

        x = attention(q.to(dtype), k.to(dtype), v.to(dtype), k_lens=context_lens)
        x = x.to(dtype)

        if latents_num_frames is None:
            latents_num_frames = 21

        if vocal_context is not None and len(vocal_context.shape) == 4:
            vocal_q = q.view(b * latents_num_frames, -1, n, d)
            vocal_ip_key = self.k_vocal(vocal_context).view(b * latents_num_frames, -1, n, d)
            vocal_ip_value = self.v_vocal(vocal_context).view(b * latents_num_frames, -1, n, d)
            vocal_x = attention(
                vocal_q.to(dtype), vocal_ip_key.to(dtype), vocal_ip_value.to(dtype),
                k_lens=vocal_context_lens,
            )
            vocal_x = vocal_x.view(b, q.size(1), n, d)
            vocal_x = vocal_x.flatten(2)
        elif vocal_context is not None:
            vocal_ip_key = self.k_vocal(vocal_context).view(b, -1, n, d)
            vocal_ip_value = self.v_vocal(vocal_context).view(b, -1, n, d)
            vocal_x = attention(
                q.to(dtype), vocal_ip_key.to(dtype), vocal_ip_value.to(dtype), k_lens=None,
            )
            vocal_x = vocal_x.flatten(2)
        else:
            vocal_x = 0

        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x + vocal_x
        x = self.o(x)
        return x


class WanAttentionBlock(nn.Module):
    def __init__(self, cross_attn_type, dim, ffn_dim, num_heads,
                 window_size=(-1, -1), qk_norm=True, cross_attn_norm=False, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanI2VTalkingCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)

    def forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens,
                dtype=torch.float32, vocal_context=None, vocal_context_lens=None,
                latents_num_frames=None, **kwargs):
        e = (self.modulation + e).chunk(6, dim=1)

        # self-attention
        temp_x = self.norm1(x) * (1 + e[1]) + e[0]
        temp_x = temp_x.to(dtype)
        y = self.self_attn(temp_x, seq_lens, grid_sizes, freqs, dtype)
        x = x + y * e[2]

        # cross-attention & ffn
        def cross_attn_ffn(x, context, context_lens, e, vocal_context, vocal_context_lens):
            x = x + self.cross_attn(
                self.norm3(x), context, context_lens, dtype,
                vocal_context, vocal_context_lens, latents_num_frames,
            )
            temp_x = self.norm2(x) * (1 + e[4]) + e[3]
            temp_x = temp_x.to(dtype)
            y = self.ffn(temp_x)
            x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, context_lens, e, vocal_context, vocal_context_lens)
        return x


class Head(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim ** 0.5)

    def forward(self, x, e):
        e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        x = self.head(self.norm(x) * (1 + e[1]) + e[0])
        return x


class MLPProj(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))

    def forward(self, image_embeds):
        return self.proj(image_embeds)


class WanTransformer3DFantasyModel(nn.Module):
    """Wan diffusion backbone for StableAvatar with audio conditioning.

    Supports image-to-video generation with vocal/audio conditioning via
    FantasyTalkingVocalCondition1BModel.
    """

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        model_type='i2v',
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
    ):
        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # Embeddings
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'), nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # Blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # Head
        self.head = Head(dim, out_dim, patch_size, eps)

        # RoPE frequencies
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.d = d
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ], dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        self.gradient_checkpointing = False

        # Vocal projector
        self.vocal_projector = FantasyTalkingVocalCondition1BModel(
            audio_in_dim=768, audio_proj_dim=1536, dit_dim=dim)

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        vocal_embeddings=None,
        video_sample_n_frames=81,
        attention_kwargs=None,
        feature_indices=None,
        return_features_early=False,
    ):
        """Forward pass through the diffusion model.

        Args:
            x (List[Tensor]): List of input video tensors, each [C_in, F, H, W]
            t (Tensor): Diffusion timesteps [B] or [B, T]
            context (List[Tensor]): List of text embeddings [L, C]
            seq_len (int): Maximum sequence length for positional encoding
            clip_fea (Tensor): CLIP image features for i2v mode
            y (List[Tensor]): Conditional video inputs for i2v mode
            vocal_embeddings (Tensor): Audio embeddings [B, T_audio, 768]
            video_sample_n_frames (int): Number of video frames
            attention_kwargs (dict): Extra kwargs for causal attention (KV cache)
            feature_indices (set): Block indices to extract features from for discriminator.
            return_features_early (bool): If True and all features collected, return early.
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None

        device = self.patch_embedding.weight.device
        dtype = x.dtype
        if self.freqs.device != device and torch.device(type="meta") != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # Patch embedding
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x
        ])

        # Time embeddings
        with amp.autocast('cuda', dtype=torch.float32):
            e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            e0 = e0.to(dtype)
            e = e.to(dtype)

        # Context embeddings (text + CLIP image)
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ])
        )
        context_clip = self.img_emb(clip_fea)
        context = torch.concat([context_clip, context], dim=1)

        # Vocal embeddings
        if vocal_embeddings is not None:
            if vocal_embeddings.size()[0] > 1:
                vocal_embeddings_single = vocal_embeddings[-1:]
                vocal_context, vocal_context_lens = self.vocal_projector(
                    vocal_embeddings=vocal_embeddings_single,
                    video_sample_n_frames=video_sample_n_frames,
                    latents=x[-1:], e0=e0[-1:], e=e[-1:],
                )
                vocal_context = torch.cat([torch.zeros_like(vocal_context), vocal_context, vocal_context])
            else:
                vocal_context, vocal_context_lens = self.vocal_projector(
                    vocal_embeddings=vocal_embeddings,
                    video_sample_n_frames=video_sample_n_frames,
                    latents=x, e0=e0, e=e,
                )
        else:
            vocal_context = None
            vocal_context_lens = None

        # Transformer blocks
        features = []
        frames_per_batch = (video_sample_n_frames - 1) // 4 + 1
        for idx, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, e0, seq_lens, grid_sizes, self.freqs,
                    context, context_lens, dtype,
                    vocal_context, vocal_context_lens, frames_per_batch,
                    **ckpt_kwargs,
                )
            else:
                kwargs = dict(
                    e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes,
                    freqs=self.freqs, context=context, context_lens=context_lens,
                    dtype=dtype, vocal_context=vocal_context,
                    vocal_context_lens=vocal_context_lens,
                    latents_num_frames=frames_per_batch,
                )
                x = block(x, **kwargs)

            if feature_indices is not None and idx in feature_indices:
                features.append(x)
                if return_features_early and len(features) == len(feature_indices):
                    return features

        # Head
        x = self.head(x, e)

        # Unpatchify
        x = self.unpatchify(x, grid_sizes)
        x = torch.stack(x)

        if feature_indices is not None and len(features) > 0:
            return x, features
        return x

    def unpatchify(self, x, grid_sizes):
        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out
