"""
Causal Cosmos Predict2 network with inline KV caching for Self-Forcing training.

This module implements CausalCosmosPredict2 which adds causal (autoregressive) generation
capabilities to CosmosPredict2 via monkey-patching, following the same pattern used by
CausalWan (fastgen/networks/Wan/network_causal.py).

Key differences from CosmosPredict2:
- Inherits from CausalFastGenNetwork (chunk_size, total_num_frames, clear_caches)
- SAC is forced to NONE (incompatible with KV cache)
- Block forward is monkey-patched to support external KV caches
- RoPE forward is patched to support temporal offsets for AR decoding
- DiT forward is patched to manage cache allocation and block masks
"""

import types
import math
import os
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tqdm.auto import tqdm
from diffusers import UniPCMultistepScheduler

from fastgen.networks.network import CausalFastGenNetwork
from fastgen.networks.cosmos_predict2.network import CosmosPredict2
from fastgen.networks.cosmos_predict2.modules import (
    Block,
    CheckpointMode,
    SACConfig,
)
from fastgen.networks.noise_schedule import NET_PRED_TYPES
from fastgen.utils.distributed.fsdp import apply_fsdp_checkpointing
import fastgen.utils.logging_utils as logger

# Optional FlexAttention (falls back to SDP if unavailable)
_disable_flex_env = os.environ.get("FASTGEN_DISABLE_FLEX_ATTENTION", "0") == "1"
try:
    if _disable_flex_env:
        raise ImportError("FlexAttention disabled via FASTGEN_DISABLE_FLEX_ATTENTION=1")
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention, BlockMask

    FLEX_ATTENTION_AVAILABLE = True
    try:
        import torch._dynamo as _dynamo

        _dynamo.config.optimize_ddp = False
    except Exception:
        pass

    _compile_mode = os.environ.get("TORCH_COMPILE_MODE", "default")
    _disable_compile_wrap = (
        os.environ.get("TORCH_COMPILE_DISABLE", "0") == "1" or os.environ.get("FASTGEN_FLEX_COMPILE", "1") == "0"
    )
    if not _disable_compile_wrap:
        flex_attention = torch.compile(flex_attention, dynamic=False, mode=_compile_mode)
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    create_block_mask = None  # type: ignore
    flex_attention = None  # type: ignore

    class BlockMask:  # type: ignore
        pass


# ---------------------------------------------------------------------------
# Helper functions (module-level, bound via types.MethodType or called directly)
# ---------------------------------------------------------------------------


def _cosmos_rope_forward_with_time_offset(
    self, x_B_T_H_W_C: torch.Tensor, fps: Optional[torch.Tensor] = None, start_frame: int = 0
) -> torch.Tensor:
    """Rotary embedding with a temporal offset for autoregressive causal decoding.

    Mirrors CausalWan's ``_rope_forward_with_time_offset`` but adapts to
    ``VideoRopePosition3DEmb.generate_embeddings`` logic.

    Args:
        x_B_T_H_W_C: Patched input tensor of shape (B, T, H, W, C).
        fps: Optional FPS tensor for temporal conditioning.
        start_frame: Temporal offset in units of *patched* frames.

    Returns:
        RoPE embeddings of shape (T*H*W, 1, 1, D).
    """
    # Ensure buffers are on the same device as input
    if self.seq.device != x_B_T_H_W_C.device:
        self.seq = self.seq.to(x_B_T_H_W_C.device)
        self.dim_spatial_range = self.dim_spatial_range.to(x_B_T_H_W_C.device)
        self.dim_temporal_range = self.dim_temporal_range.to(x_B_T_H_W_C.device)

    B, T, H, W, _ = x_B_T_H_W_C.shape

    h_theta = 10000.0 * self.h_ntk_factor
    w_theta = 10000.0 * self.w_ntk_factor
    t_theta = 10000.0 * self.t_ntk_factor

    h_spatial_freqs = 1.0 / (h_theta ** self.dim_spatial_range.float())
    w_spatial_freqs = 1.0 / (w_theta ** self.dim_spatial_range.float())
    temporal_freqs = 1.0 / (t_theta ** self.dim_temporal_range.float())

    half_emb_h = torch.outer(self.seq[:H], h_spatial_freqs)
    half_emb_w = torch.outer(self.seq[:W], w_spatial_freqs)

    # Temporal: use offset slice self.seq[start_frame : start_frame + T]
    temporal_seq = self.seq[start_frame : start_frame + T]
    # Pad if we overrun the buffer
    if temporal_seq.shape[0] < T:
        pad_val = self.seq[-1:].expand(T - temporal_seq.shape[0])
        temporal_seq = torch.cat([temporal_seq, pad_val], dim=0)

    if self.enable_fps_modulation and fps is not None:
        half_emb_t = torch.outer(temporal_seq / fps[:1] * self.base_fps, temporal_freqs)
    else:
        half_emb_t = torch.outer(temporal_seq, temporal_freqs)

    from einops import repeat

    em_T_H_W_D = torch.cat(
        [
            repeat(half_emb_t, "t d -> t h w d", h=H, w=W),
            repeat(half_emb_h, "h d -> t h w d", t=T, w=W),
            repeat(half_emb_w, "w d -> t h w d", t=T, h=H),
        ]
        * 2,
        dim=-1,
    )

    return rearrange(em_T_H_W_D, "t h w d -> (t h w) 1 1 d").float()


def _prepare_blockwise_causal_attn_mask_cosmos(
    device: torch.device,
    num_frames: int,
    frame_seqlen: int,
    chunk_size: int,
) -> "BlockMask | None":
    """Construct a block-wise causal attention mask for FlexAttention.

    Mirrors CausalWan's ``_prepare_blockwise_causal_attn_mask``.

    Cosmos blocks operate on flattened ``(B, T*H*W, D)`` tokens where each frame
    contributes ``frame_seqlen = H_patched * W_patched`` tokens.
    """
    if not FLEX_ATTENTION_AVAILABLE:
        return None

    logger.info("creating blockwise causal attn mask for Cosmos teacher-forcing / diffusion-forcing")

    total_length = num_frames * frame_seqlen
    padded_length = math.ceil(total_length / 128) * 128 - total_length

    ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

    num_chunks = num_frames // chunk_size
    remaining_size = num_frames % chunk_size

    frame_counts: List[int] = []
    if num_frames > 0:
        if num_chunks == 0:
            frame_counts.append(remaining_size)
        else:
            frame_counts.append(chunk_size + remaining_size)
            frame_counts.extend([chunk_size] * max(num_chunks - 1, 0))

    current_start = 0
    for frames_in_chunk in frame_counts:
        chunk_len_tokens = frames_in_chunk * frame_seqlen
        ends[current_start : current_start + chunk_len_tokens] = current_start + chunk_len_tokens
        current_start += chunk_len_tokens

    def attention_mask(b, h, q_idx, kv_idx) -> torch.Tensor:
        return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)

    block_mask = create_block_mask(
        attention_mask,
        B=None,
        H=None,
        Q_LEN=total_length + padded_length,
        KV_LEN=total_length + padded_length,
        _compile=False,
        device=device,
    )
    return block_mask


def _create_external_caches_cosmos(
    transformer: nn.Module,
    x_B_T_H_W_D: torch.Tensor,
    crossattn_emb: torch.Tensor,
    num_frames: int,
    total_num_frames: int,
    cache_tag: str,
    cur_start_frame: int,
) -> None:
    """Preallocate external KV caches per block for causal generation.

    Mirrors CausalWan's ``_create_external_caches``.
    """
    B, T, H, W, D = x_B_T_H_W_D.shape
    frame_seqlen = H * W
    capacity_tokens = frame_seqlen * total_num_frames

    if not hasattr(transformer, "_external_self_kv_list"):
        transformer._external_self_kv_list = []
    if not hasattr(transformer, "_external_cross_kv_list"):
        transformer._external_cross_kv_list = []
    self_list = transformer._external_self_kv_list
    cross_list = transformer._external_cross_kv_list

    need_reinit = len(self_list) != len(transformer.blocks) or len(cross_list) != len(transformer.blocks)
    if not need_reinit and len(self_list) > 0 and self_list[0] is not None:
        existing_cache = self_list[0].get(cache_tag, {}).get("k", None)
        if existing_cache is not None and existing_cache.shape[0] != B:
            logger.info(f"Batch size changed from {existing_cache.shape[0]} to {B}, reallocating caches")
            need_reinit = True

    if need_reinit:
        logger.info("creating external caches for Cosmos causal generation")
        self_list.clear()
        cross_list.clear()
        for _ in transformer.blocks:
            self_list.append({})
            cross_list.append({})

    if any(cache_tag not in cache for cache in self_list):
        for i, block in enumerate(transformer.blocks):
            n_heads = block.self_attn.n_heads
            head_dim = block.self_attn.head_dim

            want_shape = (B, capacity_tokens, n_heads, head_dim)
            need_alloc = False
            if cache_tag not in self_list[i]:
                need_alloc = True
            else:
                k_buf = self_list[i][cache_tag].get("k", None)
                v_buf = self_list[i][cache_tag].get("v", None)
                if (
                    not isinstance(k_buf, torch.Tensor)
                    or not isinstance(v_buf, torch.Tensor)
                    or tuple(k_buf.shape) != want_shape
                    or tuple(v_buf.shape) != want_shape
                ):
                    need_alloc = True

            if need_alloc:
                ctx_len = crossattn_emb.shape[1]
                want_shape_cross = (B, ctx_len, n_heads, head_dim)
                device = x_B_T_H_W_D.device
                dtype = x_B_T_H_W_D.dtype
                self_list[i][cache_tag] = {
                    "k": torch.zeros(want_shape, device=device, dtype=dtype),
                    "v": torch.zeros(want_shape, device=device, dtype=dtype),
                    "len": cur_start_frame * frame_seqlen,
                }
                cross_list[i][cache_tag] = {
                    "k": torch.zeros(want_shape_cross, device=device, dtype=dtype),
                    "v": torch.zeros(want_shape_cross, device=device, dtype=dtype),
                    "is_init": False,
                }


def _cosmos_self_attention_with_cache(
    attn: nn.Module,
    x: torch.Tensor,
    rope_emb: torch.Tensor,
    block_mask: "BlockMask | None",
    external_self_kv: Dict[str, Any],
    cache_tag: str,
    store_kv: bool,
    cache_start_idx: Optional[int],
) -> torch.Tensor:
    """Self-attention with external KV cache support.

    Mirrors the self-attention path of ``CausalWanAttnProcessor``.

    Args:
        attn: The Cosmos ``Attention`` module.
        x: Flattened hidden states ``(B, L, D)``.
        rope_emb: RoPE embeddings ``(L, 1, 1, D)``.
        block_mask: FlexAttention block mask (teacher-forcing).
        external_self_kv: Per-tag KV cache dict.
        cache_tag: Cache identifier.
        store_kv: Whether to store current K/V into cache.
        cache_start_idx: Override for cache start position.

    Returns:
        Attention output ``(B, L, D)``.
    """
    q, k, v = attn.compute_qkv(x, None, rope_emb=rope_emb)
    # q, k, v: (B, L, H, D)

    bsz, seqlen, nheads, head_dim = q.shape
    per_tag = external_self_kv.get(cache_tag, {}) if external_self_kv else {}
    use_external_cache = (
        bool(per_tag) and "k" in per_tag and "v" in per_tag and "len" in per_tag
    ) and block_mask is None
    use_flex_attention = not use_external_cache and FLEX_ATTENTION_AVAILABLE and block_mask is not None

    if use_external_cache:
        k_buf, v_buf = per_tag["k"], per_tag["v"]
        if cache_start_idx is None:
            cache_start_idx = int(per_tag["len"])
        end_idx = cache_start_idx + seqlen

        if store_kv:
            per_tag["len"] = end_idx
            k_buf[:, cache_start_idx:end_idx, :, :] = k.detach()
            v_buf[:, cache_start_idx:end_idx, :, :] = v.detach()

        if cache_start_idx == 0:
            k_full, v_full = k, v
        else:
            with torch.no_grad():
                k_cached = k_buf[:, :cache_start_idx, :, :]
                v_cached = v_buf[:, :cache_start_idx, :, :]
            k_full = torch.cat([k_cached, k], dim=1)
            v_full = torch.cat([v_cached, v], dim=1)

        # (B, L, H, D) -> (B, H, L, D) for SDPA
        q_sdpa = q.permute(0, 2, 1, 3)
        k_sdpa = k_full.permute(0, 2, 1, 3)
        v_sdpa = v_full.permute(0, 2, 1, 3)
        out = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa)
        out = out.permute(0, 2, 1, 3)  # back to (B, L, H, D)

    elif use_flex_attention:
        FLEX_PADDING_SIZE = 128
        padded_len = (math.ceil(seqlen / FLEX_PADDING_SIZE) * FLEX_PADDING_SIZE) - seqlen

        if padded_len > 0:
            pad_shape = (bsz, padded_len, nheads, head_dim)
            pad_q = torch.zeros(pad_shape, device=q.device, dtype=q.dtype)
            pad_k = torch.zeros(pad_shape, device=k.device, dtype=k.dtype)
            pad_v = torch.zeros(pad_shape, device=v.device, dtype=v.dtype)
            q = torch.cat([q, pad_q], dim=1)
            k = torch.cat([k, pad_k], dim=1)
            v = torch.cat([v, pad_v], dim=1)

        # flex_attention expects (B, H, L, D)
        q, k, v = (t.permute(0, 2, 1, 3).contiguous() for t in (q, k, v))
        out = flex_attention(query=q, key=k, value=v, block_mask=block_mask)
        out = out.permute(0, 2, 1, 3)

        if padded_len > 0:
            out = out[:, :-padded_len, :, :]
    else:
        # Standard SDPA
        q_sdpa = q.permute(0, 2, 1, 3)
        k_sdpa = k.permute(0, 2, 1, 3)
        v_sdpa = v.permute(0, 2, 1, 3)
        out = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa)
        out = out.permute(0, 2, 1, 3)

    # (B, L, H, D) -> (B, L, H*D)
    result = rearrange(out, "b s h d -> b s (h d)")
    return attn.output_dropout(attn.output_proj(result))


def _cosmos_cross_attention_with_cache(
    attn: nn.Module,
    x: torch.Tensor,
    context: torch.Tensor,
    external_cross_kv: Dict[str, Any],
    cache_tag: str,
    store_kv: bool,
) -> torch.Tensor:
    """Cross-attention with static cache for text conditioning.

    Mirrors the cross-attention cache path of ``CausalWanAttnProcessor``.
    """
    k_use, v_use = None, None
    use_cache = bool(external_cross_kv)

    if use_cache:
        per_tag = external_cross_kv.get(cache_tag, {})
        cache_initialized = per_tag and bool(per_tag.get("is_init", False))
        grad_enabled = torch.is_grad_enabled()

        if cache_initialized and not grad_enabled:
            k_use = per_tag["k"]
            v_use = per_tag["v"]

    if k_use is not None and v_use is not None:
        # Use cached K/V — only need to compute Q
        q = attn.q_proj(x)
        q = rearrange(q, "b ... (h d) -> b ... h d", h=attn.n_heads, d=attn.head_dim)
        q = attn.q_norm(q)
        k, v = k_use, v_use
    else:
        q, k, v = attn.compute_qkv(x, context, rope_emb=None)
        if use_cache and store_kv:
            external_cross_kv[cache_tag] = {"k": k, "v": v, "is_init": True}

    q_sdpa = q.permute(0, 2, 1, 3)
    k_sdpa = k.permute(0, 2, 1, 3).to(q_sdpa.dtype)
    v_sdpa = v.permute(0, 2, 1, 3).to(q_sdpa.dtype)
    out = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa)
    result = rearrange(out.permute(0, 2, 1, 3), "b s h d -> b s (h d)")
    return attn.output_dropout(attn.output_proj(result))


def _cosmos_block_forward_with_cache(
    self,
    x_B_T_H_W_D: torch.Tensor,
    emb_B_T_D: torch.Tensor,
    crossattn_emb: torch.Tensor,
    rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
    adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
    extra_per_block_pos_emb: Optional[torch.Tensor] = None,
    crossattn_gate_scale: float = 1.0,
    block_mask: "BlockMask | None" = None,
    attention_cache_kwargs: Optional[Dict[str, Any]] = None,
    cache_start_idx: Optional[int] = None,
) -> torch.Tensor:
    """Block forward with inline KV cache support.

    Mirrors CausalWan's ``_wan_block_forward_inline_cache`` but preserves
    Cosmos' AdaLN modulation logic.
    """
    attention_cache_kwargs = attention_cache_kwargs or {}
    cache_tag = attention_cache_kwargs.get("cache_tag", "pos")
    store_kv = attention_cache_kwargs.get("store_kv", False)
    external_self_kv = attention_cache_kwargs.get("external_self_kv", {})
    external_cross_kv = attention_cache_kwargs.get("external_cross_kv", {})

    if extra_per_block_pos_emb is not None:
        x_B_T_H_W_D = x_B_T_H_W_D + extra_per_block_pos_emb

    # Compute modulation parameters (fp32 for numerical stability if configured)
    with torch.amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
        if self.use_adaln_lora:
            shift_self, scale_self, gate_self = (self.adaln_modulation_self_attn(emb_B_T_D) + adaln_lora_B_T_3D).chunk(
                3, dim=-1
            )
            shift_cross, scale_cross, gate_cross = (
                self.adaln_modulation_cross_attn(emb_B_T_D) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
            shift_mlp, scale_mlp, gate_mlp = (self.adaln_modulation_mlp(emb_B_T_D) + adaln_lora_B_T_3D).chunk(3, dim=-1)
        else:
            shift_self, scale_self, gate_self = self.adaln_modulation_self_attn(emb_B_T_D).chunk(3, dim=-1)
            shift_cross, scale_cross, gate_cross = self.adaln_modulation_cross_attn(emb_B_T_D).chunk(3, dim=-1)
            shift_mlp, scale_mlp, gate_mlp = self.adaln_modulation_mlp(emb_B_T_D).chunk(3, dim=-1)

    # Reshape (B, T, D) -> (B, T, 1, 1, D) for broadcasting
    shift_self = rearrange(shift_self, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
    scale_self = rearrange(scale_self, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
    gate_self = rearrange(gate_self, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
    shift_cross = rearrange(shift_cross, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
    scale_cross = rearrange(scale_cross, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
    gate_cross = rearrange(gate_cross, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
    shift_mlp = rearrange(shift_mlp, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
    scale_mlp = rearrange(scale_mlp, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
    gate_mlp = rearrange(gate_mlp, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)

    B, T, H, W, D = x_B_T_H_W_D.shape

    # 1. Self-attention
    normalized_x = self.layer_norm_self_attn(x_B_T_H_W_D) * (1 + scale_self) + shift_self
    flat_x = rearrange(normalized_x, "b t h w d -> b (t h w) d")
    sa_out = _cosmos_self_attention_with_cache(
        self.self_attn,
        flat_x,
        rope_emb_L_1_1_D,
        block_mask,
        external_self_kv,
        cache_tag,
        store_kv,
        cache_start_idx,
    )
    sa_out_5d = rearrange(sa_out, "b (t h w) d -> b t h w d", t=T, h=H, w=W)
    x_B_T_H_W_D = x_B_T_H_W_D + gate_self * sa_out_5d

    # 2. Cross-attention
    normalized_x = self.layer_norm_cross_attn(x_B_T_H_W_D) * (1 + scale_cross) + shift_cross
    flat_x = rearrange(normalized_x, "b t h w d -> b (t h w) d")
    ca_out = _cosmos_cross_attention_with_cache(
        self.cross_attn,
        flat_x,
        crossattn_emb,
        external_cross_kv,
        cache_tag,
        store_kv,
    )
    ca_out_5d = rearrange(ca_out, "b (t h w) d -> b t h w d", t=T, h=H, w=W)
    x_B_T_H_W_D = ca_out_5d * (gate_cross * crossattn_gate_scale) + x_B_T_H_W_D

    # 3. MLP
    normalized_x = self.layer_norm_mlp(x_B_T_H_W_D) * (1 + scale_mlp) + shift_mlp
    mlp_out = self.mlp(normalized_x)
    x_B_T_H_W_D = x_B_T_H_W_D + gate_mlp * mlp_out

    return x_B_T_H_W_D


def _cosmos_dit_forward_causal(
    self,
    x_B_C_T_H_W: torch.Tensor,
    timesteps_B_T: torch.Tensor,
    crossattn_emb: torch.Tensor,
    fps: Optional[torch.Tensor] = None,
    padding_mask: Optional[torch.Tensor] = None,
    video_condition_mask: Optional[torch.Tensor] = None,
    skip_layers: Optional[List[int]] = None,
    feature_indices: Optional[Set[int]] = None,
    return_features_early: bool = False,
    return_logvar: bool = False,
    adaln_lora_scale: float = 1.0,
    crossattn_gate_scale: float = 1.0,
    # --- causal kwargs ---
    cache_tag: str = "pos",
    store_kv: bool = False,
    cur_start_frame: int = 0,
    is_ar: bool = False,
    chunk_size: int = 3,
    total_num_frames: int = 24,
) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """CosmosPredict2DiT forward with causal kwargs for KV cache and block masks.

    Replaces the original ``CosmosPredict2DiT.forward`` via monkey-patch.
    """
    if feature_indices is None:
        feature_indices = set()

    # --- Patch embed ---
    # prepare_embedded_sequence handles padding mask, video condition mask, patch embed
    # but we need to override the RoPE to use time offset
    # We temporarily use the standard path then override RoPE below

    # Add video condition mask channel
    if self.add_video_condition_mask:
        if video_condition_mask is not None:
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, video_condition_mask.type_as(x_B_C_T_H_W)], dim=1)
        else:
            B, C, T, H, W = x_B_C_T_H_W.shape
            zeros_mask = torch.zeros(B, 1, T, H, W, dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device)
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, zeros_mask], dim=1)

    # Add padding mask channel
    if self.concat_padding_mask:
        if padding_mask is not None:
            if padding_mask.ndim == 3:
                padding_mask = padding_mask.unsqueeze(1)
            padding_mask = F.interpolate(
                padding_mask.float(),
                size=x_B_C_T_H_W.shape[-2:],
                mode="nearest",
            ).type_as(x_B_C_T_H_W)
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, padding_mask.unsqueeze(2).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)], dim=1
            )
        else:
            B, C, T, H, W = x_B_C_T_H_W.shape
            pad = torch.zeros(B, 1, 1, H, W, device=x_B_C_T_H_W.device, dtype=x_B_C_T_H_W.dtype)
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, pad.repeat(1, 1, T, 1, 1)], dim=1)

    x_B_T_H_W_D = self.x_embedder(x_B_C_T_H_W)

    # Extra per-block positional embeddings
    if self.extra_per_block_abs_pos_emb:
        extra_pos_emb = self.extra_pos_embedder(x_B_T_H_W_D, fps=fps)
    else:
        extra_pos_emb = None

    # RoPE with time offset
    num_frames = x_B_T_H_W_D.shape[1]

    # Infer start frame from cache if cur_start_frame == 0
    actual_start_frame = cur_start_frame
    if actual_start_frame == 0:
        frame_seqlen = x_B_T_H_W_D.shape[2] * x_B_T_H_W_D.shape[3]
        kv_container = getattr(self, "_external_self_kv_list", None)
        if isinstance(kv_container, list) and len(kv_container) == len(self.blocks):
            for per_block in kv_container:
                if not isinstance(per_block, dict):
                    continue
                tag_entry = per_block.get(cache_tag, None)
                if isinstance(tag_entry, dict):
                    cached_tokens = int(tag_entry.get("len", 0))
                    if cached_tokens > 0:
                        actual_start_frame = cached_tokens // frame_seqlen
                        break

    rope_emb_L_1_1_D = self.pos_embedder(x_B_T_H_W_D, fps=fps, start_frame=actual_start_frame)

    # Timestep embedding
    with torch.amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
        t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

    if adaln_lora_B_T_3D is not None and adaln_lora_scale != 1.0:
        adaln_lora_B_T_3D = adaln_lora_B_T_3D * adaln_lora_scale

    # Cross-attention projection
    if self.use_crossattn_projection and hasattr(self, "crossattn_proj"):
        crossattn_emb = self.crossattn_proj(crossattn_emb)

    # --- Cache / mask setup ---
    B, T_pat, H_pat, W_pat, D = x_B_T_H_W_D.shape
    frame_seqlen = H_pat * W_pat

    # AR mode: allocate external caches
    if num_frames < total_num_frames:
        _create_external_caches_cosmos(
            self,
            x_B_T_H_W_D,
            crossattn_emb,
            num_frames=num_frames,
            total_num_frames=total_num_frames,
            cache_tag=cache_tag,
            cur_start_frame=actual_start_frame,
        )

    # Teacher-forcing mode: build FlexAttention block mask once
    current_block_mask = None
    if not is_ar:
        if getattr(self, "block_mask", None) is None and num_frames == total_num_frames and FLEX_ATTENTION_AVAILABLE:
            self.block_mask = _prepare_blockwise_causal_attn_mask_cosmos(
                x_B_T_H_W_D.device,
                num_frames=num_frames,
                frame_seqlen=frame_seqlen,
                chunk_size=chunk_size,
            )
        current_block_mask = getattr(self, "block_mask", None)

    # --- Block loop ---
    features = []
    for idx, block in enumerate(self.blocks):
        if skip_layers is not None and idx in skip_layers:
            continue

        # Build per-block cache kwargs
        attn_cache_kwargs = {
            "cache_tag": cache_tag,
            "store_kv": store_kv,
        }

        proper_cache_len = None
        if hasattr(self, "_external_self_kv_list") and idx < len(self._external_self_kv_list):
            original_cache = self._external_self_kv_list[idx]
            if cache_tag in original_cache:
                proper_cache_len = actual_start_frame * frame_seqlen
                original_cache[cache_tag]["len"] = proper_cache_len
                stable_cache = {
                    cache_tag: {
                        "k": original_cache[cache_tag]["k"],
                        "v": original_cache[cache_tag]["v"],
                        "len": proper_cache_len,
                    }
                }
                attn_cache_kwargs["external_self_kv"] = stable_cache
            else:
                attn_cache_kwargs["external_self_kv"] = original_cache

        if hasattr(self, "_external_cross_kv_list") and idx < len(self._external_cross_kv_list):
            attn_cache_kwargs["external_cross_kv"] = self._external_cross_kv_list[idx]

        x_B_T_H_W_D = block(
            x_B_T_H_W_D,
            t_embedding_B_T_D,
            crossattn_emb,
            rope_emb_L_1_1_D=rope_emb_L_1_1_D,
            adaln_lora_B_T_3D=adaln_lora_B_T_3D,
            extra_per_block_pos_emb=extra_pos_emb,
            crossattn_gate_scale=crossattn_gate_scale,
            block_mask=current_block_mask,
            attention_cache_kwargs=attn_cache_kwargs,
            cache_start_idx=proper_cache_len,
        )

        if feature_indices and idx in feature_indices:
            features.append(x_B_T_H_W_D.clone())

        if return_features_early and len(features) == len(feature_indices):
            return features

    # --- Final layer + unpatchify ---
    x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
    output = self.unpatchify(x_B_T_H_W_O)

    if len(feature_indices) == 0:
        out = output
    else:
        out = [output, features]

    if return_logvar:
        if not hasattr(self, "logvar_linear"):
            raise RuntimeError(
                "logvar_linear layer is required when return_logvar=True. "
                "Set enable_logvar_linear=True in model config."
            )
        logvar = self.logvar_linear(t_embedding_B_T_D)
        return out, logvar

    return out


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class CausalCosmosPredict2(CausalFastGenNetwork, CosmosPredict2):
    """
    Causal CosmosPredict2 network with inline KV caching for Self-Forcing training.

    MRO: CausalCosmosPredict2 -> CausalFastGenNetwork -> CosmosPredict2 -> FastGenNetwork

    This class:
    - Forces SAC to NONE (KV cache is incompatible with SAC)
    - Monkey-patches the transformer blocks, RoPE, and DiT forward for causal operation
    - Implements clear_caches() for cache management
    - Adds causal kwargs (cache_tag, store_kv, cur_start_frame, is_ar) to forward()
    - Provides AR sample() with chunk-based denoising
    """

    def __init__(
        self,
        chunk_size: int = 3,
        total_num_frames: int = 24,
        delete_cache_on_clear: bool = False,
        use_fsdp_checkpoint: bool = False,
        # All CosmosPredict2 kwargs
        model_channels: int = 2048,
        num_blocks: int = 28,
        num_heads: int = 16,
        max_img_h: int = 240,
        max_img_w: int = 240,
        max_frames: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        patch_spatial: int = 2,
        patch_temporal: int = 1,
        concat_padding_mask: bool = True,
        add_video_condition_mask: bool = True,
        mlp_ratio: float = 4.0,
        crossattn_emb_channels: int = 1024,
        pos_emb_cls: str = "rope3d",
        rope_h_extrapolation_ratio: float = 3.0,
        rope_w_extrapolation_ratio: float = 3.0,
        rope_t_extrapolation_ratio: float = 1.0,
        use_adaln_lora: bool = True,
        adaln_lora_dim: int = 256,
        adaln_lora_scale: float = 1.0,
        crossattn_gate_scale: float = 1.0,
        extra_per_block_abs_pos_emb: bool = False,
        sac_config: Optional[SACConfig] = None,
        enable_logvar_linear: bool = True,
        net_pred_type: str = "flow",
        schedule_type: str = "rf",
        use_crossattn_projection: bool = True,
        crossattn_proj_in_channels: int = 100352,
        text_encoder_model_name: str = "nvidia/Cosmos-Reason1-7B",
        use_wan_fp32_strategy: bool = True,
        fps: float = 24.0,
        is_video2world: bool = False,
        num_conditioning_frames: int = 1,
        **model_kwargs,
    ):
        # Force SAC to NONE — KV cache and SAC are incompatible
        forced_sac = SACConfig(mode=CheckpointMode.NONE)

        # Call CosmosPredict2.__init__ via super() (MRO handles CausalFastGenNetwork first)
        super().__init__(
            chunk_size=chunk_size,
            total_num_frames=total_num_frames,
            model_channels=model_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            max_img_h=max_img_h,
            max_img_w=max_img_w,
            max_frames=max_frames,
            in_channels=in_channels,
            out_channels=out_channels,
            patch_spatial=patch_spatial,
            patch_temporal=patch_temporal,
            concat_padding_mask=concat_padding_mask,
            add_video_condition_mask=add_video_condition_mask,
            mlp_ratio=mlp_ratio,
            crossattn_emb_channels=crossattn_emb_channels,
            pos_emb_cls=pos_emb_cls,
            rope_h_extrapolation_ratio=rope_h_extrapolation_ratio,
            rope_w_extrapolation_ratio=rope_w_extrapolation_ratio,
            rope_t_extrapolation_ratio=rope_t_extrapolation_ratio,
            use_adaln_lora=use_adaln_lora,
            adaln_lora_dim=adaln_lora_dim,
            adaln_lora_scale=adaln_lora_scale,
            crossattn_gate_scale=crossattn_gate_scale,
            extra_per_block_abs_pos_emb=extra_per_block_abs_pos_emb,
            sac_config=forced_sac,
            enable_logvar_linear=enable_logvar_linear,
            net_pred_type=net_pred_type,
            schedule_type=schedule_type,
            use_crossattn_projection=use_crossattn_projection,
            crossattn_proj_in_channels=crossattn_proj_in_channels,
            text_encoder_model_name=text_encoder_model_name,
            use_wan_fp32_strategy=use_wan_fp32_strategy,
            fps=fps,
            is_video2world=is_video2world,
            num_conditioning_frames=num_conditioning_frames,
            **model_kwargs,
        )
        self._delete_cache_on_clear = delete_cache_on_clear
        self._use_fsdp_checkpoint = use_fsdp_checkpoint

        # Apply monkey-patches for causal operation
        self._apply_causal_patches()

    def _apply_causal_patches(self) -> None:
        """Monkey-patch the transformer for causal operation."""
        # Patch RoPE to support time offset
        self.transformer.pos_embedder.forward = types.MethodType(
            _cosmos_rope_forward_with_time_offset, self.transformer.pos_embedder
        )

        # Patch DiT forward to handle causal kwargs
        self.transformer.forward = types.MethodType(_cosmos_dit_forward_causal, self.transformer)

        # Initialize block_mask attribute
        self.transformer.block_mask = None

        # Patch each block's forward for inline cache support
        for block in self.transformer.blocks:
            block.forward = types.MethodType(_cosmos_block_forward_with_cache, block)

    def fully_shard(self, **kwargs):
        """Fully shard with FSDP activation checkpointing for the causal student.

        SAC is disabled (incompatible with KV cache due to wrapping order),
        so we use FSDP-compatible activation checkpointing instead.
        This must be applied AFTER monkey-patching so the checkpoint wrapper
        wraps the KV-cache-aware forward, not the original one.
        """
        if self._use_fsdp_checkpoint:
            apply_fsdp_checkpointing(self.transformer, check_fn=lambda m: isinstance(m, Block))
            logger.info("Applied FSDP activation checkpointing to causal Cosmos transformer blocks")

        super().fully_shard(**kwargs)

    def clear_caches(self) -> None:
        """Clear all internal KV caches."""
        if self._delete_cache_on_clear:
            if hasattr(self.transformer, "_external_self_kv_list"):
                del self.transformer._external_self_kv_list
            if hasattr(self.transformer, "_external_cross_kv_list"):
                del self.transformer._external_cross_kv_list
            torch.cuda.empty_cache()
            return

        if hasattr(self.transformer, "_external_self_kv_list"):
            for kvb in self.transformer._external_self_kv_list:
                if isinstance(kvb, dict):
                    for sub in kvb.values():
                        if isinstance(sub, dict):
                            sub["len"] = 0
                            sub["k"] = torch.zeros_like(sub["k"])
                            sub["v"] = torch.zeros_like(sub["v"])
        if hasattr(self.transformer, "_external_cross_kv_list"):
            for kvb in self.transformer._external_cross_kv_list:
                if isinstance(kvb, dict):
                    for sub in kvb.values():
                        if isinstance(sub, dict):
                            sub["is_init"] = False
                            sub["k"] = torch.zeros_like(sub["k"])
                            sub["v"] = torch.zeros_like(sub["v"])

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Union[torch.Tensor, Dict[str, Any]] = None,
        r: Optional[torch.Tensor] = None,
        return_features_early: bool = False,
        feature_indices: Optional[Set[int]] = None,
        return_logvar: bool = False,
        unpatchify_features: bool = True,
        fwd_pred_type: Optional[str] = None,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        skip_layers: Optional[List[int]] = None,
        # --- causal kwargs ---
        cache_tag: str = "pos",
        cur_start_frame: int = 0,
        store_kv: bool = False,
        is_ar: bool = False,
        **fwd_kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with causal KV cache support.

        Extends CosmosPredict2.forward() with cache_tag, store_kv, cur_start_frame, is_ar.
        """
        if feature_indices is None:
            feature_indices = {}
        if return_features_early and len(feature_indices) == 0:
            return []

        if fps is None:
            fps = torch.full((x_t.shape[0],), self.fps, device=x_t.device)

        if fwd_pred_type is None:
            fwd_pred_type = self.net_pred_type
        else:
            assert fwd_pred_type in NET_PRED_TYPES, f"{fwd_pred_type} is not supported as fwd_pred_type"

        # Handle dict-style condition
        conditioning_latents = None
        condition_mask = None
        if isinstance(condition, dict):
            text_embeds = condition["text_embeds"]
            conditioning_latents = condition.get("conditioning_latents")
            condition_mask = condition.get("condition_mask")
        else:
            text_embeds = condition

        # Video2world: replace input with conditioning latents (chunk-aware)
        model_input = x_t
        conditioning_latents_full = None
        condition_mask_C = None
        chunk_condition_mask = condition_mask  # None for non-V2W, full mask otherwise

        if conditioning_latents is not None and condition_mask is not None:
            B, C, T_chunk, H, W = x_t.shape
            cur_end_frame = cur_start_frame + T_chunk
            T_cond = conditioning_latents.shape[2]

            # Slice condition_mask to current chunk temporal range
            chunk_condition_mask = condition_mask[:, :, cur_start_frame:cur_end_frame]
            condition_mask_C = chunk_condition_mask.expand(-1, C, -1, -1, -1)

            # Map conditioning latents to current chunk frame positions
            conditioning_latents_full = torch.zeros_like(x_t)
            overlap_start = max(cur_start_frame, 0)
            overlap_end = min(cur_end_frame, T_cond)
            if overlap_start < overlap_end:
                local_start = overlap_start - cur_start_frame
                local_end = overlap_end - cur_start_frame
                conditioning_latents_full[:, :, local_start:local_end] = conditioning_latents[
                    :, :, overlap_start:overlap_end
                ]

            model_input = conditioning_latents_full * condition_mask_C + x_t * (1 - condition_mask_C)

        model_outputs = self.transformer(
            x_B_C_T_H_W=model_input,
            timesteps_B_T=t,
            crossattn_emb=text_embeds,
            fps=fps,
            padding_mask=padding_mask,
            video_condition_mask=chunk_condition_mask,
            skip_layers=skip_layers,
            feature_indices=feature_indices,
            return_features_early=return_features_early,
            return_logvar=return_logvar,
            adaln_lora_scale=self.adaln_lora_scale,
            crossattn_gate_scale=self.crossattn_gate_scale,
            # causal kwargs
            cache_tag=cache_tag,
            store_kv=store_kv,
            cur_start_frame=cur_start_frame,
            is_ar=is_ar,
            chunk_size=self.chunk_size,
            total_num_frames=self.total_num_frames,
        )

        # Handle early feature return
        if return_features_early:
            assert len(model_outputs) == len(feature_indices)
            if unpatchify_features:
                model_outputs = [rearrange(f, "B T H W D -> B D T H W") for f in model_outputs]
            else:
                model_outputs = [rearrange(f, "B T H W D -> B (T H W) D") for f in model_outputs]
            return model_outputs

        if return_logvar:
            out, logvar = model_outputs[0], model_outputs[1]
        else:
            out = model_outputs

        # Reshape t for convert_model_output: [B,T] -> [B,1,T] so expand_like
        # produces [B,1,T,1,1] which broadcasts correctly with [B,C,T,H,W]
        t_convert = t.unsqueeze(1) if t.dim() == 2 else t

        if len(feature_indices) == 0:
            assert isinstance(out, torch.Tensor)
            out = self.noise_scheduler.convert_model_output(
                model_input, out, t_convert, src_pred_type=self.net_pred_type, target_pred_type=fwd_pred_type
            )
            if conditioning_latents is not None and condition_mask is not None and fwd_pred_type == "x0":
                out = conditioning_latents_full * condition_mask_C + out * (1 - condition_mask_C)
        else:
            assert isinstance(out, list)
            out[0] = self.noise_scheduler.convert_model_output(
                model_input, out[0], t_convert, src_pred_type=self.net_pred_type, target_pred_type=fwd_pred_type
            )
            if conditioning_latents is not None and condition_mask is not None and fwd_pred_type == "x0":
                out[0] = conditioning_latents_full * condition_mask_C + out[0] * (1 - condition_mask_C)
            if unpatchify_features:
                out[1] = [rearrange(f, "B T H W D -> B D T H W") for f in out[1]]
            else:
                out[1] = [rearrange(f, "B T H W D -> B (T H W) D") for f in out[1]]

        if return_logvar:
            return out, logvar
        return out

    def sample(
        self,
        noise: torch.Tensor,
        condition: Optional[Any] = None,
        neg_condition: Optional[Any] = None,
        guidance_scale: Optional[float] = 5.0,
        num_steps: int = 50,
        shift: float = 5.0,
        context_noise: float = 0,
        fps: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Autoregressive sampling with CFG and UniPC scheduler.

        Follows CausalWan.sample() pattern: chunk-based AR loop with KV cache.
        """
        assert self.schedule_type == "rf", f"{self.schedule_type} is not supported"

        was_training = self.training
        self.eval()
        self.clear_caches()

        x = noise
        batch_size = x.shape[0]
        num_frames = x.shape[2]

        if fps is None:
            fps = torch.full((batch_size,), self.fps, device=x.device)

        # Handle dict-style condition
        cond_embeds = condition
        neg_cond_embeds = neg_condition
        if isinstance(condition, dict):
            cond_embeds = condition.get("text_embeds", condition)
        if isinstance(neg_condition, dict):
            neg_cond_embeds = neg_condition.get("text_embeds", neg_condition)

        if self.sample_scheduler is None:
            self.sample_scheduler = UniPCMultistepScheduler(
                num_train_timesteps=1000,
                prediction_type="flow_prediction",
                use_flow_sigmas=True,
                flow_shift=shift,
            )

        num_chunks = num_frames // self.chunk_size
        remaining_size = num_frames % self.chunk_size

        for i in range(max(1, num_chunks)):
            if num_chunks == 0:
                start, end = 0, remaining_size
            else:
                start = 0 if i == 0 else self.chunk_size * i + remaining_size
                end = self.chunk_size * (i + 1) + remaining_size

            x_next = x[:, :, start:end]

            self.sample_scheduler.config.flow_shift = shift
            self.sample_scheduler.set_timesteps(num_inference_steps=num_steps, device=noise.device)
            timesteps = self.sample_scheduler.timesteps

            for timestep in tqdm(timesteps, total=num_steps, desc=f"Sampling chunk {i}"):
                t = (timestep / self.sample_scheduler.config.num_train_timesteps).expand(batch_size)
                t = self.noise_scheduler.safe_clamp(t, min=self.noise_scheduler.min_t, max=self.noise_scheduler.max_t)
                t = t.to(x.dtype)

                flow_pred = self(
                    x_next,
                    t,
                    condition=cond_embeds,
                    cache_tag="pos",
                    cur_start_frame=start,
                    store_kv=False,
                    is_ar=True,
                    fps=fps,
                )
                if guidance_scale is not None and guidance_scale > 1.0:
                    flow_uncond = self(
                        x_next,
                        t,
                        condition=neg_cond_embeds,
                        cache_tag="neg",
                        cur_start_frame=start,
                        store_kv=False,
                        is_ar=True,
                        fps=fps,
                    )
                    flow_pred = flow_uncond + guidance_scale * (flow_pred - flow_uncond)

                x_next = self.sample_scheduler.step(flow_pred, timestep, x_next, return_dict=False)[0]

            x[:, :, start:end] = x_next

            # Finalize cache: re-run with t=0 (or context_noise) to store KV
            x_cache = x_next
            t_cache = torch.full((batch_size,), 0, device=x.device, dtype=x.dtype)
            if context_noise > 0:
                t_cache = torch.full((batch_size,), context_noise, device=x.device, dtype=x.dtype)
                x_cache = self.noise_scheduler.forward_process(x_next, torch.randn_like(x_next), t_cache)

            _ = self(
                x_cache,
                t_cache,
                condition=cond_embeds,
                cache_tag="pos",
                cur_start_frame=start,
                store_kv=True,
                is_ar=True,
                fps=fps,
            )
            if guidance_scale is not None and guidance_scale > 1.0:
                _ = self(
                    x_cache,
                    t_cache,
                    condition=neg_cond_embeds,
                    cache_tag="neg",
                    cur_start_frame=start,
                    store_kv=True,
                    is_ar=True,
                    fps=fps,
                )

        self.clear_caches()
        self.train(was_training)
        return x
