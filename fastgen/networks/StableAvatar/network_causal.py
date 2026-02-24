# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Causal StableAvatar network wrapper for Self-Forcing framework.

Adds causal attention with KV caching and chunk-by-chunk vocal projector
processing to enable autoregressive generation for the Self-Forcing pipeline.
"""

import types
import math
from typing import Optional, List, Set, Union, Tuple, Dict

from tqdm.auto import tqdm
import torch
import torch.amp as amp

from fastgen.networks.network import CausalFastGenNetwork
from fastgen.networks.StableAvatar.network import StableAvatar
from fastgen.networks.noise_schedule import NET_PRED_TYPES
from fastgen.networks.StableAvatar.models.wan_fantasy_transformer3d_1B import (
    rope_params,
    rope_apply,
    sinusoidal_embedding_1d,
    attention,
)
from fastgen.networks.StableAvatar.models.vocal_projector_fantasy import (
    split_audio_sequence,
    split_tensor_with_padding,
)
import fastgen.utils.logging_utils as logger


# ============================================================================
# Monkey-patch functions for causal inference
# ============================================================================


def _causal_rope_apply(x, grid_sizes, freqs, time_offset=0):
    """RoPE application with temporal offset for autoregressive generation.

    Same as rope_apply but adds time_offset to the temporal frequency indices,
    so that each AR chunk gets position embeddings consistent with its absolute
    temporal position in the full video.
    """
    n, c = x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float32).reshape(seq_len, n, -1, 2)
        )
        # Apply temporal offset to frequency indices
        f_offset = time_offset
        freqs_i = torch.cat([
            freqs[0][f_offset:f_offset + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1).reshape(seq_len, 1, -1)

        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).float()


def _causal_self_attn_forward(self_attn, x, seq_lens, grid_sizes, freqs, dtype,
                               attention_kwargs=None):
    """Causal self-attention with KV cache support.

    Replaces WanSelfAttention.forward to add:
    - KV caching for autoregressive generation
    - Temporal RoPE offset
    - Causal masking within chunks
    """
    attention_kwargs = attention_kwargs or {}
    cache_tag = attention_kwargs.get("cache_tag", "pos")
    store_kv = attention_kwargs.get("store_kv", False)
    time_offset = attention_kwargs.get("time_offset", 0)
    is_ar = attention_kwargs.get("is_ar", False)

    b, s, n, d = *x.shape[:2], self_attn.num_heads, self_attn.head_dim

    q = self_attn.norm_q(self_attn.q(x.to(dtype))).view(b, s, n, d)
    k = self_attn.norm_k(self_attn.k(x.to(dtype))).view(b, s, n, d)
    v = self_attn.v(x.to(dtype)).view(b, s, n, d)

    # Apply RoPE with temporal offset
    q_rope = _causal_rope_apply(q, grid_sizes, freqs, time_offset=time_offset).to(dtype)
    k_rope = _causal_rope_apply(k, grid_sizes, freqs, time_offset=time_offset).to(dtype)

    if is_ar and hasattr(self_attn, "_kv_cache"):
        cache = self_attn._kv_cache
        if cache_tag not in cache:
            cache[cache_tag] = {"k": None, "v": None}
        tag_cache = cache[cache_tag]

        if store_kv:
            # Storing: append current K,V to cache
            if tag_cache["k"] is not None:
                tag_cache["k"] = torch.cat([tag_cache["k"], k_rope.detach()], dim=1)
                tag_cache["v"] = torch.cat([tag_cache["v"], v.detach().to(dtype)], dim=1)
            else:
                tag_cache["k"] = k_rope.detach()
                tag_cache["v"] = v.detach().to(dtype)
        else:
            # Reading: prepend cached K,V
            if tag_cache["k"] is not None:
                k_rope = torch.cat([tag_cache["k"], k_rope], dim=1)
                v_full = torch.cat([tag_cache["v"], v.to(dtype)], dim=1)
            else:
                v_full = v.to(dtype)

            x_out = attention(
                q=q_rope,
                k=k_rope,
                v=v_full,
                k_lens=None,
                window_size=self_attn.window_size,
            )
            x_out = x_out.to(dtype).flatten(2)
            return self_attn.o(x_out)

    # Non-AR path or store_kv: standard attention
    x_out = attention(
        q=q_rope,
        k=k_rope,
        v=v.to(dtype),
        k_lens=seq_lens,
        window_size=self_attn.window_size,
    )
    x_out = x_out.to(dtype).flatten(2)
    return self_attn.o(x_out)


def _causal_cross_attn_forward(cross_attn, x, context, context_lens, dtype,
                                vocal_context=None, vocal_context_lens=None,
                                latents_num_frames=None,
                                attention_kwargs=None):
    """Causal cross-attention with KV cache for text+image (constant), no cache for vocal.

    Replaces WanI2VTalkingCrossAttention.forward to add caching for the
    text+image branches (which are constant across chunks) while keeping
    vocal attention uncached (it changes per chunk).
    """
    attention_kwargs = attention_kwargs or {}
    cache_tag = attention_kwargs.get("cache_tag", "pos")
    store_kv = attention_kwargs.get("store_kv", False)
    is_ar = attention_kwargs.get("is_ar", False)

    context_img = context[:, :257]
    context_text = context[:, 257:]
    b, n, d = x.size(0), cross_attn.num_heads, cross_attn.head_dim

    q = cross_attn.norm_q(cross_attn.q(x.to(dtype))).view(b, -1, n, d)

    # Text+Image K,V (constant, cacheable)
    if is_ar and hasattr(cross_attn, "_cross_cache"):
        cache = cross_attn._cross_cache
        if cache_tag not in cache:
            cache[cache_tag] = {"is_init": False}
        tag_cache = cache[cache_tag]

        if not tag_cache["is_init"]:
            k_text = cross_attn.norm_k(cross_attn.k(context_text.to(dtype))).view(b, -1, n, d)
            v_text = cross_attn.v(context_text.to(dtype)).view(b, -1, n, d)
            k_img = cross_attn.norm_k_img(cross_attn.k_img(context_img.to(dtype))).view(b, -1, n, d)
            v_img = cross_attn.v_img(context_img.to(dtype)).view(b, -1, n, d)
            tag_cache["k_text"] = k_text.detach()
            tag_cache["v_text"] = v_text.detach()
            tag_cache["k_img"] = k_img.detach()
            tag_cache["v_img"] = v_img.detach()
            tag_cache["is_init"] = True
        else:
            k_text = tag_cache["k_text"]
            v_text = tag_cache["v_text"]
            k_img = tag_cache["k_img"]
            v_img = tag_cache["v_img"]
    else:
        k_text = cross_attn.norm_k(cross_attn.k(context_text.to(dtype))).view(b, -1, n, d)
        v_text = cross_attn.v(context_text.to(dtype)).view(b, -1, n, d)
        k_img = cross_attn.norm_k_img(cross_attn.k_img(context_img.to(dtype))).view(b, -1, n, d)
        v_img = cross_attn.v_img(context_img.to(dtype)).view(b, -1, n, d)

    # Image attention
    img_x = attention(q.to(dtype), k_img.to(dtype), v_img.to(dtype), k_lens=None)
    img_x = img_x.to(dtype)

    # Text attention
    text_x = attention(q.to(dtype), k_text.to(dtype), v_text.to(dtype), k_lens=context_lens)
    text_x = text_x.to(dtype)

    # Vocal attention (NOT cached — changes per chunk)
    if latents_num_frames is None:
        latents_num_frames = 21

    if vocal_context is not None and len(vocal_context.shape) == 4:
        vocal_q = q.view(b * latents_num_frames, -1, n, d)
        vocal_ip_key = cross_attn.k_vocal(vocal_context).view(b * latents_num_frames, -1, n, d)
        vocal_ip_value = cross_attn.v_vocal(vocal_context).view(b * latents_num_frames, -1, n, d)
        vocal_x = attention(
            vocal_q.to(dtype), vocal_ip_key.to(dtype), vocal_ip_value.to(dtype),
            k_lens=vocal_context_lens,
        )
        vocal_x = vocal_x.view(b, q.size(1), n, d).flatten(2)
    elif vocal_context is not None:
        vocal_ip_key = cross_attn.k_vocal(vocal_context).view(b, -1, n, d)
        vocal_ip_value = cross_attn.v_vocal(vocal_context).view(b, -1, n, d)
        vocal_x = attention(
            q.to(dtype), vocal_ip_key.to(dtype), vocal_ip_value.to(dtype), k_lens=None,
        )
        vocal_x = vocal_x.flatten(2)
    else:
        vocal_x = 0

    x = text_x.flatten(2) + img_x.flatten(2) + vocal_x
    x = cross_attn.o(x)
    return x


def _causal_block_forward(block, x, e, seq_lens, grid_sizes, freqs, context, context_lens,
                           dtype=torch.float32, vocal_context=None, vocal_context_lens=None,
                           latents_num_frames=None, attention_kwargs=None, **kwargs):
    """Causal block forward that passes attention_kwargs to sub-modules."""
    e_chunks = (block.modulation + e).chunk(6, dim=1)

    # Self-attention with causal KV cache
    temp_x = block.norm1(x) * (1 + e_chunks[1]) + e_chunks[0]
    temp_x = temp_x.to(dtype)
    y = _causal_self_attn_forward(
        block.self_attn, temp_x, seq_lens, grid_sizes, freqs, dtype,
        attention_kwargs=attention_kwargs,
    )
    x = x + y * e_chunks[2]

    # Cross-attention with cached text+image, uncached vocal
    cross_out = _causal_cross_attn_forward(
        block.cross_attn, block.norm3(x), context, context_lens, dtype,
        vocal_context=vocal_context, vocal_context_lens=vocal_context_lens,
        latents_num_frames=latents_num_frames,
        attention_kwargs=attention_kwargs,
    )
    x = x + cross_out

    # FFN
    temp_x = block.norm2(x) * (1 + e_chunks[4]) + e_chunks[3]
    temp_x = temp_x.to(dtype)
    y = block.ffn(temp_x)
    x = x + y * e_chunks[5]
    return x


def _causal_transformer_forward(
    transformer, x, t, context, seq_len, clip_fea=None, y=None,
    vocal_embeddings=None, video_sample_n_frames=81,
    attention_kwargs=None,
):
    """Causal transformer forward that passes attention_kwargs through to blocks.

    Replaces WanTransformer3DFantasyModel.forward to support:
    - Temporal RoPE offset via attention_kwargs
    - KV cache pass-through to blocks
    - Per-chunk vocal context (computed externally by CausalStableAvatar)
    """
    attn_kwargs = attention_kwargs or {}
    time_offset = attn_kwargs.get("time_offset", 0)
    vocal_context_override = attn_kwargs.get("vocal_context", None)
    vocal_context_lens_override = attn_kwargs.get("vocal_context_lens", None)

    if transformer.model_type == 'i2v':
        assert clip_fea is not None and y is not None

    device = transformer.patch_embedding.weight.device
    dtype = x.dtype
    if transformer.freqs.device != device and torch.device(type="meta") != device:
        transformer.freqs = transformer.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # Patch embedding
    x = [transformer.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x
    ])

    # Time embeddings
    with amp.autocast('cuda', dtype=torch.float32):
        e = transformer.time_embedding(sinusoidal_embedding_1d(transformer.freq_dim, t).float())
        e0 = transformer.time_projection(e).unflatten(1, (6, transformer.dim))
        e0 = e0.to(dtype)
        e = e.to(dtype)

    # Context embeddings
    context_lens = None
    context = transformer.text_embedding(
        torch.stack([
            torch.cat([u, u.new_zeros(transformer.text_len - u.size(0), u.size(1))])
            for u in context
        ])
    )
    context_clip = transformer.img_emb(clip_fea)
    context = torch.concat([context_clip, context], dim=1)

    # Vocal context: use override if provided (chunk-based), else compute normally
    if vocal_context_override is not None:
        vocal_context = vocal_context_override
        vocal_context_lens = vocal_context_lens_override
    elif vocal_embeddings is not None:
        if vocal_embeddings.size()[0] > 1:
            vocal_embeddings_single = vocal_embeddings[-1:]
            vocal_context, vocal_context_lens = transformer.vocal_projector(
                vocal_embeddings=vocal_embeddings_single,
                video_sample_n_frames=video_sample_n_frames,
                latents=x[-1:], e0=e0[-1:], e=e[-1:],
            )
            vocal_context = torch.cat([torch.zeros_like(vocal_context), vocal_context, vocal_context])
        else:
            vocal_context, vocal_context_lens = transformer.vocal_projector(
                vocal_embeddings=vocal_embeddings,
                video_sample_n_frames=video_sample_n_frames,
                latents=x, e0=e0, e=e,
            )
    else:
        vocal_context = None
        vocal_context_lens = None

    # Add time_offset to attention_kwargs for RoPE
    block_attn_kwargs = dict(attn_kwargs)
    block_attn_kwargs["time_offset"] = time_offset

    # Transformer blocks
    frames_per_batch = (video_sample_n_frames - 1) // 4 + 1
    for block in transformer.blocks:
        x = _causal_block_forward(
            block, x,
            e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes,
            freqs=transformer.freqs, context=context, context_lens=context_lens,
            dtype=dtype, vocal_context=vocal_context,
            vocal_context_lens=vocal_context_lens,
            latents_num_frames=frames_per_batch,
            attention_kwargs=block_attn_kwargs,
        )

    # Head & unpatchify
    x = transformer.head(x, e)
    x = transformer.unpatchify(x, grid_sizes)
    x = torch.stack(x)
    return x


# ============================================================================
# CausalStableAvatar class
# ============================================================================


class CausalStableAvatar(CausalFastGenNetwork, StableAvatar):
    """Causal StableAvatar student model for Self-Forcing distillation.

    Adds causal attention + KV caching + vocal projector splitting to enable
    autoregressive chunk-by-chunk generation with the Self-Forcing pipeline.
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
        chunk_size: int = 3,
        total_num_frames: int = 21,
        use_fsdp_checkpoint: bool = True,
        **model_kwargs,
    ):
        super().__init__(
            checkpoint_path=checkpoint_path,
            dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            net_pred_type=net_pred_type,
            schedule_type=schedule_type,
            video_sample_n_frames=video_sample_n_frames,
            load_pretrained=load_pretrained,
            chunk_size=chunk_size,
            total_num_frames=total_num_frames,
            **model_kwargs,
        )

        # Initialize KV caches on each block
        self._init_kv_caches()

        # Precomputed vocal splits (set during rollout)
        self._precomputed_vocal_splits = None
        self._precomputed_vocal_context_lens = None

    def _init_kv_caches(self):
        """Initialize empty KV cache dicts on each attention sub-module."""
        for block in self.transformer.blocks:
            block.self_attn._kv_cache = {}
            block.cross_attn._cross_cache = {}

    def clear_caches(self) -> None:
        """Clear all KV caches and precomputed vocal splits."""
        for block in self.transformer.blocks:
            if hasattr(block.self_attn, "_kv_cache"):
                block.self_attn._kv_cache = {}
            if hasattr(block.cross_attn, "_cross_cache"):
                block.cross_attn._cross_cache = {}
        self._precomputed_vocal_splits = None
        self._precomputed_vocal_context_lens = None
        torch.cuda.empty_cache()

    def precompute_audio_splits(self, vocal_embeddings, video_sample_n_frames=81):
        """Run Stage A of the vocal projector once and cache the split audio.

        This splits the audio embeddings into per-latent-frame chunks using
        the vocal projector's Stage A (proj_model + split_audio_sequence).
        The result is cached for chunk-by-chunk processing during rollout.

        Args:
            vocal_embeddings: [B, T_audio, 768] raw audio embeddings.
            video_sample_n_frames: Number of pixel-space video frames.
        """
        with torch.no_grad():
            proj_model = self.transformer.vocal_projector.proj_model
            vocal_proj_feature = proj_model(vocal_embeddings)
            pos_idx_ranges = split_audio_sequence(
                vocal_proj_feature.size(1), num_frames=video_sample_n_frames,
            )
            vocal_proj_split, vocal_context_lens = split_tensor_with_padding(
                vocal_proj_feature, pos_idx_ranges, expand_length=4,
            )
            # vocal_proj_split: [B, num_latent_frames, L_pad, C]
            self._precomputed_vocal_splits = vocal_proj_split
            self._precomputed_vocal_context_lens = vocal_context_lens

    def forward_vocal_chunk(self, latents_chunk, e0, e, lf_start, lf_end):
        """Run Stage B of the vocal projector for a specific chunk of latent frames.

        Args:
            latents_chunk: [B, L_chunk, D] patchified latent tokens for this chunk.
            e0: [B, 6, D] time embeddings (projected).
            e: [B, D] time embeddings.
            lf_start: Start latent frame index.
            lf_end: End latent frame index.

        Returns:
            vocal_context: [B, n_lf, L_pad, 1536] vocal context for this chunk.
            vocal_context_lens: [n_lf] context lengths.
        """
        if self._precomputed_vocal_splits is None:
            return None, None

        # Slice the precomputed splits for this chunk
        vocal_chunk = self._precomputed_vocal_splits[:, lf_start:lf_end]  # [B, n_lf, L_pad, C]
        vocal_context_lens = self._precomputed_vocal_context_lens

        n_lf = vocal_chunk.size(1)

        # Run Stage B: VocalAttentionBlocks + Final_Head
        from einops import rearrange
        vocal_flat = rearrange(vocal_chunk, "b f n c -> b (f n) c")

        for block in self.transformer.vocal_projector.blocks:
            vocal_flat = block(
                x=vocal_flat, e=e0, context=latents_chunk,
                q_lens=vocal_context_lens, latents_num_frames=n_lf,
            )

        vocal_out = self.transformer.vocal_projector.final_head(vocal_flat, e)
        vocal_out = rearrange(vocal_out, "b (f n) c -> b f n c", f=n_lf)
        return vocal_out, vocal_context_lens

    def _compute_timestep_inputs(
        self,
        timestep: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute timestep input for causal StableAvatar.

        Always expands timestep to [B, T] for causal processing.
        """
        timestep = self.noise_scheduler.rescale_t(timestep)
        if timestep.ndim == 1:
            timestep = timestep.view(-1, 1)
        if mask is not None:
            p_t = self.transformer.patch_size[0]
            timestep = mask[:, ::p_t, 0, 0] * timestep
        return timestep

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
        cache_tag: str = "pos",
        cur_start_frame: int = 0,
        store_kv: bool = False,
        is_ar: bool = False,
        **fwd_kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Causal forward pass with chunk-aware audio.

        Args:
            x_t: Noisy chunk latents [B, C, T_chunk, H, W].
            t: Timestep [B] or [B, T].
            condition: Dict with text_embeds, first_frame_cond, vocal_embeddings, clip_fea.
            cache_tag: KV cache tag ("pos" or "neg").
            cur_start_frame: Current start frame index in latent space.
            store_kv: Whether to store KV cache after this forward.
            is_ar: Whether this is autoregressive mode.
        """
        assert isinstance(condition, dict), "condition must be a dict"
        assert "text_embeds" in condition
        assert "first_frame_cond" in condition

        if feature_indices is None:
            feature_indices = {}
        if return_features_early and len(feature_indices) == 0:
            return []

        if fwd_pred_type is None:
            fwd_pred_type = self.net_pred_type
        else:
            assert fwd_pred_type in NET_PRED_TYPES

        text_embeds = condition["text_embeds"]
        first_frame_cond = condition["first_frame_cond"]
        vocal_embeddings = condition.get("vocal_embeddings")
        clip_fea = condition.get("clip_fea")

        # Replace first frame with clean conditioning if we're at the start
        bsz, C, T_chunk, H, W = x_t.shape
        latent_model_input = x_t
        timestep_mask = None

        if cur_start_frame == 0:
            latent_model_input, first_frame_mask = self._replace_first_frame(
                first_frame_cond, x_t, return_mask=True,
            )
            timestep_mask = first_frame_mask[:, 0]

        timestep = self._compute_timestep_inputs(t, timestep_mask)

        # Prepare inputs
        x_list = [latent_model_input[i] for i in range(bsz)]
        if isinstance(text_embeds, torch.Tensor):
            context_list = [text_embeds[i] for i in range(text_embeds.shape[0])]
        else:
            context_list = list(text_embeds)

        y_list = [first_frame_cond[i] for i in range(first_frame_cond.shape[0])]

        # Compute sequence length for this chunk
        p_t, p_h, p_w = self.transformer.patch_size
        f_patches = T_chunk // p_t
        h_patches = H // p_h
        w_patches = W // p_w
        seq_len = f_patches * h_patches * w_patches * 2  # x2 for y concat

        # Compute temporal RoPE offset (in patch frames)
        time_offset = cur_start_frame // p_t

        # Build attention_kwargs
        attn_kwargs = {
            "cache_tag": cache_tag,
            "store_kv": store_kv,
            "is_ar": is_ar,
            "time_offset": time_offset,
        }

        # Compute chunk-specific vocal context if precomputed
        if self._precomputed_vocal_splits is not None and is_ar:
            # Get latent frame range for this chunk
            lf_start = cur_start_frame // p_t
            lf_end = (cur_start_frame + T_chunk) // p_t
            # We need the patchified latents for the vocal projector Stage B;
            # for simplicity, pass vocal_embeddings=None and handle vocal in attn_kwargs
            attn_kwargs["vocal_context"] = None  # will be computed in transformer forward
            attn_kwargs["vocal_context_lens"] = None
            # Pass vocal_embeddings=None to skip the transformer's internal vocal computation
            vocal_embeddings_for_fwd = None
        else:
            vocal_embeddings_for_fwd = vocal_embeddings

        # Call causal transformer forward
        out = _causal_transformer_forward(
            self.transformer,
            x=x_list, t=timestep, context=context_list,
            seq_len=seq_len, clip_fea=clip_fea, y=y_list,
            vocal_embeddings=vocal_embeddings_for_fwd,
            video_sample_n_frames=self.video_sample_n_frames,
            attention_kwargs=attn_kwargs,
        )

        # Convert model output
        out = self.noise_scheduler.convert_model_output(
            x_t, out, t, src_pred_type=self.net_pred_type, target_pred_type=fwd_pred_type,
        )

        # Replace first frame in output
        if cur_start_frame == 0:
            out = self._replace_first_frame(first_frame_cond, out)

        if len(feature_indices) > 0:
            return [out, []]

        return out

    def sample(
        self,
        noise: torch.FloatTensor,
        condition: Optional[Dict[str, torch.Tensor]] = None,
        neg_condition: Optional[Dict[str, torch.Tensor]] = None,
        guidance_scale: Optional[float] = 5.0,
        sample_steps: Optional[int] = 50,
        shift: float = 5.0,
        context_noise: float = 0,
        **kwargs,
    ) -> torch.Tensor:
        """Autoregressive teacher sampling with KV cache.

        Follows the pattern from CausalWanI2V.sample().
        """
        assert self.schedule_type == "rf"
        first_frame_cond = None
        if isinstance(condition, dict) and "first_frame_cond" in condition:
            first_frame_cond = condition["first_frame_cond"]

        was_training = self.training
        self.eval()
        self.clear_caches()

        x = noise
        batch_size = x.shape[0]
        num_frames = x.shape[2]
        num_chunks = num_frames // self.chunk_size
        remaining_size = num_frames % self.chunk_size
        time_rescale_factor = self.unipc_scheduler.config.num_train_timesteps

        for i in range(max(1, num_chunks)):
            if num_chunks == 0:
                start, end = 0, remaining_size
            else:
                start = 0 if i == 0 else self.chunk_size * i + remaining_size
                end = self.chunk_size * (i + 1) + remaining_size

            x_next = x[:, :, start:end]
            self.unipc_scheduler.config.flow_shift = shift
            self.unipc_scheduler.set_timesteps(num_inference_steps=sample_steps, device=noise.device)
            timesteps = self.unipc_scheduler.timesteps

            for timestep in tqdm(timesteps, total=sample_steps - 1):
                t = (timestep / time_rescale_factor).expand(batch_size)
                flow_pred = self(
                    x_next, t, condition,
                    cache_tag="pos", cur_start_frame=start, store_kv=False, is_ar=True,
                )
                if guidance_scale is not None:
                    flow_uncond = self(
                        x_next, t, neg_condition,
                        cache_tag="neg", cur_start_frame=start, store_kv=False, is_ar=True,
                    )
                    flow_pred = flow_uncond + guidance_scale * (flow_pred - flow_uncond)

                if first_frame_cond is not None and start == 0:
                    flow_pred = flow_pred.clone()
                    flow_pred[:, :, 0] = 0.0

                x_next = self.unipc_scheduler.step(flow_pred, timestep, x_next, return_dict=False)[0]

                if first_frame_cond is not None and start == 0:
                    x_next = x_next.clone()
                    x_next[:, :, 0] = first_frame_cond[:, :, 0]

            x[:, :, start:end] = x_next

            # Update KV cache
            x_cache = x_next
            t_cache = torch.full((batch_size,), 0, device=x.device, dtype=x.dtype)
            if context_noise > 0:
                t_cache = torch.full((batch_size,), context_noise, device=x.device, dtype=x.dtype)
                x_cache = self.noise_scheduler.forward_process(x_next, torch.randn_like(x_next), t_cache)

            _ = self(x_cache, t_cache, condition,
                     cache_tag="pos", cur_start_frame=start, store_kv=True, is_ar=True)
            if guidance_scale is not None:
                _ = self(x_cache, t_cache, neg_condition,
                         cache_tag="neg", cur_start_frame=start, store_kv=True, is_ar=True)

        self.clear_caches()
        self.train(was_training)

        if first_frame_cond is not None:
            self._replace_first_frame(condition["first_frame_cond"], x)
        return x
