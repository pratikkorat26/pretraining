"""
ReadableLM Modeling
===================

A decoder-only transformer (OLMo3-style) built for **readability first**.

File organisation (top → bottom):
    1. Utility: repeat_kv
    2. RMSNorm
    3. RotaryEmbedding + apply_rotary_pos_emb
    4. ReadableAttention  (eager matmul → softmax, GQA via repeat_kv)
    5. ReadableMLP         (SwiGLU: gate * up → down)
    6. ReadableDecoderLayer (pre-norm residual)
    7. ReadableLMModel      (base model, returns BaseModelOutputWithPast)
    8. ReadableLMForCausalLM (causal LM head, returns CausalLMOutputWithPast)

Design choices kept simple on purpose:
    - Eager attention only (no FlashAttn / SDPA).
    - One causal-mask helper lives in ReadableLMModel.
    - KV cache is a plain tuple-of-tuples (HF legacy format).

TODO (omitted for simplicity):
    - FlashAttention / SDPA / FlexAttn backends
    - Sliding-window or chunked attention masks
    - Dynamic NTK-aware / YaRN RoPE scaling
    - Gradient checkpointing
    - Tensor-parallel sharding annotations
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from configuration_readable_lm import ReadableLMConfig


def _get_cache_length(past_key_values) -> int:
    """
    Extract the cached sequence length from either a DynamicCache object
    (HF v4.50+) or a legacy tuple-of-tuples KV cache.

    Returns 0 if past_key_values is None or empty.
    """
    if past_key_values is None:
        return 0
    # DynamicCache (HF v4.50+) has get_seq_length()
    if hasattr(past_key_values, "get_seq_length"):
        return past_key_values.get_seq_length()
    # Legacy tuple-of-tuples: ((key, value), ...) per layer
    if isinstance(past_key_values, (tuple, list)) and len(past_key_values) > 0:
        return past_key_values[0][0].shape[2]
    return 0


def _get_layer_cache(past_key_values, layer_idx: int):
    """
    Extract (key, value) for a specific layer from either DynamicCache
    or legacy tuple-of-tuples format.

    Returns None if past_key_values is None or the layer has no cache.
    """
    if past_key_values is None:
        return None
    # DynamicCache stores caches in .key_cache / .value_cache lists
    if hasattr(past_key_values, "key_cache"):
        if layer_idx < len(past_key_values.key_cache):
            return (past_key_values.key_cache[layer_idx],
                    past_key_values.value_cache[layer_idx])
        return None
    # Legacy format
    if isinstance(past_key_values, (tuple, list)) and layer_idx < len(past_key_values):
        return past_key_values[layer_idx]
    return None


# ---------------------------------------------------------------------------
# 1. Utility: repeat_kv
# ---------------------------------------------------------------------------
def repeat_kv(
    hidden_states: torch.Tensor,
    num_repeats: int,
) -> torch.Tensor:
    """
    Repeat key/value heads so they match the number of query heads (GQA).

    When num_key_value_heads < num_attention_heads, each KV head is shared
    across `num_repeats` query heads.  This function expands the KV tensor
    along the head dimension so that standard batched matmul works.

    Args:
        hidden_states: (batch, num_kv_heads, seq_len, head_dim)
        num_repeats:   num_attention_heads // num_key_value_heads

    Returns:
        Tensor of shape (batch, num_attention_heads, seq_len, head_dim)
    """
    if num_repeats == 1:
        return hidden_states

    batch_size, num_kv_heads, seq_len, head_dim = hidden_states.shape
    # Insert a repeat dimension and expand, then flatten back.
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch_size, num_kv_heads, num_repeats, seq_len, head_dim
    )
    return hidden_states.reshape(batch_size, num_kv_heads * num_repeats, seq_len, head_dim)


# ---------------------------------------------------------------------------
# 2. RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalisation (Zhang & Sennrich, 2019).

    Unlike LayerNorm, RMSNorm does not subtract the mean — it only
    divides by the root-mean-square of the activations, then scales
    by a learnable weight vector.

    Args (init):
        hidden_size: Dimensionality of the input.
        epsilon:     Small constant for numerical stability.

    Forward input:  (*, hidden_size)
    Forward output: (*, hidden_size)  — same shape, normalised.
    """

    def __init__(self, hidden_size: int, epsilon: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.epsilon = epsilon

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        # Compute in float32 for numerical stability
        variance = hidden_states.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.epsilon)
        return (self.weight * hidden_states).to(input_dtype)


# ---------------------------------------------------------------------------
# 3. Rotary Position Embeddings (RoPE)
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """
    Precomputes and caches cos/sin tables for Rotary Position Embeddings.

    RoPE encodes absolute position information by rotating pairs of
    dimensions in query/key vectors.  The rotation angle for dimension
    pair *i* at position *p* is  p / (theta^(2i/d)).

    Args (init):
        head_dim:  Dimensionality of each attention head.
        max_position_embeddings: Maximum sequence length to precompute.
        rope_theta: Base frequency (default 10 000).

    Forward input:  position_ids — (batch, seq_len) of integer positions.
    Forward output: (cos, sin) each of shape (batch, 1, seq_len, head_dim).
    """

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int = 2048,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta

        # Inverse frequencies: theta^(-2i/d) for i = 0, 1, ..., d/2-1
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        # Register as buffer (not a parameter — no gradient).
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(
        self,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cos and sin embeddings for the given positions.

        Args:
            position_ids: (batch, seq_len) integer position indices.

        Returns:
            cos: (batch, 1, seq_len, head_dim) — cosine part.
            sin: (batch, 1, seq_len, head_dim) — sine part.
        """
        # inv_freq shape: (head_dim/2,)
        # position_ids shape: (batch, seq_len)
        inv_freq = self.inv_freq[None, :, None].float()          # (1, head_dim/2, 1)
        position_ids_float = position_ids[:, None, :].float()    # (batch, 1, seq_len)

        # Outer product: (batch, head_dim/2, seq_len)
        freqs = inv_freq * position_ids_float
        # Transpose to (batch, seq_len, head_dim/2)
        freqs = freqs.transpose(1, 2)

        # Duplicate to cover full head_dim: [θ0, θ1, ..., θ0, θ1, ...]
        emb = torch.cat([freqs, freqs], dim=-1)  # (batch, seq_len, head_dim)

        cos = emb.cos().unsqueeze(1)  # (batch, 1, seq_len, head_dim)
        sin = emb.sin().unsqueeze(1)  # (batch, 1, seq_len, head_dim)
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Swap and negate the two halves of the last dimension.

    For input [..., d], splits into [..., d//2] chunks and returns
    [-x2, x1].  This is the standard helper for applying the RoPE
    rotation matrix.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Rotary Position Embeddings to query and key tensors.

    Uses the standard formula:
        q_rotated = q * cos + rotate_half(q) * sin

    Args:
        query: (batch, num_heads, seq_len, head_dim)
        key:   (batch, num_kv_heads, seq_len, head_dim)
        cos:   (batch, 1, seq_len, head_dim) — broadcastable over heads.
        sin:   (batch, 1, seq_len, head_dim)

    Returns:
        (query_rotated, key_rotated) with same shapes as inputs.
    """
    query_rotated = (query * cos) + (_rotate_half(query) * sin)
    key_rotated = (key * cos) + (_rotate_half(key) * sin)
    return query_rotated, key_rotated


# ---------------------------------------------------------------------------
# 4. ReadableAttention (eager, with GQA + KV cache)
# ---------------------------------------------------------------------------

class ReadableAttention(nn.Module):
    """
    Multi-head attention with Grouped-Query Attention (GQA) support.

    Supports two backends controlled by config.attn_implementation:
      - "sdpa":  PyTorch's F.scaled_dot_product_attention, which auto-dispatches
                 to FlashAttention-2 or memory-efficient attention on supported GPUs.
      - "eager": Explicit matmul → scale → mask → softmax → matmul (readable, debuggable).

    Both paths share the same projection, RoPE, and KV-cache logic.

    Forward inputs:
        hidden_states:  (batch, seq_len, hidden_size)
        attention_mask: (batch, 1, seq_len, kv_len) — additive mask (0 or -inf).
        position_ids:   (batch, seq_len)
        past_key_value: Optional tuple (key, value) from previous steps.
        use_cache:      Whether to return updated KV cache.

    Forward outputs:
        (attention_output, present_key_value | None)
        attention_output shape: (batch, seq_len, hidden_size)
    """

    def __init__(self, config: ReadableLMConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_kv_groups = config.num_attention_heads // config.num_key_value_heads
        self.attn_implementation = config.attn_implementation

        assert self.head_dim * config.num_attention_heads == config.hidden_size, (
            "hidden_size must be divisible by num_attention_heads"
        )

        # Projections
        self.query_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.key_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.value_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.output_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        # RoPE
        self.rotary_emb = RotaryEmbedding(
            head_dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
        )

        self.scaling = 1.0 / math.sqrt(self.head_dim)

    def _eager_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Explicit matmul → scale → mask → softmax → matmul attention.

        All inputs are (batch, num_heads, seq_len, head_dim) after GQA expansion.
        Returns (batch, num_heads, q_len, head_dim).
        """
        attention_scores = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scaling
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_weights = F.softmax(attention_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        return torch.matmul(attention_weights, value_states)

    def _sdpa_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        PyTorch SDPA — auto-dispatches to FlashAttention-2 or memory-efficient
        attention based on GPU capability, dtype, and head_dim.

        All inputs are (batch, num_heads, seq_len, head_dim) after GQA expansion.
        Returns (batch, num_heads, q_len, head_dim).
        """
        # SDPA can use is_causal=True for faster path when no padding mask is needed
        # (all mask values are 0, meaning no padding tokens are masked).
        # When a padding mask is present (some values are -inf), we must pass it explicitly.
        is_causal = False
        sdpa_mask = attention_mask
        if attention_mask is not None:
            has_padding = torch.isinf(attention_mask).any()
            if not has_padding:
                # Pure causal mask with no padding — let SDPA handle it natively
                is_causal = (query_states.shape[2] > 1)  # causal only for prefill, not single-token decode
                sdpa_mask = None

        return F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=sdpa_mask,
            is_causal=is_causal,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape

        # --- Project Q, K, V ---
        query_states = self.query_proj(hidden_states)
        key_states = self.key_proj(hidden_states)
        value_states = self.value_proj(hidden_states)

        # Reshape to (batch, heads, seq_len, head_dim)
        query_states = query_states.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # --- Apply RoPE ---
        cos, sin = self.rotary_emb(position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # --- KV cache: concatenate past keys/values ---
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)

        present_key_value = (key_states, value_states) if use_cache else None

        # --- GQA: expand KV heads to match query heads ---
        key_states = repeat_kv(key_states, self.num_kv_groups)
        value_states = repeat_kv(value_states, self.num_kv_groups)

        # --- Dispatch to attention backend ---
        if self.attn_implementation == "sdpa":
            attention_output = self._sdpa_attention(query_states, key_states, value_states, attention_mask)
        else:
            attention_output = self._eager_attention(query_states, key_states, value_states, attention_mask)
        # attention_output shape: (batch, num_heads, q_len, head_dim)

        # --- Reshape back to (batch, seq_len, hidden_size) ---
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.reshape(batch_size, seq_len, self.hidden_size)
        attention_output = self.output_proj(attention_output)

        return attention_output, present_key_value


# ---------------------------------------------------------------------------
# 5. ReadableMLP (SwiGLU — gate * up → down)
# ---------------------------------------------------------------------------

class ReadableMLP(nn.Module):
    """
    Gated MLP with SwiGLU activation (Shazeer, 2020).

    Computes:  down_proj( silu(gate_proj(x)) * up_proj(x) )

    This is the same "gate * up" pattern used in LLaMA and OLMo.

    Args (init):
        config: ReadableLMConfig with hidden_size and intermediate_size.

    Forward input:  (batch, seq_len, hidden_size)
    Forward output: (batch, seq_len, hidden_size)
    """

    def __init__(self, config: ReadableLMConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        return self.down_proj(gate * up)


# ---------------------------------------------------------------------------
# 6. ReadableDecoderLayer (pre-norm residual block)
# ---------------------------------------------------------------------------

class ReadableDecoderLayer(nn.Module):
    """
    One transformer decoder layer with pre-norm residuals (OLMo3 ordering).

    Structure:
        residual = x
        x = attention(input_norm(x))
        x = residual + x
        residual = x
        x = mlp(post_attention_norm(x))
        x = residual + x

    Args (init):
        config: ReadableLMConfig.

    Forward inputs:
        hidden_states, attention_mask, position_ids, past_key_value, use_cache
    Forward output:
        (hidden_states, present_key_value)
    """

    def __init__(self, config: ReadableLMConfig):
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size, epsilon=config.rms_norm_epsilon)
        self.self_attention = ReadableAttention(config)
        self.post_attention_norm = RMSNorm(config.hidden_size, epsilon=config.rms_norm_epsilon)
        self.mlp = ReadableMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # --- Self-attention block ---
        residual = hidden_states
        hidden_states = self.input_norm(hidden_states)
        attention_output, present_key_value = self.self_attention(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + attention_output

        # --- MLP block ---
        residual = hidden_states
        hidden_states = self.post_attention_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, present_key_value


# ---------------------------------------------------------------------------
# 7. ReadableLMModel (base model — embeddings → layers → final norm)
# ---------------------------------------------------------------------------
class ReadableLMModel(PreTrainedModel):
    """
    Base decoder-only transformer model (no language-modelling head).

    Takes input_ids → token embeddings → N decoder layers → final RMSNorm,
    and returns hidden states with optional KV cache.

    This class owns the causal mask construction so that individual
    attention layers don't need to worry about it.

    Forward inputs:
        input_ids:        (batch, seq_len)
        attention_mask:   (batch, seq_len) — 1 for real tokens, 0 for padding.
        position_ids:     (batch, seq_len) — auto-computed if not provided.
        past_key_values:  Tuple of (key, value) for each layer, or None.
        use_cache:        Whether to return new KV cache entries.

    Forward output:
        BaseModelOutputWithPast(last_hidden_state, past_key_values)
    """

    config_class = ReadableLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False  # TODO: add gradient checkpointing

    def __init__(self, config: ReadableLMConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [ReadableDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.final_norm = RMSNorm(config.hidden_size, epsilon=config.rms_norm_epsilon)

        # Weight initialisation (HF convention)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed_tokens = value

    # ---- Causal mask helper ----

    @staticmethod
    def _make_causal_mask(
        query_length: int,
        key_value_length: int,
        dtype: torch.dtype,
        device: torch.device,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Build the combined causal + padding mask used by all attention layers.

        Returns a (batch, 1, query_length, key_value_length) additive mask
        where allowed positions are 0.0 and masked positions are -inf.

        Args:
            query_length:     Length of the current query (1 during decode).
            key_value_length: Total KV length including past cache.
            dtype:            Model dtype for the mask tensor.
            device:           Device for the mask tensor.
            attention_mask:   Optional (batch, key_value_length) with 1=keep, 0=pad.

        Returns:
            Combined mask of shape (batch_or_1, 1, query_length, key_value_length).
        """
        # Causal (lower-triangular) mask
        # For decode (query_length=1), every past position is visible.
        causal_mask = torch.full(
            (query_length, key_value_length), fill_value=float("-inf"), dtype=dtype, device=device
        )
        # Allow attending to current and all previous positions
        mask_cond = torch.arange(key_value_length, device=device)
        # Each query position q can attend to kv positions <= (kv_length - query_length + q)
        causal_mask.masked_fill_(
            mask_cond <= (mask_cond[-(query_length):].unsqueeze(-1)), 0.0
        )
        # Reshape to (1, 1, query_length, key_value_length) for broadcasting
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        if attention_mask is not None:
            # attention_mask: (batch, kv_len) — 1 for real tokens, 0 for padding
            # Convert to additive: 0 → -inf, 1 → 0
            # NOTE: We use masked_fill instead of (1-mask)*-inf because
            # 0.0 * -inf = NaN in IEEE 754, which would poison all gradients.
            padding_mask = torch.zeros(
                attention_mask.shape[0], 1, 1, attention_mask.shape[1],
                dtype=dtype, device=attention_mask.device,
            )
            padding_mask.masked_fill_(attention_mask[:, None, None, :] == 0, float("-inf"))
            causal_mask = causal_mask + padding_mask

        return causal_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        use_cache: Optional[bool] = None,
    ) -> BaseModelOutputWithPast:
        batch_size, seq_len = input_ids.shape
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # --- Compute past length for position_ids and mask ---
        past_length = _get_cache_length(past_key_values)

        # --- Position IDs ---
        if position_ids is None:
            position_ids = torch.arange(
                past_length, past_length + seq_len, device=input_ids.device
            ).unsqueeze(0).expand(batch_size, -1)

        # --- Build combined causal + padding mask ---
        key_value_length = past_length + seq_len

        # Expand attention_mask to cover full kv length if provided
        if attention_mask is not None and attention_mask.shape[1] < key_value_length:
            # During generation, HF may pass the full-length attention_mask;
            # this branch handles edge cases.
            pad_len = key_value_length - attention_mask.shape[1]
            attention_mask = F.pad(attention_mask, (pad_len, 0), value=1)

        combined_mask = self._make_causal_mask(
            query_length=seq_len,
            key_value_length=key_value_length,
            dtype=self.embed_tokens.weight.dtype,
            device=input_ids.device,
            attention_mask=attention_mask,
        )

        # --- Embedding ---
        hidden_states = self.embed_tokens(input_ids)

        # --- Pass through decoder layers ---
        new_key_values: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx, layer in enumerate(self.layers):
            layer_past = _get_layer_cache(past_key_values, layer_idx)

            hidden_states, present_key_value = layer(
                hidden_states,
                attention_mask=combined_mask,
                position_ids=position_ids,
                past_key_value=layer_past,
                use_cache=use_cache,
            )
            if use_cache:
                new_key_values.append(present_key_value)

        # --- Final norm ---
        hidden_states = self.final_norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=tuple(new_key_values) if use_cache else None,
        )


# ---------------------------------------------------------------------------
# 8. ReadableLMForCausalLM (language-modelling head)
# ---------------------------------------------------------------------------

class ReadableLMForCausalLM(PreTrainedModel, GenerationMixin):
    """
    Decoder-only transformer with a causal language-modelling head.

    Wraps ReadableLMModel and adds a linear projection from hidden states
    to vocabulary logits.  When `labels` are provided, computes the
    standard next-token cross-entropy loss.

    Supports HuggingFace `generate()` via `prepare_inputs_for_generation`.

    Forward inputs:
        input_ids, attention_mask, position_ids, past_key_values, use_cache,
        labels (optional — shifted internally for causal LM loss).

    Forward output:
        CausalLMOutputWithPast(loss, logits, past_key_values)
    """

    config_class = ReadableLMConfig
    base_model_prefix = "model"

    def __init__(self, config: ReadableLMConfig):
        super().__init__(config)
        self.model = ReadableLMModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        use_cache: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        # --- Base model forward ---
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        # --- Compute loss if labels are provided ---
        loss = None
        if labels is not None:
            # Shift so that token[i] predicts token[i+1]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
        )

    # ---- Generation support ----

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        """
        Prepare model inputs for one step of autoregressive generation.

        During the first step (prefill), the full input_ids are passed.
        For subsequent steps, only the last token is passed and the rest
        comes from the KV cache.

        Returns a dict of keyword arguments for self.forward().
        """
        # Detect non-empty cache (works for both DynamicCache and tuples)
        cache_length = _get_cache_length(past_key_values)
        if cache_length > 0:
            # Only feed the last token — the rest is in the cache.
            input_ids = input_ids[:, -1:]

        # Compute position_ids from past cache length
        past_length = cache_length
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(
            past_length, past_length + seq_len, device=input_ids.device
        ).unsqueeze(0).expand(input_ids.shape[0], -1)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": True,
        }
