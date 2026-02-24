# ReadableLM — Decoder-Only Transformer from Scratch

A production-ready, OLMo3/LLaMA-style decoder-only language model implemented from scratch in ~800 lines of PyTorch, with **readability** and **correctness** as first-class goals.

Fully native to the HuggingFace ecosystem — `save_pretrained()`, `from_pretrained()`, and `model.generate()` work out of the box with zero external dependencies beyond PyTorch and Transformers.

---

## Highlights

| Capability | Details |
|---|---|
| **Attention** | Dual-backend: Flash Attention via PyTorch SDPA (default) + eager matmul→softmax fallback |
| **GQA** | Grouped-Query Attention — `num_kv_heads < num_heads` with `repeat_kv` expansion |
| **RoPE** | Rotary Position Embeddings with configurable `rope_theta` |
| **Gated MLP** | SwiGLU activation: `down(silu(gate(x)) ⊙ up(x))` |
| **Normalization** | RMSNorm (pre-norm residuals, OLMo3/LLaMA ordering) |
| **KV Cache** | Correct prefill + incremental decode, compatible with HF's `DynamicCache` |
| **Training** | Causal LM loss with label masking (`ignore_index=-100`) |
| **Generation** | Full `GenerationMixin` support — greedy, sampling, beam search |

## Architecture

```
input_ids
    │
    ▼
┌──────────────────┐
│  Token Embedding │
└────────┬─────────┘
         │
         ▼  × N layers
┌──────────────────────────────────┐
│  RMSNorm → Multi-Head Attention  │──→ + residual
│          (GQA + RoPE + KV Cache) │
│                                  │
│  RMSNorm → Gated MLP (SwiGLU)   │──→ + residual
└──────────────────────────────────┘
         │
         ▼
┌──────────────────┐
│    Final RMSNorm │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  LM Head (linear)│ → vocab logits
└──────────────────┘
```

## Repository Structure

```
pretraining/
├── configuration_readable_lm.py   # ReadableLMConfig (PretrainedConfig)
├── modeling_readable_lm.py        # Full model implementation
│   ├── repeat_kv()                #   GQA head expansion
│   ├── RMSNorm                    #   Root-mean-square normalization
│   ├── RotaryEmbedding            #   RoPE cos/sin precomputation
│   ├── apply_rotary_pos_emb()     #   RoPE rotation application
│   ├── ReadableAttention          #   SDPA / eager attention + KV cache
│   ├── ReadableMLP                #   SwiGLU gated MLP
│   ├── ReadableDecoderLayer       #   Pre-norm residual block
│   ├── ReadableLMModel            #   Base model (→ BaseModelOutputWithPast)
│   └── ReadableLMForCausalLM      #   CausalLM head (→ CausalLMOutputWithPast)
├── train_minimal.py               # Training loop + save/load + generate
├── test_kv_cache_equivalence.py   # KV cache correctness test
└── README.md
```

## Quick Start

### Install

```bash
pip install torch transformers
```

### Run Tests

```bash
python test_kv_cache_equivalence.py
```

Verifies that full-sequence forward and prefill+decode produce identical logits (within `atol=1e-5`), including multi-step decode consistency.

### Train + Generate

```bash
python train_minimal.py
```

Trains a small model on dummy data, saves to `./output_readable_lm/`, reloads from disk, and runs deterministic generation — proving the full save/load/generate pipeline works end-to-end.

```
Epoch 1/3   avg_loss=6.33
Epoch 2/3   avg_loss=1.04
Epoch 3/3   avg_loss=0.07

Prompt:    'Attention is'
Generated: 'Attention is all you need to build great language models.'
```

### Switch Attention Backend

```python
from configuration_readable_lm import ReadableLMConfig
from modeling_readable_lm import ReadableLMForCausalLM

# Default: SDPA (auto-dispatches to FlashAttention-2 on supported GPUs)
config = ReadableLMConfig(attn_implementation="sdpa")

# Fallback: explicit matmul → softmax (useful for debugging)
config = ReadableLMConfig(attn_implementation="eager")

model = ReadableLMForCausalLM(config)
```

## Design Decisions

| Decision | Rationale |
|---|---|
| **Pre-norm residuals** | `x + attn(norm(x))` — matches OLMo3, LLaMA, Mistral; more stable than post-norm at scale |
| **SwiGLU over GELU** | ~1% better quality per FLOP (Shazeer 2020); standard in modern LLMs |
| **GQA over MHA** | Reduces KV cache memory proportional to group ratio with minimal quality loss |
| **RoPE over learned positional** | Extrapolates to longer sequences; no embedding table overhead |
| **SDPA default** | Zero-cost abstraction — same API, but FlashAttention-2 kernel when available |
| **`masked_fill` for mask construction** | Avoids `0 × -inf = NaN` (IEEE 754) — a subtle but critical correctness detail |
| **DynamicCache + legacy tuple compat** | Works seamlessly with both HF v4.50+ (`DynamicCache`) and older versions |

## What's Intentionally Omitted

These are deliberate scope cuts for readability, not oversights:

- Sliding-window / chunked attention masks
- Dynamic NTK-aware / YaRN RoPE scaling
- Gradient checkpointing
- Tensor-parallel sharding annotations
- Tokenizer training (uses existing GPT-2 tokenizer)
- Multi-GPU / FSDP setup

## License

MIT
