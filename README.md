# ReadableLM — Decoder-Only Transformer from Scratch

A production-ready, OLMo3/LLaMA-style decoder-only language model implemented from scratch in PyTorch, with **readability** and **correctness** as first-class goals.

Fully native to the HuggingFace ecosystem — `save_pretrained()`, `from_pretrained()`, and `model.generate()` work out of the box. Includes a complete streaming training pipeline for single-GPU pretraining.

---

## Highlights

| Capability | Details |
|---|---|
| **Attention** | Dual-backend: Flash Attention via PyTorch SDPA (default) + eager fallback |
| **GQA** | Grouped-Query Attention with `repeat_kv` expansion |
| **RoPE** | Rotary Position Embeddings with configurable `rope_theta` |
| **Gated MLP** | SwiGLU activation: `down(silu(gate(x)) ⊙ up(x))` |
| **Normalization** | RMSNorm (pre-norm residuals, OLMo3/LLaMA ordering) |
| **KV Cache** | Correct prefill + decode, compatible with HF's `DynamicCache` |
| **Training** | HF Trainer with streaming data, token packing, custom optimizer |
| **Generation** | Full `GenerationMixin` — greedy, sampling, beam search |

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
│
│  ── Model ──────────────────────────────────────────────
├── configuration_readable_lm.py   # ReadableLMConfig
├── modeling_readable_lm.py        # Full model (RMSNorm, RoPE, GQA, SwiGLU, etc.)
│
│  ── Training Pipeline ──────────────────────────────────
├── train_streaming.py             # Main training script (HF Trainer)
├── data_streaming.py              # Streaming dataset + 4K token packing
├── collator.py                    # Minimal stacking collator
├── callbacks.py                   # Generation + token count callbacks
│
│  ── Testing & Demos ────────────────────────────────────
├── train_minimal.py               # Quick demo: train on dummy data
├── test_kv_cache_equivalence.py   # KV cache correctness test
└── README.md
```

---

## Quick Start

### Install

```bash
pip install torch transformers datasets
```

### Run KV Cache Tests

```bash
python test_kv_cache_equivalence.py
```

### Quick Demo (CPU, ~30 seconds)

```bash
python train_minimal.py
```

Trains a tiny model on dummy data, saves to disk, reloads, and generates text — proving the full HF pipeline works.

---

## Production Training

### Single-GPU Pretraining

```bash
python train_streaming.py
```

This launches a full training run with:
- **Streaming data** from `allenai/c4` (never loads into memory)
- **4K token packing** (no padding waste)
- **~250M parameter** model with GQA
- **Gradient checkpointing** for VRAM efficiency
- **bf16 mixed precision**
- **Cosine LR** with linear warmup

### Expected VRAM Usage

| Component | VRAM |
|---|---|
| Model parameters (bf16) | ~500 MB |
| Gradient checkpointing activations | ~1-2 GB |
| Optimizer states (AdamW, fp32) | ~2 GB |
| Gradients + buffer | ~1-2 GB |
| **Total** | **~5-6 GB** |

Fits comfortably on an 8GB GPU (RTX 4070, RTX 3070, etc.).

### How Streaming Works

```
HuggingFace Hub          data_streaming.py           Trainer
     │                        │                        │
     │  ──stream──→  Tokenize on the fly               │
     │                        │                        │
     │               Pack into 4096-token chunks        │
     │                        │                        │
     │                collator.py stacks batch          │
     │                        │                        │
     │                        └────→ Forward + backward │
```

1. `datasets.load_dataset(..., streaming=True)` fetches data in small chunks
2. Each document is tokenized and appended to a token buffer
3. When the buffer reaches 4096 tokens, a sample is emitted
4. Remainder tokens are dropped (no padding, ever)
5. A shuffle buffer of 10K examples provides randomness

### Resume from Checkpoint

```bash
python train_streaming.py --resume_from_checkpoint output/checkpoint-5000
```

The Trainer saves checkpoints every 5,000 steps (configurable), keeping the last 2. Resume picks up optimizer state, LR schedule, and global step count.

### Change Dataset

```bash
# Use FineWeb-Edu (10B token sample)
python train_streaming.py --dataset_name="HuggingFaceFW/fineweb-edu" \
                          --dataset_config="sample-10BT"

# Use The Pile
python train_streaming.py --dataset_name="EleutherAI/the_pile" \
                          --dataset_config=None
```

Any HuggingFace dataset with a `text` column works out of the box.

### Training Configuration

| Parameter | Value | Rationale |
|---|---|---|
| Batch size | 1 × 16 accum = 16 effective | Fits 8GB VRAM |
| Sequence length | 4096 | Standard for modern LLMs |
| Tokens/step | 65,536 | 16 × 4096 |
| Total steps | 300,000 | ~20B tokens |
| Warmup | 3,000 steps (1%) | Standard cosine warmup |
| Body LR | 2e-4 | Standard for 250M models |
| Embedding LR | 1e-4 | Lower to stabilize lookup tables |
| Weight decay | 0.1 (body only) | No decay on embeddings |
| Grad clip | 1.0 | Prevents spikes |

---

## Design Decisions

| Decision | Rationale |
|---|---|
| **Pre-norm residuals** | `x + attn(norm(x))` — matches OLMo3, LLaMA, Mistral |
| **SwiGLU over GELU** | ~1% better quality per FLOP (Shazeer 2020) |
| **GQA over MHA** | Reduces KV cache memory proportional to group ratio |
| **RoPE over learned** | Extrapolates to longer sequences; no embedding table |
| **SDPA default** | Flash Attention kernel when available, zero API change |
| **`masked_fill` masks** | Avoids `0 × -inf = NaN` (IEEE 754) correctness bug |
| **Token packing** | No padding waste — 100% of compute goes to real tokens |
| **Streaming data** | Constant memory regardless of dataset size |
| **Per-group LR** | Embeddings are lookup tables — need lower LR, no decay |

## What's Intentionally Omitted

Deliberate scope cuts for readability:

- Sliding-window / chunked attention
- Dynamic NTK-aware / YaRN RoPE scaling
- Tensor-parallel / FSDP distributed training
- DeepSpeed integration
- Multi-dataset mixing
- Tokenizer training

## License

MIT
