"""
train_streaming.py
------------------
Production training script for ReadableLM using HuggingFace Trainer with
streaming data, token packing, and periodic generation monitoring.

Designed for single-GPU training on an 8GB card (e.g. RTX 4070).

Usage:
    python train_streaming.py

    # Resume from checkpoint:
    python train_streaming.py --resume_from_checkpoint output/checkpoint-5000

    # Override dataset:
    python train_streaming.py --dataset_name="HuggingFaceFW/fineweb-edu" \\
                              --dataset_config="sample-10BT"

Key design choices:
    - Streaming dataset (never loads full data into memory)
    - Fixed-length 4K token packing (no padding waste)
    - Per-param-group learning rates (embeddings vs transformer body)
    - Cosine LR schedule with linear warmup
    - Gradient checkpointing for VRAM savings
    - bf16 mixed precision (falls back to fp16)
    - Generation samples every 500 steps for qualitative monitoring
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys

# Allow imports from the same directory
sys.path.insert(0, os.path.dirname(__file__))

import torch
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    get_cosine_schedule_with_warmup,
)

from configuration_readable_lm import ReadableLMConfig
from modeling_readable_lm import ReadableLMForCausalLM
from data_streaming import create_streaming_dataset
from collator import PackedCollator
from callbacks import GenerationCallback, TokenCountCallback

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULTS = {
    # Model
    "vocab_size": 50304,          # divisible by 64 for GPU efficiency
    "hidden_size": 512,
    "num_hidden_layers": 22,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,     # GQA: 4 KV heads, 4 queries per KV head
    "intermediate_size": 2048,
    "max_position_embeddings": 4096,
    "attn_implementation": "sdpa",

    # Data
    "dataset_name": "allenai/c4",
    "dataset_config": "en",
    "dataset_split": "train",
    "text_column": "text",
    "max_seq_length": 4096,
    "shuffle_buffer_size": 10_000,

    # Training
    "total_steps": 300_000,
    "warmup_steps": 3_000,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "body_lr": 2e-4,
    "embed_lr": 1e-4,
    "weight_decay": 0.1,
    "max_grad_norm": 1.0,

    # Logging & saving
    "logging_steps": 50,
    "save_steps": 5_000,
    "save_total_limit": 2,
    "output_dir": "output",

    # Evaluation
    "eval_steps": 10_000,
    "eval_max_samples": 20,       # keep eval lightweight to reduce training stalls
    "eval_split": "train",        # many datasets only have "train"; uses different seed

    # Generation callback
    "generation_every_n_steps": 0,  # 0 disables generation callback during training
    "generation_prompt": "Explain gradient descent in simple terms.",
    "generation_max_new_tokens": 128,

    # Misc
    "seed": 42,
    "tf32": True,
    "torch_compile": False,
}

# Streaming tokenization can bottleneck throughput; keep at least one worker.
if os.name == "nt":
    DEFAULTS["dataloader_num_workers"] = 1
else:
    DEFAULTS["dataloader_num_workers"] = 2


# ---------------------------------------------------------------------------
# Optimizer with per-parameter-group learning rates
# ---------------------------------------------------------------------------


def create_optimizer(model: ReadableLMForCausalLM, args) -> torch.optim.AdamW:
    """
    Build AdamW with two parameter groups:

    Group 1 — Transformer body (attention, MLP, norms):
        lr = 2e-4, weight_decay = 0.1

    Group 2 — Embeddings + LM head:
        lr = 1e-4, weight_decay = 0.0

    Embeddings use a lower learning rate and no weight decay because they
    are lookup tables, not linear transforms, and are more sensitive to
    large updates early in training.
    """
    embedding_names = {"embed_tokens", "lm_head"}

    embed_params = []
    body_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_embedding = any(emb_name in name for emb_name in embedding_names)
        if is_embedding:
            embed_params.append(param)
        else:
            body_params.append(param)

    param_groups = [
        {
            "params": body_params,
            "lr": args["body_lr"],
            "weight_decay": args["weight_decay"],
        },
        {
            "params": embed_params,
            "lr": args["embed_lr"],
            "weight_decay": 0.0,
        },
    ]

    adamw_kwargs = {
        "betas": (0.9, 0.95),
        "eps": 1e-8,
    }
    # Fused AdamW is substantially faster on modern CUDA GPUs.
    if torch.cuda.is_available() and "fused" in inspect.signature(torch.optim.AdamW).parameters:
        adamw_kwargs["fused"] = True

    return torch.optim.AdamW(param_groups, **adamw_kwargs)


# ---------------------------------------------------------------------------
# Custom Trainer to inject our optimizer + scheduler
# ---------------------------------------------------------------------------


class ReadableLMTrainer(Trainer):
    """
    Extends HF Trainer to use our custom optimizer with per-param-group
    learning rates and a cosine schedule with linear warmup.

    By overriding create_optimizer_and_scheduler(), we keep full control
    over the optimizer while still using the standard Trainer loop for
    everything else (gradient accumulation, checkpointing, logging, etc.).
    """

    def __init__(self, pipeline_args: dict, **kwargs):
        self.pipeline_args = pipeline_args
        super().__init__(**kwargs)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """Create AdamW with param groups + cosine schedule."""
        self.optimizer = create_optimizer(self.model, self.pipeline_args)

        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.pipeline_args["warmup_steps"],
            num_training_steps=num_training_steps,
        )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> dict:
    """Parse CLI arguments, falling back to DEFAULTS for anything not specified."""
    parser = argparse.ArgumentParser(description="Train ReadableLM with streaming data")

    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dataloader_num_workers", type=int, default=None)
    parser.add_argument("--tf32", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=None)
    parser.add_argument("--torch_compile", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=None)

    cli_args = parser.parse_args()

    # Start with defaults, override with CLI
    args = dict(DEFAULTS)
    for key, value in vars(cli_args).items():
        if value is not None:
            args[key] = value

    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # ── Device info ───────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    dtype_label = "bf16" if use_bf16 else ("fp16" if device == "cuda" else "fp32")

    print(f"Device: {device} | Precision: {dtype_label}")
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        if args["tf32"]:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
            print("TF32: enabled (matmul + cuDNN)")
        else:
            print("TF32: disabled")

    # ── Tokenizer ─────────────────────────────────────────────────────────
    print("\nLoading tokenizer (gpt2)…")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"  pad_token = eos_token = '{tokenizer.eos_token}'")

    # ── Model ─────────────────────────────────────────────────────────────
    config = ReadableLMConfig(
        vocab_size=args["vocab_size"],
        hidden_size=args["hidden_size"],
        num_hidden_layers=args["num_hidden_layers"],
        num_attention_heads=args["num_attention_heads"],
        num_key_value_heads=args["num_key_value_heads"],
        intermediate_size=args["intermediate_size"],
        max_position_embeddings=args["max_position_embeddings"],
        attn_implementation=args["attn_implementation"],
        use_cache=False,  # important: disable KV-cache during training
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    model = ReadableLMForCausalLM(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {num_params / 1e6:.0f}M parameters")
    print(f"  hidden_size={config.hidden_size}, layers={config.num_hidden_layers}")
    print(f"  heads={config.num_attention_heads}, kv_heads={config.num_key_value_heads}")
    print(f"  intermediate_size={config.intermediate_size}")
    print(f"  attention: {config.attn_implementation}")

    # ── Streaming Dataset ─────────────────────────────────────────────────
    print(f"\nStreaming dataset: {args['dataset_name']} ({args['dataset_config']})")
    print(f"  Packing to {args['max_seq_length']} tokens per sample")
    print(f"  Shuffle buffer: {args['shuffle_buffer_size']} examples")

    train_dataset = create_streaming_dataset(
        dataset_name=args["dataset_name"],
        dataset_config=args["dataset_config"],
        split=args["dataset_split"],
        tokenizer=tokenizer,
        max_seq_length=args["max_seq_length"],
        text_column=args["text_column"],
        shuffle_buffer_size=args["shuffle_buffer_size"],
        seed=args["seed"],
    )

    # ── Eval Dataset ──────────────────────────────────────────────────────
    # Uses a different seed so it streams different data than training.
    # Capped at eval_max_samples to keep eval fast and deterministic.
    eval_dataset = create_streaming_dataset(
        dataset_name=args["dataset_name"],
        dataset_config=args["dataset_config"],
        split=args["eval_split"],
        tokenizer=tokenizer,
        max_seq_length=args["max_seq_length"],
        text_column=args["text_column"],
        shuffle_buffer_size=0,                     # no shuffle for reproducible eval
        max_samples=args["eval_max_samples"],
        seed=args["seed"] + 1,                     # different seed → different data
    )
    print(f"  Eval: {args['eval_max_samples']} samples every {args['eval_steps']} steps")

    # ── Effective batch info ──────────────────────────────────────────────
    effective_batch = args["per_device_train_batch_size"] * args["gradient_accumulation_steps"]
    tokens_per_step = effective_batch * args["max_seq_length"]
    total_tokens = args["total_steps"] * tokens_per_step

    print(f"\nTraining plan:")
    print(f"  Batch: {args['per_device_train_batch_size']} × {args['gradient_accumulation_steps']} accum = {effective_batch} effective")
    print(f"  Tokens/step: {tokens_per_step:,}")
    print(f"  Total steps: {args['total_steps']:,}")
    print(f"  Total tokens: {total_tokens / 1e9:.1f}B")
    print(f"  Warmup: {args['warmup_steps']:,} steps")
    print(f"  LR: body={args['body_lr']}, embed={args['embed_lr']}")
    print(f"  DataLoader workers: {args['dataloader_num_workers']}")
    print(f"  torch.compile: {args['torch_compile']}")

    # ── Callbacks ─────────────────────────────────────────────────────────
    token_count_callback = TokenCountCallback(
        seq_length=args["max_seq_length"],
        log_every_n_steps=args["logging_steps"],
    )
    callbacks = [token_count_callback]
    if args["generation_every_n_steps"] > 0:
        generation_callback = GenerationCallback(
            tokenizer=tokenizer,
            prompt=args["generation_prompt"],
            every_n_steps=args["generation_every_n_steps"],
            max_new_tokens=args["generation_max_new_tokens"],
            temperature=0.8,
            top_p=0.95,
        )
        callbacks.append(generation_callback)

    # ── Training arguments ────────────────────────────────────────────────
    training_args_kwargs = dict(
        output_dir=args["output_dir"],
        max_steps=args["total_steps"],
        per_device_train_batch_size=args["per_device_train_batch_size"],
        per_device_eval_batch_size=args["per_device_train_batch_size"],
        gradient_accumulation_steps=args["gradient_accumulation_steps"],
        max_grad_norm=args["max_grad_norm"],

        # Precision
        fp16=(device == "cuda" and not use_bf16),
        bf16=use_bf16,

        # Memory optimization
        gradient_checkpointing=False,

        # Logging
        logging_steps=args["logging_steps"],
        logging_first_step=True,
        report_to="none",               # set to "wandb" if you want W&B

        # Evaluation
        eval_strategy="steps",
        eval_steps=args["eval_steps"],

        # Checkpointing
        save_steps=args["save_steps"],
        save_total_limit=args["save_total_limit"],

        # Misc
        seed=args["seed"],
        remove_unused_columns=False,     # required for pre-packed IterableDataset
        dataloader_pin_memory=True,
        dataloader_num_workers=args["dataloader_num_workers"],
    )
    if "torch_compile" in inspect.signature(TrainingArguments.__init__).parameters:
        training_args_kwargs["torch_compile"] = (device == "cuda" and args["torch_compile"])
    if args["dataloader_num_workers"] > 0:
        training_args_kwargs["dataloader_prefetch_factor"] = 4
        training_args_kwargs["dataloader_persistent_workers"] = True
    training_args = TrainingArguments(**training_args_kwargs)

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = ReadableLMTrainer(
        pipeline_args=args,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=PackedCollator(),
        callbacks=callbacks,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    resume_ckpt = args.get("resume_from_checkpoint")
    if resume_ckpt:
        print(f"\nResuming from: {resume_ckpt}")
    print("\nStarting training…\n")

    trainer.train(resume_from_checkpoint=resume_ckpt)

    # ── Save final model ──────────────────────────────────────────────────
    final_dir = os.path.join(args["output_dir"], "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nFinal model saved to '{final_dir}'")


if __name__ == "__main__":
    main()
