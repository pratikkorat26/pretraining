"""
train_minimal.py
----------------
Minimal training script for ReadableLM using the HuggingFace Trainer API.

Demonstrates that ReadableLM is a fully HF-native model — it plugs directly
into the standard Trainer pipeline with zero custom training code.

Usage:
    python train_minimal.py

What it does:
  1. Loads GPT-2 tokenizer (sets pad_token = eos_token).
  2. Builds a small ReadableLM from a fresh config.
  3. Creates a tiny dummy dataset of tokenized sentences.
  4. Trains with HuggingFace Trainer (causal LM objective).
  5. Saves model + tokenizer via save_pretrained().
  6. Reloads from disk and runs model.generate() to prove
     the full save/load/generate pipeline works.
"""

import os
import sys

# Allow imports from the same directory
sys.path.insert(0, os.path.dirname(__file__))

import torch
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from torch.utils.data import Dataset

from configuration_readable_lm import ReadableLMConfig
from modeling_readable_lm import ReadableLMForCausalLM

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUTPUT_DIR  = "output_readable_lm"
MAX_SEQ_LEN = 64

CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "Transformers are powerful sequence-to-sequence models.",
    "Attention is all you need to build great language models.",
    "Rotary embeddings give the model a sense of token position.",
    "Grouped query attention reduces memory without hurting quality.",
    "Language models predict the next token given prior context.",
    "Training on simple data teaches the model basic patterns.",
    "The decoder stack processes tokens left to right.",
]

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class TinyTextDataset(Dataset):
    """
    Tokenises a list of strings and returns fixed-length input_ids.

    The HF DataCollatorForLanguageModeling will handle creating labels
    (clone of input_ids with pad positions set to -100) automatically.
    """

    def __init__(self, texts: list[str], tokenizer, max_len: int, repeats: int = 30):
        self.samples = []
        expanded = texts * repeats  # repeat corpus for more training data

        for text in expanded:
            tokens = tokenizer(
                text,
                max_length=max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            self.samples.append({
                "input_ids": tokens["input_ids"].squeeze(0),
                "attention_mask": tokens["attention_mask"].squeeze(0),
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.samples[idx]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    # ── Tokenizer ─────────────────────────────────────────────────────────
    print("Loading tokenizer (gpt2)…")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # GPT-2 has no pad token by default; setting it to eos is standard practice.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"  Set pad_token = eos_token = '{tokenizer.eos_token}'")

    # ── Model ─────────────────────────────────────────────────────────────
    config = ReadableLMConfig(
        vocab_size=len(tokenizer),
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        intermediate_size=512,
        max_position_embeddings=MAX_SEQ_LEN,
        attn_implementation="sdpa",
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = ReadableLMForCausalLM(config)
    model = torch.compile(model)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # ── Dataset ───────────────────────────────────────────────────────────
    dataset = TinyTextDataset(CORPUS, tokenizer, max_len=MAX_SEQ_LEN)
    print(f"Dataset: {len(dataset)} samples, max_length={MAX_SEQ_LEN}")

    # ── HuggingFace Trainer ───────────────────────────────────────────────
    # DataCollatorForLanguageModeling automatically creates `labels` from
    # `input_ids` and masks pad tokens with -100 for causal LM training.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # causal LM — not masked language modelling
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=10,
        per_device_train_batch_size=8,
        learning_rate=3e-4,
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_steps=10,
        save_strategy="no",        # we save manually after training
        report_to="none",          # disable wandb / tensorboard
        seed=42,
        fp16=False,
        bf16=(device == "cuda"),   # use bf16 on GPU for speed
        # torch.compile wraps the model in OptimizedModule, which hides
        # the forward() signature from Trainer's column auto-detection.
        # Setting this to False ensures all dataset columns are kept.
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print("\nStarting HF Trainer…")
    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nModel saved to '{OUTPUT_DIR}'")

    # ── Reload + generate ─────────────────────────────────────────────────
    print("\nReloading model from disk and running generation…")
    reloaded_model = ReadableLMForCausalLM.from_pretrained(OUTPUT_DIR).to(device)
    reloaded_tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    reloaded_model.eval()

    if reloaded_tokenizer.pad_token is None:
        reloaded_tokenizer.pad_token = reloaded_tokenizer.eos_token

    prompt = "Attention is"
    prompt_ids = reloaded_tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        generated_ids = reloaded_model.generate(
            prompt_ids,
            max_new_tokens=20,
            do_sample=False,
            use_cache=True,
            pad_token_id=reloaded_tokenizer.pad_token_id,
        )

    generated_text = reloaded_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"Prompt:    '{prompt}'")
    print(f"Generated: '{generated_text}'")
    print("\nDone.")


if __name__ == "__main__":
    main()