"""
train_minimal.py
----------------
Minimal training loop for ReadableLM on a tiny dummy dataset.

Usage:
    python train_minimal.py

What it does:
  1. Builds a tiny ReadableLM config (fast to train on CPU).
  2. Loads the GPT-2 tokenizer (or any AutoTokenizer target).
  3. Creates a small dummy dataset of repeated short sentences.
  4. Runs a simple PyTorch training loop (no Trainer dependency).
  5. Saves the model + config to output_dir.
  6. Reloads from output_dir and runs model.generate() to prove
     save/load + generation work end-to-end.

No GPU required — this is intentionally tiny.
"""

import os
import sys

# Allow imports from the same directory
sys.path.insert(0, os.path.dirname(__file__))

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from configuration_readable_lm import ReadableLMConfig
from modeling_readable_lm import ReadableLMForCausalLM

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUTPUT_DIR      = "output_readable_lm"
MAX_SEQ_LEN     = 64
BATCH_SIZE      = 4
NUM_EPOCHS      = 10
LEARNING_RATE   = 3e-4
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

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
    Tokenises a list of strings and returns fixed-length (MAX_SEQ_LEN)
    input_ids + labels tensors.  Labels equal input_ids (standard causal LM).

    Sequences shorter than MAX_SEQ_LEN are padded with the pad token;
    labels at pad positions are set to -100 so the loss ignores them.
    """

    def __init__(self, texts: list[str], tokenizer, max_len: int):
        self.tokenizer = tokenizer
        self.max_len   = max_len
        # Repeat corpus so we have a few hundred examples
        self.texts = texts * 30

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text   = self.texts[idx]
        tokens = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = tokens["input_ids"].squeeze(0)      # (max_len,)
        attention_mask = tokens["attention_mask"].squeeze(0) # (max_len,)

        # Mask pad tokens in labels so they don't contribute to loss
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print(f"Training on device: {DEVICE}")

    # ── Tokenizer ────────────────────────────────────────────────────────────
    print("Loading tokenizer (gpt2)…")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # GPT-2 has no pad token by default; set it to eos so padding works.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"  Set pad_token = eos_token = '{tokenizer.eos_token}'")

    # ── Model ─────────────────────────────────────────────────────────────────
    config = ReadableLMConfig(
        vocab_size=len(tokenizer),       # match tokenizer
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        intermediate_size=512,
        max_position_embeddings=MAX_SEQ_LEN,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = ReadableLMForCausalLM(config).to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # ── Data ──────────────────────────────────────────────────────────────────
    dataset    = TinyTextDataset(CORPUS, tokenizer, max_len=MAX_SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    for epoch in range(1, NUM_EPOCHS + 1):
        total_loss = 0.0
        for step, batch in enumerate(dataloader):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)

            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = output.loss

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"  Epoch {epoch}/{NUM_EPOCHS}  avg_loss={avg_loss:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nModel saved to '{OUTPUT_DIR}'")

    # ── Reload + generate ────────────────────────────────────────────────────
    print("\nReloading model from disk and running generation…")
    reloaded_model = ReadableLMForCausalLM.from_pretrained(OUTPUT_DIR).to(DEVICE)
    reloaded_tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    reloaded_model.eval()

    prompt = "Attention is"
    prompt_ids = reloaded_tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)

    with torch.no_grad():
        generated_ids = reloaded_model.generate(
            prompt_ids,
            max_new_tokens=20,
            do_sample=False,          # greedy — deterministic
            use_cache=True,
        )

    generated_text = reloaded_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"Prompt:    '{prompt}'")
    print(f"Generated: '{generated_text}'")
    print("\nDone.")


if __name__ == "__main__":
    main()