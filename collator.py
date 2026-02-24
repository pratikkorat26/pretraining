"""
collator.py
-----------
Minimal data collator for pre-packed, fixed-length sequences.

Since the streaming dataset already packs tokens into exactly
max_seq_length with no padding, the collator's only job is to
stack tensors into a batch and ensure correct dtypes.

No dynamic padding, no special logic — just stacking.
"""

from __future__ import annotations

import torch


class PackedCollator:
    """
    Collates a list of pre-packed samples into a batch.

    Each sample is a dict with:
        - input_ids:      (seq_len,) long tensor
        - labels:         (seq_len,) long tensor
        - attention_mask: (seq_len,) long tensor

    The collator stacks them into:
        - input_ids:      (batch_size, seq_len) long tensor
        - labels:         (batch_size, seq_len) long tensor
        - attention_mask: (batch_size, seq_len) long tensor

    No padding is performed — all sequences are already the same length.
    """

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.stack([f["input_ids"] for f in features]).long(),
            "labels": torch.stack([f["labels"] for f in features]).long(),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]).long(),
        }
