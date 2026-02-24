"""
data_streaming.py
-----------------
Streaming dataset with fixed-length token packing for causal LM training.

Loads any HuggingFace text dataset in streaming mode (never loads the full
dataset into memory), tokenizes on the fly, and packs tokens into
fixed-length sequences of exactly `max_seq_length` tokens.

The packing strategy concatenates tokenized documents into a long stream
and slices it into non-overlapping chunks.  No padding is ever inserted;
remainder tokens that don't fill a complete chunk are dropped.

Usage:
    dataset = create_streaming_dataset(
        dataset_name="allenai/c4",
        dataset_config="en",
        split="train",
        tokenizer=tokenizer,
        max_seq_length=4096,
        shuffle_buffer_size=10_000,
    )
"""

from __future__ import annotations

from typing import Iterator

import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizerBase


class PackedStreamingDataset(IterableDataset):
    """
    An IterableDataset that streams text data, tokenizes it, and packs
    tokens into fixed-length sequences.

    Each __iter__ call yields a dict with:
        - input_ids: (max_seq_length,) — packed token ids
        - labels:    (max_seq_length,) — identical to input_ids (causal LM)
        - attention_mask: (max_seq_length,) — all ones (no padding)

    Documents are concatenated into a continuous token stream.  When the
    buffer reaches max_seq_length tokens, a sample is emitted and the
    buffer is reset.  Leftover tokens at the end of the stream are dropped.

    Args:
        dataset_name:        HuggingFace dataset identifier (e.g. "allenai/c4").
        dataset_config:      Dataset config/subset (e.g. "en").  None if not needed.
        split:               Dataset split (e.g. "train").
        tokenizer:           A HuggingFace tokenizer instance.
        max_seq_length:      Number of tokens per packed sample.
        text_column:         Name of the text column in the dataset.
        shuffle_buffer_size: Number of raw examples to buffer for shuffling.
                             Set to 0 to disable shuffling.
        max_samples:         Maximum packed samples to yield per epoch.
                             None = unlimited (for training). Set to a finite
                             value for eval so the Trainer knows when to stop.
        seed:                Random seed for shuffle reproducibility.
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_config: str | None,
        split: str,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int = 4096,
        text_column: str = "text",
        shuffle_buffer_size: int = 10_000,
        max_samples: int | None = None,
        seed: int = 42,
    ):
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.split = split
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.text_column = text_column
        self.shuffle_buffer_size = shuffle_buffer_size
        self.max_samples = max_samples
        self.seed = seed

    def _raw_stream(self):
        """
        Load the HuggingFace dataset in streaming mode.  Returns an iterable
        of raw text examples, optionally shuffled via a buffer.
        """
        dataset = load_dataset(
            self.dataset_name,
            self.dataset_config,
            split=self.split,
            streaming=True,
        )
        if self.shuffle_buffer_size > 0:
            dataset = dataset.shuffle(
                seed=self.seed,
                buffer_size=self.shuffle_buffer_size,
            )
        return dataset

    def _tokenize_batch(self, texts: list[str]) -> list[int]:
        """
        Tokenize a batch of documents at once and return a flat token list.

        Batch tokenization is 5-10x faster than calling encode() per document
        because the tokenizer can process multiple strings in parallel.
        An EOS token is appended after each document to mark boundaries.
        """
        if not texts:
            return []

        # Batch encode — returns list of lists
        batch_ids = self.tokenizer(
            texts,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]

        tokens: list[int] = []
        eos = self.tokenizer.eos_token_id
        for doc_ids in batch_ids:
            tokens.extend(doc_ids)
            if eos is not None:
                tokens.append(eos)
        return tokens

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """
        Pack tokens into fixed-length chunks and yield training samples.

        Documents are batch-tokenized (TOKENIZE_BATCH_SIZE at a time) and
        their tokens are accumulated in a buffer.  When the buffer reaches
        max_seq_length, a sample is sliced off.  Remainder tokens carry
        over to the next batch.  Leftover tokens at the very end are dropped.

        If max_samples is set, stops after emitting that many samples.
        """
        TOKENIZE_BATCH_SIZE = 64   # docs per tokenizer call
        buffer: list[int] = []
        buffer_start = 0
        samples_emitted = 0
        doc_batch: list[str] = []
        seq_len = self.max_seq_length

        for example in self._raw_stream():
            text = example[self.text_column]
            if not text or not text.strip():
                continue
            doc_batch.append(text)

            # Tokenize in batches for throughput
            if len(doc_batch) >= TOKENIZE_BATCH_SIZE:
                buffer.extend(self._tokenize_batch(doc_batch))
                doc_batch = []

                # Emit as many complete samples as possible
                while (len(buffer) - buffer_start) >= seq_len:
                    input_ids = torch.tensor(
                        buffer[buffer_start: buffer_start + seq_len], dtype=torch.long
                    )
                    buffer_start += seq_len
                    yield {
                        "input_ids": input_ids,
                        "labels": input_ids.clone(),
                        "attention_mask": torch.ones_like(input_ids),
                    }
                    samples_emitted += 1
                    if self.max_samples is not None and samples_emitted >= self.max_samples:
                        return
                # Periodically compact consumed prefix to avoid unbounded list growth.
                if buffer_start >= 1_000_000:
                    buffer = buffer[buffer_start:]
                    buffer_start = 0

        # Flush remaining docs in the last partial batch
        if doc_batch:
            buffer.extend(self._tokenize_batch(doc_batch))
            while (len(buffer) - buffer_start) >= seq_len:
                input_ids = torch.tensor(
                    buffer[buffer_start: buffer_start + seq_len], dtype=torch.long
                )
                buffer_start += seq_len
                yield {
                    "input_ids": input_ids,
                    "labels": input_ids.clone(),
                    "attention_mask": torch.ones_like(input_ids),
                }
                samples_emitted += 1
                if self.max_samples is not None and samples_emitted >= self.max_samples:
                    return

        # Drop remainder — no padding in packed training


def create_streaming_dataset(
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int = 4096,
    text_column: str = "text",
    shuffle_buffer_size: int = 10_000,
    max_samples: int | None = None,
    seed: int = 42,
) -> PackedStreamingDataset:
    """
    Factory function to create a packed streaming dataset.

    This is the primary public API.  See PackedStreamingDataset for details.

    Args:
        dataset_name:        HuggingFace dataset identifier.
        dataset_config:      Dataset config/subset name, or None.
        split:               Dataset split name.
        tokenizer:           HuggingFace tokenizer.
        max_seq_length:      Tokens per packed sample (default 4096).
        text_column:         Column name containing text.
        shuffle_buffer_size: Buffer size for streaming shuffle.
        max_samples:         Max samples to yield (None = unlimited).
        seed:                Random seed.

    Returns:
        A PackedStreamingDataset ready for use with HF Trainer.
    """
    return PackedStreamingDataset(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        text_column=text_column,
        shuffle_buffer_size=shuffle_buffer_size,
        max_samples=max_samples,
        seed=seed,
    )
