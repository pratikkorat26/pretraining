"""
callbacks.py
------------
Custom TrainerCallbacks for the ReadableLM training pipeline.

Includes:
    - GenerationCallback: runs model.generate() on a fixed prompt every N steps.
    - TokenCountCallback: tracks total tokens seen during training.
"""

from __future__ import annotations

import torch
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    PreTrainedTokenizerBase,
)


class GenerationCallback(TrainerCallback):
    """
    Generates text from a fixed prompt at regular intervals during training.

    Useful for qualitative monitoring — watching the model's outputs evolve
    from random tokens to coherent text as training progresses.

    Args:
        tokenizer:      Tokenizer for encoding prompts and decoding outputs.
        prompt:         Fixed text prompt to generate from.
        every_n_steps:  Run generation every N global steps.
        max_new_tokens: Maximum tokens to generate per sample.
        temperature:    Sampling temperature (< 1 = more focused).
        top_p:          Nucleus sampling threshold.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        prompt: str = "Explain gradient descent in simple terms.",
        every_n_steps: int = 500,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ):
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.every_n_steps = every_n_steps
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        """Run generation every `every_n_steps` global steps."""
        if state.global_step == 0 or state.global_step % self.every_n_steps != 0:
            return

        if model is None:
            return

        self._generate_sample(model, state.global_step)

    @torch.no_grad()
    def _generate_sample(self, model, step: int) -> None:
        """
        Encode the prompt, run model.generate(), and print the output.

        The model is temporarily set to eval mode, then restored.
        """
        was_training = model.training
        model.eval()

        device = next(model.parameters()).device
        input_ids = self.tokenizer.encode(
            self.prompt, return_tensors="pt"
        ).to(device)

        generated_ids = model.generate(
            input_ids,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        generated_text = self.tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        )

        print(f"\n{'='*60}")
        print(f"[Step {step}] Generation sample")
        print(f"{'='*60}")
        print(f"Prompt:    {self.prompt}")
        print(f"Generated: {generated_text}")
        print(f"{'='*60}\n")

        if was_training:
            model.train()


class TokenCountCallback(TrainerCallback):
    """
    Tracks and logs the total number of tokens seen during training.

    Computes: tokens_seen = global_step × gradient_accumulation × batch_size × seq_length

    This metric is useful for comparing training runs with different
    batch sizes or sequence lengths on an equal footing.

    Args:
        seq_length:  Number of tokens per sample (e.g. 4096).
        log_every_n_steps: Log token count every N steps (aligns with trainer logging).
    """

    def __init__(self, seq_length: int = 4096, log_every_n_steps: int = 50):
        self.seq_length = seq_length
        self.log_every_n_steps = log_every_n_steps

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict | None = None,
        **kwargs,
    ):
        """Inject tokens_seen into the training logs."""
        if logs is None:
            return

        tokens_per_step = (
            args.per_device_train_batch_size
            * args.gradient_accumulation_steps
            * self.seq_length
        )
        # Handle multi-GPU (though this pipeline targets single GPU)
        tokens_per_step *= args.world_size

        tokens_seen = state.global_step * tokens_per_step
        logs["tokens_seen"] = tokens_seen
