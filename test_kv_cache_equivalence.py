"""
KV Cache Equivalence Test
=========================

Verifies that the KV cache produces correct results by comparing:

    (a) FULL forward:   model([prompt + next_token])  → logits for all tokens
    (b) CACHED forward: model(prompt, use_cache=True)  → cache
                        model(next_token, past_key_values=cache) → logits for next_token

The logits for the last token must match within a tight tolerance.
This is the fundamental correctness check for any autoregressive model.

Usage:
    python test_kv_cache_equivalence.py
"""

import sys
import torch

# Allow running from the repo root
sys.path.insert(0, ".")

from configuration_readable_lm import ReadableLMConfig
from modeling_readable_lm import ReadableLMForCausalLM


def test_kv_cache_equivalence() -> None:
    """
    Compare logits from full-sequence forward vs prefill+decode forward.

    Steps:
        1. Create a tiny model with deterministic weights.
        2. Construct a prompt (5 tokens) and a next_token (1 token).
        3. Run full forward on [prompt + next_token] → get logits[-1].
        4. Run prefill on prompt with use_cache=True → get cache.
        5. Run decode on next_token with past_key_values=cache → get logits[0].
        6. Assert that logits from (3) and (5) match.
    """
    print("=" * 60)
    print("KV Cache Equivalence Test")
    print("=" * 60)

    # --- Setup: reproducible tiny model ---
    torch.manual_seed(42)
    config = ReadableLMConfig(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,  # GQA: 2 KV heads shared by 4 query heads
        intermediate_size=128,
        max_position_embeddings=128,
        use_cache=True,
    )
    model = ReadableLMForCausalLM(config)
    model.eval()

    # --- Input data ---
    prompt_ids = torch.tensor([[10, 20, 30, 40, 50]])        # (1, 5)
    next_token_id = torch.tensor([[60]])                      # (1, 1)
    full_ids = torch.cat([prompt_ids, next_token_id], dim=1)  # (1, 6)

    print(f"Prompt:     {prompt_ids.tolist()}")
    print(f"Next token: {next_token_id.tolist()}")
    print(f"Full input: {full_ids.tolist()}")

    with torch.no_grad():
        # --- (a) Full forward on entire sequence ---
        full_output = model(input_ids=full_ids, use_cache=False)
        full_logits = full_output.logits                      # (1, 6, vocab_size)
        logits_from_full = full_logits[:, -1, :]              # (1, vocab_size)

        # --- (b) Prefill + decode ---
        prefill_output = model(input_ids=prompt_ids, use_cache=True)
        cache = prefill_output.past_key_values

        decode_output = model(
            input_ids=next_token_id,
            past_key_values=cache,
            use_cache=True,
        )
        logits_from_decode = decode_output.logits[:, -1, :]   # (1, vocab_size)

    # --- Compare ---
    max_abs_diff = (logits_from_full - logits_from_decode).abs().max().item()
    mean_abs_diff = (logits_from_full - logits_from_decode).abs().mean().item()

    print(f"\nMax  absolute difference: {max_abs_diff:.2e}")
    print(f"Mean absolute difference: {mean_abs_diff:.2e}")

    # Tolerance: fp32 accumulation should give very tight match
    tolerance = 1e-5
    if max_abs_diff < tolerance:
        print(f"\n✓ PASSED — logits match within tolerance ({tolerance})")
    else:
        print(f"\n✗ FAILED — max diff {max_abs_diff:.2e} exceeds tolerance {tolerance}")
        sys.exit(1)

    # --- Bonus: verify cache shapes ---
    num_layers = config.num_hidden_layers
    assert len(cache) == num_layers, f"Expected {num_layers} cache entries, got {len(cache)}"
    for layer_idx, (cached_key, cached_value) in enumerate(cache):
        expected_shape = (1, config.num_key_value_heads, prompt_ids.shape[1], config.hidden_size // config.num_attention_heads)
        assert cached_key.shape == expected_shape, (
            f"Layer {layer_idx} key shape {cached_key.shape} != expected {expected_shape}"
        )
        assert cached_value.shape == expected_shape, (
            f"Layer {layer_idx} value shape {cached_value.shape} != expected {expected_shape}"
        )
    print(f"✓ Cache shapes correct for all {num_layers} layers")

    # --- Bonus: multi-step decode consistency ---
    print("\n--- Multi-step decode test ---")
    extra_tokens = torch.tensor([[70, 80, 90]])  # 3 more tokens
    full_extended = torch.cat([full_ids, extra_tokens], dim=1)  # (1, 9)

    with torch.no_grad():
        # Full forward on all 9 tokens
        full_ext_output = model(input_ids=full_extended, use_cache=False)
        full_ext_logits = full_ext_output.logits  # (1, 9, vocab_size)

        # Continue from previous cache (which had 5 prompt tokens)
        # Now decode tokens [60, 70, 80, 90] one at a time
        running_cache = cache
        for step, token_id in enumerate([60, 70, 80, 90]):
            token_tensor = torch.tensor([[token_id]])
            step_output = model(
                input_ids=token_tensor,
                past_key_values=running_cache,
                use_cache=True,
            )
            running_cache = step_output.past_key_values

            # The logit for this token should match position (5 + step) in full forward
            full_pos = 5 + step
            step_logits = step_output.logits[:, -1, :]
            ref_logits = full_ext_logits[:, full_pos, :]
            step_diff = (step_logits - ref_logits).abs().max().item()
            status = "✓" if step_diff < tolerance else "✗"
            print(f"  Step {step} (token {token_id}): max diff = {step_diff:.2e} {status}")
            assert step_diff < tolerance, f"Multi-step decode failed at step {step}"

    print("\n✓ All multi-step decode checks passed")
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_kv_cache_equivalence()
