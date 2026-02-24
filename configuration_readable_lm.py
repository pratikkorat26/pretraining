"""
ReadableLM Configuration
========================

A minimal, readable configuration for a decoder-only transformer (OLMo3-style).
Inherits from HuggingFace PretrainedConfig so that save_pretrained() /
from_pretrained() work out of the box.

Defaults define a *tiny* model suitable for testing and demonstration.
Scale up hidden_size, num_hidden_layers, etc. for real training.
"""

from transformers import PretrainedConfig


class ReadableLMConfig(PretrainedConfig):
    """
    Configuration for ReadableLM, a decoder-only transformer.

    This config stores every hyper-parameter needed to construct the model.
    Default values create a tiny model (~3 M params) useful for unit tests
    and quick iteration.  Override them for real workloads.

    Attributes:
        vocab_size: Number of tokens in the vocabulary.
        hidden_size: Dimensionality of embeddings and hidden states.
        num_hidden_layers: Number of decoder (transformer) layers.
        num_attention_heads: Number of query heads in multi-head attention.
        num_key_value_heads: Number of key/value heads (GQA when < num_attention_heads).
        intermediate_size: Inner dimensionality of the gated MLP.
        max_position_embeddings: Maximum sequence length the model supports.
        rope_theta: Base frequency for Rotary Position Embeddings.
        rms_norm_epsilon: Small constant for numerical stability in RMSNorm.
        use_cache: Whether to return KV cache by default during forward pass.
        tie_word_embeddings: Whether input and output embeddings share weights.
        attn_implementation: Attention backend — "sdpa" (Flash/memory-efficient via
            PyTorch) or "eager" (explicit matmul→softmax for debugging).
    """

    model_type: str = "readable_lm"

    def __init__(
        self,
        vocab_size: int = 50304,
        hidden_size: int = 256,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 4,
        num_key_value_heads: int = 2,
        intermediate_size: int = 512,
        max_position_embeddings: int = 2048,
        rope_theta: float = 10000.0,
        rms_norm_epsilon: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        attn_implementation: str = "sdpa",
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rms_norm_epsilon = rms_norm_epsilon
        self.use_cache = use_cache
        self.attn_implementation = attn_implementation

        # Validate GQA constraint
        assert num_attention_heads % num_key_value_heads == 0, (
            f"num_attention_heads ({num_attention_heads}) must be divisible by "
            f"num_key_value_heads ({num_key_value_heads})"
        )

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
