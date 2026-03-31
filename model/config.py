"""
model/config.py
---------------
Configuration for the SLM model family.

Registers with HuggingFace AutoConfig so the model can be loaded with:
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained("tohio/slm-125m")

Architecture: dense decoder-only transformer
    - RoPE positional embeddings
    - RMSNorm
    - SwiGLU activation
    - Grouped Query Attention (GQA)
    - No bias
    - Tied input/output embeddings
"""

from transformers import PretrainedConfig


class SLMConfig(PretrainedConfig):
    """
    Configuration for the SLM model family.

    Args:
        vocab_size (int): Vocabulary size. Default: 32000.
        hidden_size (int): Dimension of the hidden states. Default: 768.
        intermediate_size (int): Dimension of the SwiGLU FFN inner layer.
            If None, defaults to int(8/3 * hidden_size) rounded to nearest
            multiple of 256 — following LLaMA's formula.
        num_hidden_layers (int): Number of transformer blocks. Default: 12.
        num_attention_heads (int): Number of query attention heads. Default: 12.
        num_key_value_heads (int): Number of key/value heads for GQA.
            Must divide num_attention_heads evenly. Default: 4.
        max_position_embeddings (int): Maximum sequence length. Default: 2048.
        rope_theta (float): Base period for RoPE. Default: 10000.0.
        rope_scaling (dict | None): Optional RoPE scaling config for context
            extension (e.g. YaRN, linear). Default: None.
        rms_norm_eps (float): Epsilon for RMSNorm. Default: 1e-5.
        initializer_range (float): Std for weight initialization. Default: 0.02.
        tie_word_embeddings (bool): Tie input/output embeddings. Default: True.
        use_cache (bool): Whether to use KV cache during generation. Default: True.
        pad_token_id (int | None): Padding token ID. Default: 0.
        bos_token_id (int): Beginning-of-sequence token ID. Default: 2.
        eos_token_id (int): End-of-sequence token ID. Default: 3.
        attention_dropout (float): Dropout on attention weights. Default: 0.0.
        hidden_dropout (float): Dropout on hidden states. Default: 0.0.

    Example usage::

        # 125M config
        config = SLMConfig(
            vocab_size=32000,
            hidden_size=768,
            intermediate_size=2048,
            num_hidden_layers=12,
            num_attention_heads=12,
            num_key_value_heads=4,
            max_position_embeddings=2048,
        )

        # 350M config
        config = SLMConfig(
            vocab_size=32000,
            hidden_size=1024,
            intermediate_size=2816,
            num_hidden_layers=24,
            num_attention_heads=16,
            num_key_value_heads=8,
            max_position_embeddings=2048,
        )

        # 1B config
        config = SLMConfig(
            vocab_size=32000,
            hidden_size=2048,
            intermediate_size=5632,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            max_position_embeddings=4096,
            rope_theta=500000.0,  # extended base for longer context
        )
    """

    model_type = "slm"

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 768,
        intermediate_size: int | None = None,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        num_key_value_heads: int = 4,
        max_position_embeddings: int = 2048,
        rope_theta: float = 10000.0,
        rope_scaling: dict | None = None,
        rms_norm_eps: float = 1e-5,
        initializer_range: float = 0.02,
        tie_word_embeddings: bool = True,
        use_cache: bool = True,
        pad_token_id: int | None = 0,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size or _default_intermediate_size(hidden_size)
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout

        self._validate()

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _validate(self) -> None:
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads})"
            )
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        return self.hidden_size // self.num_attention_heads

    @property
    def num_query_groups(self) -> int:
        """Number of query heads per KV head (GQA groups)."""
        return self.num_attention_heads // self.num_key_value_heads


def _default_intermediate_size(hidden_size: int) -> int:
    """
    Compute SwiGLU FFN intermediate size following LLaMA's formula:
        intermediate = round(8/3 * hidden_size) rounded to nearest 256.

    The factor of 8/3 comes from SwiGLU's gating mechanism — the actual
    parameter count matches a standard FFN with 4x expansion.
    """
    raw = int(8 / 3 * hidden_size)
    return (raw + 255) // 256 * 256


# ── Predefined configs for the three model tiers ──────────────────────────────

SLM_125M = SLMConfig(
    vocab_size=32000,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    num_key_value_heads=4,
    max_position_embeddings=2048,
    rope_theta=10000.0,
)

SLM_350M = SLMConfig(
    vocab_size=32000,
    hidden_size=1024,
    num_hidden_layers=24,
    num_attention_heads=16,
    num_key_value_heads=8,
    max_position_embeddings=2048,
    rope_theta=10000.0,
)

SLM_1B = SLMConfig(
    vocab_size=32000,
    hidden_size=2048,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=8,
    max_position_embeddings=4096,
    rope_theta=500000.0,
)

CONFIGS = {
    "125m": SLM_125M,
    "350m": SLM_350M,
    "1b": SLM_1B,
}