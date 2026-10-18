from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True, eq=False, slots=True)
class ConceptSettings:
    """Configuration for interpretable concept decomposition."""

    n_concepts: int
    n_unknown_concepts: int | None = None
    # Embedding configuration
    max_concepts: int = 32
    concept_dim: int = 768
    topk_known: int = 32
    # unknown factorization
    factorize_unknown: bool = False
    factorize_rank: int = 256
    unknown_topk: int = 128

    # ConceptHead parameters
    block_size: int = 8192
    pad_multiple: int = 16

    def __post_init__(self) -> None:
        """Set derived defaults and validate."""
        if self.factorize_rank < 16 or self.factorize_rank > 1024:
            raise ValueError("factorize_rank must be > 16 and < 1024")

        # Validate topk_known doesn't exceed max_concepts
        if self.topk_known > self.max_concepts:
            raise ValueError(
                f"topk_known ({self.topk_known}) cannot exceed max_concepts ({self.max_concepts})"
            )


@dataclass(frozen=True, eq=False, slots=True)
class CausalDiffusionLMSettings:
    model_type: Literal["blockllada"] = "blockllada"

    # Architecture
    vocab_size: int = 100281
    n_layers: int = 12
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 4096
    n_kv_heads: int | None = None  # None => n_head (regular MHA) GQA

    # CausalDiffusionLM specific
    diff_block_size: int = 64  # attention block size

    # Normalization
    use_rms_norm: bool = True
    norm_eps: float = 1e-5
    norm_order: Literal["pre", "post"] = "post"
    use_qk_norm: bool = True

    # Position encoding
    use_rope: bool = True
    rope_base: float = 500000.0
    rope_full_precision: bool = True

    # MLP
    mlp_type: Literal["swiglu", "standard"] = "swiglu"
    activation: Literal["gelu", "relu", "silu"] = "gelu"
    mlp_ratio: int = 4
    intermediate_size: int | None = None

    # Training
    use_bias: bool = False
    clip_qkv: float | None = 10.0
    dropout: float = 0.0
    weight_sharing: bool = False

    torch_dtype: torch.dtype = torch.bfloat16

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def kv_head_count(self) -> int:
        """Resolved KV head count; None n_kv_heads means full MHA (n_head)."""
        return self.n_head if self.n_kv_heads is None else self.n_kv_heads

    def __post_init__(self) -> None:
        """Validate model configuration."""
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")

        if self.n_kv_heads is not None:
            if self.n_kv_heads <= 0:
                raise ValueError(f"n_kv_heads ({self.n_kv_heads}) must be >= 1")
            if self.n_head % self.n_kv_heads != 0:
                raise ValueError(
                    f"n_head ({self.n_head}) must be divisible by n_kv_heads ({self.n_kv_heads})"
                )
