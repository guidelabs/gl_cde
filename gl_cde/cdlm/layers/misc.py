from functools import lru_cache
from typing import final

from torch import nn
from torch.nn.parameter import Buffer
import torch
import torch.nn.functional as F

from ..types import CausalDiffusionLMSettings

"""
Transformer building blocks for GPT models.

Sections:
1. Normalization (RMSNorm)
2. Positional Encodings (BufferCache, RotaryEmbedding)
3. Attention (CausalSelfAttention)
4. MLP (MLP with SwiGLU and standard variants)
5. Transformer Block (Block)
"""


@final
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    A simplified LayerNorm that only uses RMS (no mean centering).
    See: https://github.com/allenai/OLMo-core/blob/main/src/olmo_core/nn/layer_norm.py#L188.

    Args:
        config: Model config with n_embd attribute
        size: Optional override for normalization size (defaults to config.n_embd)
    """

    def __init__(self, config: CausalDiffusionLMSettings, size: int | None = None) -> None:
        super().__init__()
        eps_device = torch.get_default_device()
        if eps_device.type == "meta":
            eps_device = "cpu"
        self.eps = Buffer(
            torch.tensor(config.norm_eps, dtype=torch.float32, device=eps_device),
            persistent=False,
        )
        norm_size = size if size is not None else config.n_embd
        self.weight = nn.Parameter(torch.ones(norm_size, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization.

        Args:
            x: Input tensor [..., d_model]

        Returns:
            Normalized tensor with same shape as input
        """
        og = x.dtype
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)

        return (self.weight * x).to(og)


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    # x is [B, T, H, D]; positions may be [T], [1, T], or [B, T].
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    x1, x2 = torch.chunk(x.to(torch.float32), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


@lru_cache(1)
def _cached_rope_cos_sin(
    rotary_dim: int,
    max_position_embeddings: int,
    base: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Build CPU RoPE sin/cos tables; shared across RotaryEmbedding instances.
    with torch.device("cpu"):
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)
        return freqs.cos(), freqs.sin()


@final
class RotaryEmbedding(nn.Module):
    def __init__(self, config: CausalDiffusionLMSettings) -> None:
        super().__init__()
        cos_cache, sin_cache = _cached_rope_cos_sin(config.head_dim, config.block_size, config.rope_base)
        # Clone so .to() only touches this module's buffers, not the lru_cache.
        cos_cache = cos_cache.clone()
        sin_cache = sin_cache.clone()
        # On meta, keep on CPU so Module.to(device) can relocate them.
        cache_device = torch.get_default_device()
        if cache_device.type == "meta":
            cache_device = "cpu"
        self.cos_cache = Buffer(cos_cache.to(device=cache_device), persistent=False)
        self.sin_cache = Buffer(sin_cache.to(device=cache_device), persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to query and key tensors in [B, T, H, D] layout.

        Returns:
            Rotated query and key tensors with their original shapes.
        """
        cos = self.cos_cache[positions]
        sin = self.sin_cache[positions]

        # Reshape using only sizes from the target tensor for Dynamo friendliness
        query_shape = query.shape
        query = apply_rotary_emb(query, cos, sin).view(query_shape)

        key_shape = key.shape
        key = apply_rotary_emb(key, cos, sin).view(key_shape)
        return query, key


@final
class MLP(nn.Module):
    """Multi-Layer Perceptron with support for different activation functions.

    Supports two variants:
    1. SwiGLU: GLU variant with SiLU activation (used in LLaMA, OLMo 2)
    2. Standard: Traditional MLP with configurable activation

    Args:
        config: Model config with attributes:
            - n_embd: Model dimension
            - mlp_ratio: Multiplier for hidden dimension (default: 4)
            - use_bias: Whether to use bias in linear layers
            - mlp_type: "swiglu" or "standard"
            - activation: Activation function for standard MLP ("gelu", "relu", "silu")
    """

    def __init__(self, config: CausalDiffusionLMSettings) -> None:
        super().__init__()

        if config.intermediate_size is not None:
            intermediate_size = config.intermediate_size
        else:
            intermediate_size = config.mlp_ratio * config.n_embd

        use_bias = config.use_bias
        mlp_type = config.mlp_type

        if mlp_type == "swiglu":
            # SwiGLU: needs 2x intermediate size for gating
            self.c_fc = nn.Linear(
                config.n_embd, 2 * intermediate_size, bias=use_bias, dtype=config.torch_dtype
            )
            self.c_proj = nn.Linear(intermediate_size, config.n_embd, bias=use_bias, dtype=config.torch_dtype)
            self.activation = None  # SwiGLU is handled in forward
        else:
            # Standard MLP
            self.c_fc = nn.Linear(config.n_embd, intermediate_size, bias=use_bias, dtype=config.torch_dtype)
            self.c_proj = nn.Linear(intermediate_size, config.n_embd, bias=use_bias, dtype=config.torch_dtype)

            activation_type = config.activation
            if activation_type == "gelu":
                self.activation = nn.GELU(approximate="tanh")
            elif activation_type == "relu":
                self.activation = nn.ReLU()
            elif activation_type == "silu":
                self.activation = nn.SiLU()
            else:
                raise ValueError(f"Unknown activation: {activation_type}")

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply MLP transformation.

        Args:
            x: Input tensor [B, T, n_embd]

        Returns:
            Output tensor [B, T, n_embd]
        """
        mlp_type = self.config.mlp_type

        if mlp_type == "swiglu":
            # SwiGLU: split into gate and up projection
            gate_up = self.c_fc(x)  # [B, T, 2*intermediate_size]
            up, gate = gate_up.chunk(2, dim=-1)  # Each: [B, T, intermediate_size]
            intermediate = F.silu(gate) * up  # Element-wise gating
        else:
            # Standard MLP
            intermediate = self.c_fc(x)
            intermediate = self.activation(intermediate)  # pyright: ignore[reportOptionalCall]

        return self.c_proj(intermediate)
