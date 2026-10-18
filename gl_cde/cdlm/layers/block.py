from typing import final

from torch import nn
import torch

from ..types import CausalDiffusionLMSettings, CDLMForwardArgs
from .attention import BlockCausalSelfAttention
from .misc import MLP, RMSNorm, RotaryEmbedding


@final
class CausalDiffusionLMBlock(nn.Module):
    """Transformer block for CausalDiffusionLM."""

    def __init__(self, config: CausalDiffusionLMSettings, rope: RotaryEmbedding | None = None) -> None:
        super().__init__()

        use_rms_norm = config.use_rms_norm
        if use_rms_norm:
            self.ln_1: nn.Module = RMSNorm(config)
            self.ln_2: nn.Module = RMSNorm(config)
        else:
            self.ln_1 = nn.LayerNorm(config.n_embd, dtype=config.torch_dtype)
            self.ln_2 = nn.LayerNorm(config.n_embd, dtype=config.torch_dtype)

        self.norm_order = config.norm_order
        self.attn = BlockCausalSelfAttention(config, rope)
        self.mlp = MLP(config)

    @torch.compiler.nested_compile_region
    def forward(self, x: torch.Tensor, f_args: CDLMForwardArgs) -> torch.Tensor:  # pyright: ignore[reportIncompatibleMethodOverride]
        if self.norm_order == "pre":
            x = x + self.attn(self.ln_1(x), f_args)
            x = x + self.mlp(self.ln_2(x))
        else:
            x = x + self.ln_1(self.attn(x, f_args))
            x = x + self.ln_2(self.mlp(x))

        return x
