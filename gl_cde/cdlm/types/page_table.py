from typing import Protocol

from torch import Tensor


class PageTableProtocol(Protocol):
    """Paged KV-cache interface required by BlockCausalSelfAttention."""

    def assign(
        self,
        addr: Tensor,
        k_val: Tensor,
        v_val: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Write key/value tensors into the paged cache at ``addr``."""
        ...
