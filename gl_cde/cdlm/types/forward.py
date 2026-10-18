from dataclasses import dataclass

from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask

from .page_table import PageTableProtocol


@dataclass(frozen=True, eq=False, slots=True)
class CDLMForwardArgs:
    mask: BlockMask
    positions: Tensor
    page: PageTableProtocol | None
    addr: Tensor | None

    # Injection buffers. Tensor of shape [1, T, D].
    # Contain the pre-scaled concept direction at masked positions
    # when steering is active. Always set by prepare_prefill/prepare_decode
    # via SteeringInjector.
    position_injection: Tensor
    # 0-indexed layer where steering injection begins in the forward loop.
    inject_layer: int = 7
