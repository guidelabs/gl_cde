from dataclasses import dataclass, field

from torch import Tensor
import torch


@dataclass
class InterpretableOutput:
    """Output from ConceptHead for Concept Attribution.

    Attributes:
        known_topk_indices: Top-k concept indices (B, T, k) or None
        known_topk_logits: Logits for top-k concepts (B, T, k) or None

        unknown_topk_indices: Top-k concept indices (B, T, k) or None
        unknown_topk_logits: Logits for top-k concepts (B, T, k) or None

        composed: Final composed features for LM head (B, T, D)
    """

    # Top-k concept outputs
    known_topk_indices: Tensor = field(default_factory=lambda: torch.tensor([], dtype=torch.int))
    known_topk_logits: Tensor = field(default_factory=lambda: torch.tensor([], dtype=torch.float))

    # Top-k concept outputs
    unknown_topk_indices: Tensor = field(default_factory=lambda: torch.tensor([], dtype=torch.int))
    unknown_topk_logits: Tensor = field(default_factory=lambda: torch.tensor([], dtype=torch.float))

    # Composed output
    logits: Tensor = field(default_factory=lambda: torch.tensor([], dtype=torch.float))
    composed: Tensor = field(default_factory=lambda: torch.tensor([], dtype=torch.float))
