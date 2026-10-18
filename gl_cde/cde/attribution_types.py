from dataclasses import dataclass, field

from torch import Tensor


@dataclass(frozen=True)
class AttributionResult:
    known_indices: Tensor  # (B, T, K_known)
    known_contributions: Tensor  # (B, T, K_known)
    disc_indices: Tensor  # (B, T, K_disc)
    disc_contributions: Tensor  # (B, T, K_disc)
    epsilon: Tensor  # (B, T)


@dataclass(frozen=True)
class ChunkedAttribution:
    concept_scores: list[float] = field(default_factory=list)
    indices: list[int] = field(default_factory=list)
