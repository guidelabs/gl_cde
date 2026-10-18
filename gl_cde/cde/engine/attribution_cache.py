from typing import final

import torch

from ..attribution_types import AttributionResult


@final
class AttributionCache:
    def __init__(
        self,
        block_size: int,
        topk_known: int,
        topk_unknown: int,
        total_length: int,
        device: torch.device,
    ) -> None:
        self.block_size = block_size
        self.known_indices = torch.zeros((1, total_length, topk_known), device=device, dtype=torch.long)
        self.known_contributions = torch.zeros(
            (1, total_length, topk_known), device=device, dtype=torch.float
        )
        self.disc_indices = torch.zeros((1, total_length, topk_unknown), device=device, dtype=torch.long)
        self.disc_contributions = torch.zeros(
            (1, total_length, topk_unknown), device=device, dtype=torch.float
        )
        self.epsilon = torch.zeros((1, total_length), device=device, dtype=torch.float)

    def cache_write(self, interp_output: AttributionResult, seq_pos: torch.Tensor) -> None:
        self.known_indices[:, seq_pos] = interp_output.known_indices
        self.known_contributions[:, seq_pos] = interp_output.known_contributions
        self.disc_indices[:, seq_pos] = interp_output.disc_indices
        self.disc_contributions[:, seq_pos] = interp_output.disc_contributions
        self.epsilon[:, seq_pos] = interp_output.epsilon

    def index_cache(
        self, start: int, end: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        k_idx = self.known_indices[0, start:end]  # (T, K_known)
        k_c = self.known_contributions[0, start:end]  # (T, K_known)
        d_idx = self.disc_indices[0, start:end]  # (T, K_disc)
        d_c = self.disc_contributions[0, start:end]  # (T, K_disc)
        eps = self.epsilon[0, start:end]  # (T,)
        return k_idx, k_c, d_idx, d_c, eps
