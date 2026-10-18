# Adapted from attention-gym
# Original source: https://github.com/pytorch-labs/attention-gym
# License: BSD 3-Clause (see THIRD_PARTY_LICENSES.md)
# Copyright (c) 2023, Driss Guessous

# the original implementation has some bugs and has some feature that lives outside of the PageTable class

from collections.abc import Callable
from typing import final

from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask
import torch

type MaskModFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


@final
class PageTable:
    """PageTable is a modified version of PagedAttention from attention-gym.

    PageTable improves it by:
    - subsuming the free_batch_idx into the page table, so we don't need to maintain it separately
    """

    def __init__(
        self,
        max_batch_size: int,
        page_size: int,
        max_seq_len: int,
        device: torch.device,
    ) -> None:
        if max_seq_len % page_size != 0:
            raise ValueError(f"max_seq_len ({max_seq_len}) must be divisible by page_size ({page_size})")
        self.max_batch_size = max_batch_size
        self.device = device
        self.page_size = page_size
        self.num_pages_per_obs = max_seq_len // page_size
        self.total_pages = max_batch_size * self.num_pages_per_obs
        self.max_seq_len = max_seq_len

        self.free_batch_idx = list(reversed(range(self.max_batch_size)))
        # allocated set
        self.allocated_set: set[int] = set()
        self.num_reserved_pages = torch.tensor([0 for _ in range(self.max_batch_size)], device=device)
        self.seq_offset_into_cache = torch.arange(
            self.num_pages_per_obs, dtype=torch.int32, device=self.device
        )

    def can_allocate(self) -> bool:
        """Check if we can allocate a batch idx for a new request. Does not use gpu operations.

        Returns:
            bool: True if there is a free batch index.
        """
        return len(self.free_batch_idx) > 0

    def allocate(self) -> int:
        """Allocate a new batch.

        Returns:
            int: Newly allocated batch index.
        """
        if len(self.free_batch_idx) == 0:
            raise RuntimeError("No free slots available, cannot allocate a new batch.")
        batch_idx = self.free_batch_idx.pop()

        if batch_idx < 0 or batch_idx >= self.max_batch_size:
            raise IndexError(f"Cannot allocate out of bounds index {batch_idx}")
        self.allocated_set.add(batch_idx)
        self.num_reserved_pages[batch_idx] = 0
        return batch_idx

    def reserve(self, batch_idx: int, new_offset: int) -> None:
        if batch_idx < 0 or batch_idx >= self.max_batch_size:
            raise IndexError(f"Cannot reserve out of bounds index {batch_idx}")

        if not self.is_idx_allocated(batch_idx):
            raise RuntimeError(f"Batch idx {batch_idx} is not allocated.")

        if new_offset < 0 or new_offset > self.num_pages_per_obs:
            raise ValueError(
                f"Cannot reserve {new_offset} pages for batch idx {batch_idx}; "
                + f"max is {self.num_pages_per_obs}."
            )

        self.num_reserved_pages[batch_idx] = new_offset

    def is_idx_allocated(self, idx: int) -> bool:
        return idx in self.allocated_set

    def erase(self, batch_idx: int) -> None:
        """Removes a single batch from paged attention.

        Args:
            batch_idx (int): batch index to be removed;
        """
        if batch_idx < 0 or batch_idx >= self.max_batch_size:
            raise IndexError(f"Cannot erase out of bounds index {batch_idx}.")
        if not self.is_idx_allocated(batch_idx):
            raise RuntimeError(f"Batch idx {batch_idx} is already freed.")

        self.free_batch_idx.append(batch_idx)
        self.allocated_set.remove(batch_idx)

    def assign_metadata(
        self,
        batch_idx: torch.Tensor,
        input_pos: torch.Tensor,
    ) -> Tensor:
        return (batch_idx[:, None] * self.max_seq_len + input_pos).view(-1)

    def assign(
        self,
        addr: torch.Tensor,
        k_val: torch.Tensor,
        v_val: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Assigns new contents `val` to the storage `cache` at `addr`.

        Args:
            addr (Tensor): cache indices `batch_idx * max_seq_len + pos`,
                from `assign_metadata`; shape `(B*S)`.
            k_val (Tensor): key to be assigned; shape :math:`(B, H, S, D)`
            v_val (Tensor): value to be assigned; shape :math:`(B, H, S, D)`
            k_cache (Tensor): the cache to store the keys; shape:`(1, H, max_batch_size*max_seq_len, D)`
            v_cache (Tensor): the cache to store the values; shape:`(1, H, max_batch_size*max_seq_len, D)`

        Returns:
            tuple[Tensor, Tensor]: The full updated key and value caches.
        """
        if k_val.requires_grad:
            raise RuntimeError("val must not require gradient")

        B, H, S, K_D = k_val.shape
        B_cache, H_cache, CACHE_LEN, D_cache = k_cache.shape
        assert B_cache == 1, "cache must be flattened to a batch dim of 1"
        assert H_cache == H, "number of heads must match"
        assert self.max_batch_size * self.max_seq_len == CACHE_LEN, "cache must span every batch slot"
        assert D_cache == K_D, "hidden dim must match"

        V_D = v_val.shape[3]
        if k_cache.shape[1] != H:
            raise RuntimeError(
                f"Expect val and cache has the same number of heads but got H={H} and H={k_cache.shape[1]}."
            )

        if k_cache.shape[3] != K_D:
            raise RuntimeError(
                f"Expect k_val and k_cache has the same hidden dim but got D={K_D} and D={k_cache.shape[3]}."
            )
        if v_cache.shape[3] != V_D:
            raise RuntimeError(
                f"Expect v_val and v_cache has the same hidden dim but got D={V_D} and D={v_cache.shape[3]}."
            )
        # Keep batch and sequence separate so strided values can be written directly.
        addr_bs = addr.view(B, S)
        k_cache[0, :, addr_bs, :] = k_val.permute(1, 0, 2, 3)
        v_cache[0, :, addr_bs, :] = v_val.permute(1, 0, 2, 3)
        return k_cache, v_cache

    def generate_block_mask(self, seq_length: int, batch_idx: Tensor) -> BlockMask:
        """Generate a blockmask with offsets. Assumes prefill starts at block 0, and query is 1 block.

        Args:
            seq_length (int): Length of the query sequence of each batch element.
            batch_idx (Tensor): Tensor holding indexes of each batch element.

        Returns:
            BlockMask: The decode block mask.
        """
        if seq_length < self.page_size or seq_length % self.page_size != 0:
            raise ValueError(
                f"seq_length ({seq_length}) must be a positive multiple of page_size ({self.page_size})"
            )
        batch_idx = batch_idx.to(torch.int32)
        num_pages = self.num_reserved_pages[batch_idx].to(torch.int32)
        num_blocks = seq_length // self.page_size
        # start pages at zeros in prefill, at num_reserved_pages - 1 in decode.
        start = torch.zeros_like(num_pages) if seq_length > self.page_size else num_pages - 1
        B = len(batch_idx)
        num_blocks_range = torch.arange(num_blocks, dtype=torch.int32, device=self.device)
        full_kv_num_blocks = (num_blocks_range + start[:, None] + 1).view(1, 1, B * num_blocks)
        full_kv_indices = (
            (self.seq_offset_into_cache + (batch_idx * self.num_pages_per_obs)[:, None])
            .repeat_interleave(num_blocks, dim=0)
            .view(1, 1, B * num_blocks, self.num_pages_per_obs)
        )

        # q_idx can range from 0 to batch_size*seq_length
        # k_idx can range from 0 to max_batch_size*max_seq_len
        def page_mask_mod(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, k_idx: torch.Tensor
        ) -> torch.Tensor:
            q_slot = q_idx // seq_length
            q_page = start[q_slot] + (q_idx % seq_length) // self.page_size

            k_slot = k_idx // self.max_seq_len
            k_pos = k_idx % self.max_seq_len
            k_page = k_pos // self.page_size

            slot_eq = batch_idx[q_slot] == k_slot
            can_attend = q_page >= k_page
            return slot_eq & can_attend

        return BlockMask.from_kv_blocks(
            torch.zeros_like(full_kv_num_blocks),
            torch.zeros_like(full_kv_indices),
            full_kv_num_blocks,
            full_kv_indices,
            BLOCK_SIZE=self.page_size,  # safe when page_size = diff_block_size
            mask_mod=page_mask_mod,
            seq_lengths=(B * seq_length, self.max_batch_size * self.max_seq_len),
            compute_q_blocks=False,
        )
