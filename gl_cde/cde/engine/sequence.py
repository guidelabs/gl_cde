from copy import copy
from dataclasses import dataclass
from enum import Enum, auto
from typing import NewType, final

import torch

from ...cdlm import CausalDiffusionLMSettings, ConceptSettings
from ..attribution_types import ChunkedAttribution
from ..configuration_cdlm import CDESettings
from ..sampling_params import SamplingParams
from .attribution_cache import AttributionCache

RequestIdType = NewType("RequestIdType", int)


@dataclass(frozen=True, eq=False, kw_only=True, slots=True)
class RequestOutput:
    """This may not hold any cuda tensors."""

    request_id: RequestIdType
    token_ids: list[int]
    concepts: list[ChunkedAttribution]
    finished: bool


class SequenceStatus(Enum):
    PREFILL = auto()
    RUNNING = auto()
    FINALIZE = auto()


@final
class SequenceForDiffusionLM:
    """Sequence implementation for Diffusion Language Models."""

    def __init__(
        self,
        seq_id: RequestIdType,
        token_ids: list[int],
        sampling_params: SamplingParams,
        cde_config: CDESettings,
        model_config: CausalDiffusionLMSettings,
        concept_config: ConceptSettings,
    ) -> None:
        self.seq_id = seq_id
        self.status = SequenceStatus.PREFILL
        self.token_ids = copy(token_ids)
        self.num_tokens = len(token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.cde_config = cde_config
        self.mask_token_id = cde_config.mask_token_id
        self.diff_block_size = model_config.diff_block_size
        self.block_size = cde_config.kvcache_block_size
        self.sampling_gen = torch.Generator(device=cde_config.device).manual_seed(sampling_params.seed)
        self.sampling_params = sampling_params
        self.attribution_cache = AttributionCache(
            self.block_size,
            concept_config.topk_known,
            concept_config.unknown_topk,
            concept_config.block_size,
            device=cde_config.device,
        )
        self.chunk_attributions: list[ChunkedAttribution] = []
        self.start_concept_attr = self.num_prompt_tokens
        self.start_comp_blk_idx = len(token_ids)
        self.end_comp_blk_idx = (self.num_prefill_blocks + 1) * (self.block_size)
        self._num_mask_tokens = 0
        self.masked_positions: list[int] = []
        self.prev_token_eos_or_eoc = False
        self.allocated_idx = -1

    def __len__(self) -> int:
        return self.num_tokens

    def __getitem__(self, key: int) -> int:
        return self.token_ids[key]

    def __repr__(self) -> str:
        return (
            f"SequenceForDiffusionLM(block_size={self.block_size}, "
            f"seq_id={self.seq_id}, status={self.status.name}, num_tokens={self.num_tokens}, "
            f"num_prompt_tokens={self.num_prompt_tokens}, "
            f"sampling_params={self.sampling_params}, "
            f"diff_block_size={self.diff_block_size}, "
        )

    @property
    def prompt_token_ids(self) -> list[int]:
        return self.token_ids[: self.num_prompt_tokens]

    @property
    def num_blocks(self) -> int:
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def num_prefill_blocks(self) -> int:
        return self.num_prompt_tokens // self.block_size

    @property
    def num_completed_blocks(self) -> int:
        """Number of blocks which are on the kv cache. This number excludes the active block."""
        return self.num_blocks - 1

    def block(self, i: int) -> list[int]:
        assert 0 <= i < self.num_blocks
        return self.token_ids[i * self.block_size : (i + 1) * self.block_size]

    def update_completed_indexes(self) -> None:
        if self.num_completed_blocks != self.num_prefill_blocks:
            self.start_comp_blk_idx = self.num_completed_blocks * self.block_size
        else:
            self.start_comp_blk_idx = self.num_prompt_tokens
        self.end_comp_blk_idx = (self.num_completed_blocks + 1) * self.block_size
        if (self.num_unmasked_tokens - self.num_prompt_tokens) >= self.sampling_params.max_tokens:
            self.end_comp_blk_idx = self.sampling_params.max_tokens + self.num_prompt_tokens

    def add_block(self) -> None:
        self.update_completed_indexes()
        num_mask_tokens = self.block_size - (self.num_tokens % self.block_size)
        remainder = self.block_size - num_mask_tokens
        self.masked_positions = list(range(remainder, remainder + num_mask_tokens))
        self.token_ids.extend([self.mask_token_id] * num_mask_tokens)
        self.num_tokens += num_mask_tokens
        self._num_mask_tokens += num_mask_tokens

    def allocate_idx(self, idx: int) -> None:
        self.allocated_idx = idx

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)
        self.num_tokens += 1
        if token_id == self.mask_token_id:
            self._num_mask_tokens += 1

    def unmask_token(self, position: int, token_id: int) -> None:
        if position < 0 or position >= self.block_size:
            raise IndexError(f"Position {position} is out of bounds")
        position += self.block_size * self.num_completed_blocks
        if position >= self.num_tokens:
            raise IndexError(f"Position {position} is out of bounds of len(token_ids) {self.num_tokens}")
        if self.token_ids[position] == self.mask_token_id:
            self._num_mask_tokens -= 1
            self.masked_positions.remove(position - self.block_size * self.num_completed_blocks)
        self.token_ids[position] = token_id

    def last_completed_block(self) -> list[int]:
        return self.token_ids[self.start_comp_blk_idx : self.end_comp_blk_idx]

    @property
    def num_generated_tokens(self) -> int:
        return self.num_tokens - self._num_mask_tokens - self.num_prompt_tokens

    @property
    def num_unmasked_tokens(self) -> int:
        return self.num_tokens - self._num_mask_tokens

    @property
    def num_mask_tokens(self) -> int:
        return self._num_mask_tokens

    def add_chunk_attributions(self, chunk: ChunkedAttribution) -> None:
        self.chunk_attributions.append(chunk)

    def reset_chunk_attributions(self) -> None:
        self.chunk_attributions = []

    @property
    def completion_token_ids(self) -> list[int]:
        return self.token_ids[self.num_prompt_tokens :]

    def request_output(self) -> RequestOutput:
        return RequestOutput(
            request_id=self.seq_id,
            token_ids=self.last_completed_block(),
            concepts=self.chunk_attributions.copy(),
            finished=self.status == SequenceStatus.FINALIZE,
        )
