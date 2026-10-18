from collections import deque
from typing import final

import torch

from ...cdlm import CausalDiffusionLMSettings, ConceptSettings
from ..configuration_cdlm import CDESettings
from .output_to_concept import (
    chunk_attribution,
    chunk_ranges,
)
from .paged_attention import PageTable
from .sequence import SequenceForDiffusionLM, SequenceStatus


@final
class SchedulerForDiffusionLM:
    def __init__(
        self,
        cde_config: CDESettings,
        model_config: CausalDiffusionLMSettings,
        concept_config: ConceptSettings,
    ) -> None:
        if cde_config.max_num_batched_tokens < model_config.block_size:
            raise ValueError(
                f"max_num_batched_tokens ({cde_config.max_num_batched_tokens}) must be "
                f">= model block_size ({model_config.block_size})"
            )

        self.block_size = cde_config.kvcache_block_size
        max_length = model_config.block_size
        batch_size = cde_config.max_num_seqs
        self.max_length = max_length

        # NOTE: page table allocates cuda memory
        self.page = PageTable(batch_size, self.block_size, max_length, cde_config.device)
        self.max_num_seqs = cde_config.max_num_seqs
        self.max_num_batched_tokens = cde_config.max_num_batched_tokens
        self.eos_token_id = cde_config.eos_token_id
        self.eot_token_id = cde_config.eot_token_id
        self.endofchunk_token_id = cde_config.endofchunk_token_id
        self.top_k_known = concept_config.topk_known
        self.top_k_unk = concept_config.unknown_topk
        self.num_known_concepts = concept_config.n_concepts
        self.waiting: deque[SequenceForDiffusionLM] = deque()
        self.running: deque[SequenceForDiffusionLM] = deque()
        self.diff_block_size = model_config.diff_block_size

    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def add(self, seq: SequenceForDiffusionLM) -> None:
        self.waiting.append(seq)

    def remove(self, seq: SequenceForDiffusionLM) -> None:
        if seq in self.waiting:
            self.waiting.remove(seq)
        if seq in self.running:
            self.running.remove(seq)
            if self.page.is_idx_allocated(seq.allocated_idx):
                self.page.erase(seq.allocated_idx)

    def _allocate_pages(self, seq: SequenceForDiffusionLM) -> None:
        batch_idx_int = self.page.allocate()
        seq.allocate_idx(batch_idx_int)
        # empty prompts have zero blocks but still prefill one masked block
        self.page.reserve(seq.allocated_idx, max(seq.num_blocks, 1))

    def schedule(self) -> tuple[list[SequenceForDiffusionLM], bool]:
        # prefill
        scheduled_seqs: list[SequenceForDiffusionLM] = []
        num_seqs = 0
        num_batched_tokens = 0
        phase = "none"
        while self.waiting and num_seqs < self.max_num_seqs:
            phase = "prefill"
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.page.can_allocate():
                break
            self._allocate_pages(seq)
            num_seqs += 1
            num_batched_tokens += len(seq)
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            phase = "decode"
            seq = self.running[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens:
                break
            seq = self.running.popleft()
            self.page.reserve(seq.allocated_idx, seq.num_blocks)
            num_seqs += 1
            num_batched_tokens += len(seq)
            scheduled_seqs.append(seq)
        if not scheduled_seqs:
            diag = {
                "phase": phase,
                "waiting": len(self.waiting),
                "running": len(self.running),
                "max_num_seqs": self.max_num_seqs,
                "max_num_batched_tokens": self.max_num_batched_tokens,
                "diff_block_size": self.diff_block_size,
                "can_allocate": self.page.can_allocate(),
            }
            candidates = list(self.running)[:3] + list(self.waiting)[:2]
            infos = []
            for j, s in enumerate(candidates):
                infos.append(
                    f"[{j}] status={s.status.name}," + f" len={len(s)}, diff_block={s.diff_block_size}, "
                )
            raise RuntimeError(
                "SchedulerForDiffusionLM: unable to schedule any sequence; "
                + f"state={diag}; details={' | '.join(infos)}"
            )
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def _stop_ids(self, seq: SequenceForDiffusionLM) -> set[int]:
        stop_ids = set(seq.sampling_params.stop_token_ids)
        if self.eot_token_id is not None:
            stop_ids.add(self.eot_token_id)
        return stop_ids

    def check_stop(self, token_id: int, seq: SequenceForDiffusionLM) -> bool:
        if token_id in self._stop_ids(seq):
            return True

        if seq.num_unmasked_tokens >= self.max_length:
            return True

        return (seq.num_unmasked_tokens - seq.num_prompt_tokens) >= seq.sampling_params.max_tokens

    def _truncate_generated_at_stop(self, seq: SequenceForDiffusionLM) -> None:
        stop_ids = self._stop_ids(seq)
        if not stop_ids:
            return
        for i, token_id in enumerate(seq.token_ids[seq.num_prompt_tokens :]):
            if token_id in stop_ids:
                seq.token_ids = seq.token_ids[: seq.num_prompt_tokens + i]
                return

    @torch.no_grad()
    def compute_chunked_attributions(
        self, seq: SequenceForDiffusionLM, *, compute_last_attribution: bool = False
    ) -> None:
        chunk_bounds, prev_token_eos_or_eoc = chunk_ranges(
            seq.start_concept_attr,
            seq.token_ids,
            self.endofchunk_token_id,
            self._stop_ids(seq),
            prev_token_eos_or_eoc=seq.prev_token_eos_or_eoc,
            compute_last_chunk_idx=compute_last_attribution,
        )
        seq.prev_token_eos_or_eoc = prev_token_eos_or_eoc
        seq.reset_chunk_attributions()
        for start, end in chunk_bounds:
            chunk = chunk_attribution(
                seq.attribution_cache, start, end, self.top_k_known, self.top_k_unk, self.num_known_concepts
            )
            seq.add_chunk_attributions(chunk)
            seq.start_concept_attr = end + 1

    def postprocess(
        self,
        seqs: list[SequenceForDiffusionLM],
        sample_output: list[dict[torch.Tensor, torch.Tensor]],
    ) -> list[SequenceForDiffusionLM]:
        stream_ready_seqs: list[SequenceForDiffusionLM] = []
        for seq, output in zip(seqs, sample_output, strict=True):
            match seq.status:
                case SequenceStatus.PREFILL:
                    seq.add_block()
                    seq.status = SequenceStatus.RUNNING
                case SequenceStatus.RUNNING:
                    check_end = False
                    for position, token_id in output.items():
                        token_id = int(token_id)
                        position = int(position)
                        seq.unmask_token(position, token_id)
                        check_end = check_end or self.check_stop(token_id, seq)
                        if (seq.num_unmasked_tokens >= self.max_length) or (
                            seq.num_unmasked_tokens - seq.num_prompt_tokens
                        ) >= seq.sampling_params.max_tokens:
                            break
                    if check_end:
                        if (
                            seq.num_unmasked_tokens - seq.num_prompt_tokens
                        ) >= seq.sampling_params.max_tokens:
                            seq.token_ids = seq.token_ids[
                                : seq.sampling_params.max_tokens + seq.num_prompt_tokens
                            ]
                        self._truncate_generated_at_stop(seq)
                        self.compute_chunked_attributions(seq, compute_last_attribution=True)
                        stream_ready_seqs.append(seq)
                        seq.update_completed_indexes()
                        seq.status = SequenceStatus.FINALIZE
                        self.remove(seq)
                    elif seq.num_tokens == seq.num_unmasked_tokens:
                        self.compute_chunked_attributions(seq)
                        stream_ready_seqs.append(seq)
                        seq.add_block()
                case _:
                    raise ValueError(f"Invalid SequenceStatus {seq.status}")
        return stream_ready_seqs
