"""Concept steering injection.

Steers generation toward (positive mai_lm_target) or away from (negative
mai_lm_target) target concepts by injecting a pre-scaled direction vector
into hidden states.

This injector initalizes buffers during model_runner.prepare_prefill and
model_runner.prepare_decode. Layers before inject_layer do not add the
buffer to hidden representations. Layers after inject_layer do add
the buffer to hidden representations.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self, final
import math

from torch import Tensor
import torch

from ...cdlm import InterpretableCausalDiffusionLMBackbone
from ..configuration_cdlm import CDESettings
from ..sampling_params import SamplingParams


@dataclass(frozen=True, slots=True)
class SteeringState:
    """Steering state for one request.

    Attributes:
        scaled_direction: Tensor of shape [D], concept direction pre-scaled by alpha.
        For multi-concept, this is the normalized sum of embeddings scaled
        by mai_lm_target / peak_lm.
        starting_num_masked: int, masked positions at start of this decode block.
        tokens_committed_before: int, tokens committed before this decode block.
        cutoff_tokens: int, Committed generation tokens at full alpha before injection stops.
    """

    scaled_direction: Tensor
    starting_num_masked: int
    tokens_committed_before: int
    cutoff_tokens: int


@dataclass(frozen=True, slots=True)
class BatchedSteeringState:
    """Steering state for multiple requests. See SteeringState for documentation on each attribute.

    Attributes:
        scaled_direction: Tensor of shape [B, D]
        starting_num_masked: Tensor [B]
        tokens_committed_before: Tensor [B]
        cutoff_tokens: Tensor [B]
    """

    scaled_direction: Tensor
    starting_num_masked: Tensor
    tokens_committed_before: Tensor
    cutoff_tokens: Tensor

    @classmethod
    def from_states(cls, states: Sequence[SteeringState], device: torch.device) -> Self:
        """Batch per-request SteeringStates into tensors on device.

        Args:
            states: List of SteeringStates of size B.
            device: Device to place the batched tensors on.

        Returns:
            batched_state: BatchedSteeringState
        """
        starting_num_masked = torch.tensor([s.starting_num_masked for s in states], dtype=torch.int64).to(
            device=device, non_blocking=True
        )  # [B]
        tokens_committed_before = torch.tensor(
            [s.tokens_committed_before for s in states], dtype=torch.int64
        ).to(device=device, non_blocking=True)  # [B]
        cutoff_tokens = torch.tensor([s.cutoff_tokens for s in states], dtype=torch.int64).to(
            device=device, non_blocking=True
        )  # [B]
        scaled_direction = torch.stack([s.scaled_direction for s in states])  # [B, D]
        return cls(
            scaled_direction=scaled_direction,
            starting_num_masked=starting_num_masked,
            tokens_committed_before=tokens_committed_before,
            cutoff_tokens=cutoff_tokens,
        )


@final
class SteeringInjector:
    """Concept steering injector.

    Allocates buffers and generates per request steering state when preparing prefill
    and decode. The allocated buffer is updated every diffusion sample step.

    Args:
        cde_config: Model configuration.
        model: The model (passed before torch.compile in ModelRunnerForDiffusionLM).
    """

    def __init__(
        self,
        cde_config: CDESettings,
        model: InterpretableCausalDiffusionLMBackbone,
    ) -> None:
        """Steering injector for concept steering.

        Args:
            cde_config: CDE configuration.
            model: The model.
        """
        n_layers = model.model_config.n_layers
        if not 0 <= cde_config.inject_layer < n_layers:
            raise ValueError(f"inject_layer must be in [0, {n_layers - 1}], got {cde_config.inject_layer}")

        self.cde_config = cde_config
        self.mask_token_id = cde_config.mask_token_id
        self._backbone = model
        self._model_config = model.model_config
        self._dtype = self._model_config.torch_dtype
        self._device = cde_config.device
        self._d_model = self._model_config.n_embd
        # torch.where with out= needs tensor operands; reused by update_buffer every step.
        self._zero = torch.zeros((), dtype=self._dtype, device=self._device)

    def init_prefill(self, batch_size: int, seq_len: int) -> Tensor:
        """Return zero initalized steering buffer for prefill phase.

        Args:
            batch_size: int specifing number of batch elements
            seq_len: int specifying the length of the steering buffer

        Returns:
            buffer: Tensor of shape [B, prefill_length, D]
        """
        return torch.zeros(
            (batch_size, seq_len, self._d_model),
            dtype=self._dtype,
            device=self._device,
        )

    def init_decode(self, batch_size: int) -> Tensor:
        """Return zero initalized steering buffer to be modified for decode phase.

        Args:
            batch_size: int specifing number of batch elements

        Returns:
            buffer: Tensor of shape [B, decode_length, D]
        """
        return torch.zeros(
            (batch_size, self._model_config.diff_block_size, self._d_model),
            dtype=self._dtype,
            device=self._device,
        )

    @torch.no_grad()
    def prepare_steer_state(
        self, sampling_params: SamplingParams, num_mask_tokens: int, num_generated_tokens: int
    ) -> SteeringState:
        """Package steering state.

        This function calculates steering vectors needed for decode phase
        using sampling params. Both steering vectors and various sequence metadata
        are packaged into a SteeringState type.

        Args:
            sampling_params: SamplingParams which contains steering parameters.
            num_mask_tokens: int specifing number of mask tokens.
            num_generated_tokens: int specifying number of tokens generated thus far.

        Returns:
            steer_state: SteeringState
        """
        if not sampling_params.concept_ids:
            return SteeringState(
                scaled_direction=torch.zeros(self._d_model, dtype=self._dtype, device=self._device),
                starting_num_masked=num_mask_tokens,
                cutoff_tokens=sampling_params.steering_cutoff_tokens,
                tokens_committed_before=num_generated_tokens,
            )

        concept_ids = sampling_params.concept_ids
        mai_lm_target = sampling_params.mai_lm_target

        if mai_lm_target is None:
            raise ValueError("mai_lm_target must be set when concept_ids is provided.")
        if not math.isfinite(mai_lm_target):
            raise ValueError(f"mai_lm_target must be finite, got {mai_lm_target}")

        direction, peak = self.get_direction_and_peak(concept_ids)
        # is there anyway we could improve this?
        alpha = float(mai_lm_target) / peak
        scaled_direction = (direction * alpha).to(dtype=self._dtype, device=self._device)

        return SteeringState(
            scaled_direction=scaled_direction,
            starting_num_masked=num_mask_tokens,
            cutoff_tokens=sampling_params.steering_cutoff_tokens,
            tokens_committed_before=num_generated_tokens,
        )

    @torch.inference_mode()
    def get_direction_and_peak(self, concept_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Gets directional steering vector and peak value.

        Args:
            concept_ids: list[int] list of concept ids.

        Returns:
            direction: Tensor of shape [D]. Sum of embedding vectors.
            peak: Tensor of shape [1]. max value of direction after multiplication with lm_head.
        """
        n_known = self._backbone.known_head.n_concepts
        known_ids = [c for c in concept_ids if c < n_known]
        unknown_ids = [c - n_known for c in concept_ids if c >= n_known]
        vec = torch.zeros(self._d_model, dtype=self._dtype, device=self._device)
        if known_ids:
            known_ids_t = torch.tensor(known_ids, dtype=torch.long).to(device=self._device, non_blocking=True)
            known_vecs = self._backbone.known_head._get_embedding(known_ids_t).float()
            vec += known_vecs.sum(dim=0)

        if unknown_ids:
            unknown_ids_t = torch.tensor(unknown_ids, dtype=torch.long).to(
                device=self._device, non_blocking=True
            )
            unknown_vecs = self._backbone.unknown_head._get_embedding(unknown_ids_t).float()
            vec += unknown_vecs.sum(dim=0)
        direction = vec / vec.norm(p=2).clamp_min(1e-12)
        lm_w = self._backbone.transformer.lm_head.weight

        # Exact, but still O(vocab_size * hidden_dim)
        peak = torch.matmul(lm_w, direction.to(lm_w.dtype)).amax()
        return direction, peak

    @torch.no_grad()
    def update_buffer(
        self, buffer: Tensor, input_ids: Tensor, steering_states: BatchedSteeringState | None
    ) -> None:
        """Zeros then fills injection buffers with scaled direction at masked positions.

        This function first zeros the buffer to clear stale values from the previous
        step. update_buffer then stores in buffer the steering direction on positions
        which contain the mask token in input ids.

        This function is called before every forward pass in the diffusion loop, as
        the masked positions change in input_ids every sampling step. Applies
        hard cutoff steering: full alpha until cutoff_tokens number of generated tokens
        are committed globally, then zero for the remainder.

        Args:
            buffer: Tensor for steering with shape [B, T, D]
            input_ids: Tensor of current token ids with shape [B, T]
            steering_states: BatchedSteeringState of batch size B. None during prefill (no-op).
        """
        if steering_states is None:
            return
        is_masked = input_ids == self.mask_token_id  # [B, T]
        masked_remaining = is_masked.sum(-1)  # [B]
        injecting = (
            steering_states.tokens_committed_before + steering_states.starting_num_masked - masked_remaining
        ) < steering_states.cutoff_tokens  # [B]
        selection = is_masked & injecting[:, None]  # [B, T]
        torch.where(
            selection[:, :, None], steering_states.scaled_direction[:, None, :], self._zero, out=buffer
        )  # [B, T, 1] x [B, 1, D] -> [B, T, D]
