from abc import ABCMeta, abstractmethod
from typing import Self, final

from torch.nn.parameter import Buffer
import torch

from ..configuration_cdlm import CDESettings
from ..sampling_params import SamplingParams


class SamplerForCDLM(torch.nn.Module, metaclass=ABCMeta):
    """Base sampler for CDLM models.

    Defines the interface for sampling tokens from model logits at masked positions.
    Subclasses implement different sampling strategies (e.g., autoregressive, block).
    """

    @classmethod
    @abstractmethod
    def from_cde_config(cls, cde_config: CDESettings) -> Self:
        """Factory constructor from CDESettings.

        Args:
            cde_config: CDESettings instance

        Returns:
            SamplerForCDLM: Sampler built from CDESettings.
        """
        ...

    @abstractmethod
    def forward(
        self,
        og_logits: torch.Tensor,
        input_ids: torch.Tensor,
        sampling_params: SamplingParams,
        token_ids: list[int],
        seed_generator: torch.Generator,
        pos_gpu: torch.Tensor,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample tokens from logits at masked positions.

        Args:
            og_logits: Raw model output logits [B, T, vocab_size]
            input_ids: Input token IDs including mask tokens [B, T]
            sampling_params: Sampling configuration (top_p, repetition_penalty, etc.)
            token_ids: Previously generated token IDs for repetition penalty
            seed_generator: Torch generator for reproducible sampling
            pos_cpu: list of masked positions
            pos_gpu: tensor of masked positions
            input_ids_buf: cpu tensor of input_ids
            index: which sampling step (< block_size)

        Returns:
            Single-element list containing a dict mapping position index (int)
            to sampled token ID (torch.Tensor) for each filled mask position.
            Returns [{}] if no masked positions are found.
        """
        ...


@final
class SFTSamplerForCDLM(SamplerForCDLM):
    def __init__(
        self,
        mask_id: int,
        banned_ids: list[int],
        eos_token_id: int,
        eot_token_id: int | None = None,
    ) -> None:
        super().__init__()
        self.mask_id = mask_id
        self.banned_ids = Buffer(torch.tensor(banned_ids, dtype=torch.long), persistent=False)
        self.eos_token_id_int = eos_token_id
        self.eot_token_id_int = eot_token_id
        self.neg_inf = Buffer(torch.tensor(float("-inf"), dtype=torch.float32), persistent=False)
        self.eos_token_id = Buffer(torch.tensor(eos_token_id, dtype=torch.long), persistent=False)
        self.eot_token_id = Buffer(torch.tensor(eot_token_id, dtype=torch.long), persistent=False)

    @classmethod
    def from_cde_config(cls, cde_config: CDESettings) -> Self:
        if cde_config.eos_token_id is None:
            raise ValueError("eos_token_id must be set for SFTSamplerForCDLM")
        return cls(
            mask_id=cde_config.mask_token_id,
            banned_ids=cde_config.banned_ids,
            eos_token_id=cde_config.eos_token_id,
            eot_token_id=cde_config.eot_token_id,
        )

    def _repeated_token_ids(
        self,
        token_ids: torch.Tensor,
        device: torch.device,
        vocab_size: int,
    ) -> torch.Tensor:
        repeated = token_ids.unique()
        return repeated[(repeated >= 0) & (repeated < vocab_size)]

    def forward(
        self,
        og_logits: torch.Tensor,
        input_ids: torch.Tensor,
        sampling_params: SamplingParams,
        token_ids: list[int],
        seed_generator: torch.Generator,
        pos_gpu: torch.Tensor,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """SFTSamplerForCDLM sampling strategy with some additional bells and whistles.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Sampled token ids and their sequence positions.
        """
        # find all masked positions in the input sequence
        masked_positions = (input_ids == self.mask_id).nonzero(as_tuple=False).squeeze(1)
        if masked_positions.numel() == 0:
            return (
                torch.empty(0, dtype=torch.int64, device=input_ids.device),
                torch.empty(0, dtype=torch.int64, device=input_ids.device),
            )

        all_logits = og_logits[0, masked_positions].float()

        # repetition penalty
        penalty = sampling_params.repetition_penalty
        if penalty != 1.0 and token_ids:
            repeated = self._repeated_token_ids(
                token_ids=input_ids,
                device=all_logits.device,
                vocab_size=all_logits.shape[-1],
            )
            if repeated.numel() > 0:
                values = all_logits.index_select(dim=-1, index=repeated)
                penalized = torch.where(values > 0, values / penalty, values * penalty)
                all_logits.index_copy_(dim=-1, index=repeated, source=penalized)

        # suppress EOS token
        all_logits[..., self.eos_token_id_int] = self.neg_inf

        # suppress banned tokens
        all_logits.index_fill_(
            dim=-1,
            index=self.banned_ids.to(device=all_logits.device, dtype=torch.long),
            value=float("-inf"),
        )

        probs_for_conf = torch.softmax(all_logits, dim=-1)
        confidences = probs_for_conf.max(dim=-1).values
        _, selected_pos_indices = confidences.topk(1)

        token = torch.tensor(0, device=input_ids.device, dtype=torch.long)
        selected_position = masked_positions[:1]

        # tokens_per_step is currently always 1; if this changes,
        # consider vectorizing this loop.
        for pos_index in selected_pos_indices.unbind(0):
            pos_index_1d = pos_index.reshape(1)
            row_logits = all_logits.index_select(dim=0, index=pos_index_1d).squeeze(0)

            # apply temperature and sample
            if sampling_params.use_entropy_sampling:
                pos_probs_raw = torch.softmax(row_logits, dim=-1)
                sorted_probs, _ = torch.sort(pos_probs_raw, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)

                effective_k = torch.clamp(
                    (cumsum <= sampling_params.top_p).sum() + 1,
                    min=2,
                ).to(dtype=row_logits.dtype)

                entropy = -torch.sum(pos_probs_raw * torch.log(pos_probs_raw + 1e-10))
                normalized_entropy = torch.clamp(
                    entropy / torch.log(effective_k),
                    max=1.0,
                )
                temperature = 0.3 + 0.4 * normalized_entropy
                probs = torch.softmax(row_logits / temperature, dim=-1)
                token = self._sample_top_p(probs, sampling_params.top_p, seed_generator).squeeze(0)
            elif sampling_params.top_p > 0.0 and sampling_params.temperature > 0.0:
                probs = torch.softmax(row_logits / sampling_params.temperature, dim=-1)
                token = self._sample_top_p(probs, sampling_params.top_p, seed_generator).squeeze(0)
            else:
                token = torch.argmax(row_logits)

            selected_position = masked_positions.index_select(dim=0, index=pos_index_1d)

            # check if token is EOT/EOS for propagation
            token_is_terminal = (token == self.eot_token_id) | (token == self.eos_token_id)

            if token_is_terminal:
                # propagate EOT by filling everything after this position with EOS
                first_eot_pos = selected_position.squeeze(0)
                positions = torch.arange(input_ids.shape[0], device=input_ids.device, dtype=torch.long)
                eos_positions = positions[positions > first_eot_pos]
                output_tokens = torch.cat(
                    [
                        token.reshape(1),
                        torch.full(
                            (eos_positions.shape[0],),
                            self.eos_token_id_int,
                            device=input_ids.device,
                            dtype=torch.long,
                        ),
                    ]
                )
                return output_tokens, torch.cat([first_eot_pos.reshape(1), eos_positions], dim=0)

        return token.reshape(1), selected_position

    def _sample_top_p(
        self,
        probs: torch.Tensor,
        top_p: float,
        seed_generator: torch.Generator,
    ) -> torch.Tensor:
        """Top-p sampling without CPU sync.

        Returns:
            torch.Tensor: Sampled token index.
        """
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)

        keep = cumulative - sorted_probs <= top_p
        keep[..., 0] = True

        filtered_probs = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
        filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)

        sampled_sorted_idx = torch.multinomial(filtered_probs, num_samples=1, generator=seed_generator)
        return sorted_indices.gather(dim=-1, index=sampled_sorted_idx)
