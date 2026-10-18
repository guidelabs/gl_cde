"""Interpretable CausalDiffusionLM model with concept decomposition."""

from pathlib import Path
from typing import final
import asyncio

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import nn
import torch

from .._loading import cast_state_dict_to_meta
from ..loadable import LoadableModelAbstract
from .causal_diffusion_lm_backbone import CausalDiffusionLMBackbone
from .layers import ConceptHead, ConceptHeadOutput
from .types import CausalDiffusionLMSettings, CDLMForwardArgs, ConceptSettings, InterpretableOutput


@final
class InterpretableCausalDiffusionLMBackbone(nn.Module, LoadableModelAbstract[Path]):
    """Interpretable CausalDiffusionLM backbone with concept decomposition head.

    This backbone wraps a standard CausalDiffusionLM transformer and adds:
    - Known concept head: Predicts known concepts, supports teacher forcing
    - Unknown concept head: Learns residual features (optional)

    Memory optimizations:
    - Streaming concept features (no (B, T, C) allocation)
    - Minimal training output option
    - Sparse top-k for known concepts

    The concept heads are NOT sharded by FSDP (only transformer blocks are),
    so direct weight access works correctly.

    Args:
        concept_config: Experimental concept configs
        model_config: CausalDiffusionLM model configs
    """

    def __init__(
        self,
        concept_config: ConceptSettings,
        model_config: CausalDiffusionLMSettings,
    ) -> None:
        super().__init__()
        self.concept_config = concept_config
        self.model_config = model_config

        # Create base transformer (will be sharded by FSDP)
        # CausalDiffusionLMBackbone expects CausalDiffusionLMSettings, not CDESettings
        if not isinstance(model_config, CausalDiffusionLMSettings):
            raise TypeError(
                "InterpretableCausalDiffusionLMBackbone requires config.model to be CausalDiffusionLMSettings"
            )
        self.transformer = CausalDiffusionLMBackbone(model_config)

        # Known concept head (NOT sharded - stays on each rank fully)
        self.known_head = ConceptHead(
            n_concepts=self.concept_config.n_concepts,
            concept_dim=self.concept_config.concept_dim,
            n_embd=model_config.n_embd,
            is_unknown=False,
            topk=self.concept_config.topk_known,
            block_size=self.concept_config.block_size,
            pad_multiple=self.concept_config.pad_multiple,
        )

        if self.concept_config.n_unknown_concepts is None:
            raise ValueError("n_unknown_concepts must be set")
        self.unknown_head = ConceptHead(
            n_concepts=self.concept_config.n_unknown_concepts,
            concept_dim=self.concept_config.concept_dim,
            n_embd=model_config.n_embd,
            is_unknown=True,
            topk=self.concept_config.unknown_topk,
            block_size=self.concept_config.block_size,
            pad_multiple=self.concept_config.pad_multiple,
            # Factorization params
            factorize=self.concept_config.factorize_unknown,
            factorize_rank=self.concept_config.factorize_rank,
        )

    async def prepare(self) -> Path:
        path = await asyncio.to_thread(
            hf_hub_download, "guidelabs/steerling-8b-instruct", "model.safetensors"
        )
        return Path(path)

    def materialize(self, prepared: Path, device: torch.device) -> None:
        state_dict = cast_state_dict_to_meta(
            load_file(prepared, device=str(device), backend="pread"),
            self.state_dict(),
        )
        self.load_state_dict(state_dict, assign=True, strict=True)
        self.to(device)

    def forward(
        self,
        input_ids: torch.Tensor,
        f_args: CDLMForwardArgs,
        *,
        return_hidden: bool = False,
    ) -> InterpretableOutput:
        """Forward pass with concept decomposition.

        Args:
            input_ids: Token IDs (B, T)
            f_args: Contains the following:
                mask: BlockMask for flex attention
                start_index: which position to index into block table
                positions: which positions are being processed
                page: page table object
            return_hidden: Whether to return hidden states before output projection

        Returns:
            logits: LM logits (B, T, V)
            outputs: hidden states and topk logits/indices
        """
        hidden = self.transformer(input_ids, f_args, return_hidden=True)  # (B, T, D)
        hidden_dtype = hidden.dtype
        hidden_float = hidden.float()
        known_out: ConceptHeadOutput = self.known_head(hidden_float)
        known_features = known_out.features.to(hidden_dtype)

        # Residual unknown (always defined)
        unk = hidden - known_features
        unk_for_lm: torch.Tensor = unk
        unknown_out = self.unknown_head(hidden_float)
        # After this assignment, unk_hat is a torch.Tensor
        unk_hat = unknown_out.features.to(hidden_dtype)
        unk_hat_t: torch.Tensor = unk_hat
        unk_for_lm = unk_hat_t

        epsilon = hidden - (unk_for_lm + known_features)
        unk_for_lm = unk_for_lm + epsilon

        composed = unk_for_lm + known_features
        logits = torch.tensor([], dtype=torch.float) if return_hidden else self.transformer.lm_head(composed)

        return InterpretableOutput(
            known_topk_indices=known_out.topk_indices,
            known_topk_logits=known_out.topk_logits,
            unknown_topk_indices=unknown_out.topk_indices,
            unknown_topk_logits=unknown_out.topk_logits,
            logits=logits.float(),
            composed=composed,
        )
