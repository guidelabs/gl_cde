from typing import final

from torch import nn
import torch

from .layers import CausalDiffusionLMBlock, RMSNorm, RotaryEmbedding
from .types import CausalDiffusionLMSettings, CDLMForwardArgs


@final
class CausalDiffusionLMBackbone(nn.Module):
    """CausalDiffusionLM model implementation with block-causal attention.

    This is a pure compute graph with no loss logic or training code.
    Can be used by both vanilla and interpretable modules.

    Args:
        config: CausalDiffusionLMSettings instance with model hyperparameters
    """

    def __init__(self, config: CausalDiffusionLMSettings) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        # Token embeddings (no position embeddings - using RoPE instead)
        self.tok_emb = nn.Embedding(self.vocab_size, config.n_embd, dtype=config.torch_dtype)

        # Transformer blocks with block-causal attention
        self.rope = None
        if config.use_rope:
            # use a shared instance of RotaryEmbedding for all blocks
            # to avoid creating multiple instances of the same tensor which cost memory
            self.rope = RotaryEmbedding(config)
        self.blocks = nn.ModuleList(
            [CausalDiffusionLMBlock(config, self.rope) for _ in range(config.n_layers)]
        )

        # Final layer norm
        if config.use_rms_norm:
            self.ln_f: nn.Module = RMSNorm(config)
        else:
            self.ln_f = nn.LayerNorm(config.n_embd, dtype=config.torch_dtype)

        # Output projection
        self.lm_head = nn.Linear(config.n_embd, self.vocab_size, bias=False, dtype=config.torch_dtype)

        # Weight tying (optional)
        if config.weight_sharing:
            self.tok_emb.weight = self.lm_head.weight

    def allocate_kv_cache(self, batch_size: int, max_seq_len: int) -> None:
        """Allocate paged KV cache tensors on every transformer block."""
        for module in self.blocks:
            assert isinstance(module, CausalDiffusionLMBlock)
            module.attn.allocate_kv_cache(batch_size, max_seq_len)

    def release_kv_cache(self) -> None:
        """Drop KV cache tensors on every transformer block."""
        for module in self.blocks:
            assert isinstance(module, CausalDiffusionLMBlock)
            module.attn.release_kv_cache()

    def forward(
        self,
        input_ids: torch.Tensor,
        f_args: CDLMForwardArgs,
        *,
        return_hidden: bool = False,
    ) -> torch.Tensor:
        """Forward pass through CausalDiffusionLM model.

        Args:
            input_ids: Input token indices [B, T] (may contain mask tokens)
            f_args: Contains the following:
                mask: BlockMask for flex attention
                start_index: which position to index into block table
                positions: which positions are being processed
                page: page table object
            return_hidden: Whether to return hidden states before output projection

        Returns:
            logits: Logits over vocabulary [B, T, vocab_size]
            or hidden_states: [B, T, n_embd] if return_hidden=True
        """
        _B, T = input_ids.size()
        assert self.config.block_size >= T, f"Sequence length {T} exceeds block_size {self.config.block_size}"

        # Token embeddings (RoPE handles position encoding in attention layers)
        x = self.tok_emb(input_ids)  # [B, T, n_embd]

        # Forward through transformer blocks
        for layer_idx, block in enumerate(self.blocks):
            x = block(x, f_args)
            # position_injection is always set by prepare_prefill/prepare_decode,
            # never None at runtime.
            # Steering: add per-layer injection buffer. Layers before
            # inject_layer are always zeros (no effect). Layers at and
            # after inject_layer contain the pre-scaled concept direction
            # at masked positions when steering is active.
            if layer_idx >= f_args.inject_layer:
                x = x + f_args.position_injection

        # Final layer norm
        x = self.ln_f(x)

        if return_hidden:
            return x  # [B, T, n_embd]

        return self.lm_head(x)  # [B, T, vocab_size]
