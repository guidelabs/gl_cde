from functools import partial
from typing import final

from torch import nn
from torch.nn.attention.flex_attention import BlockMask, _dense_to_ordered, flex_attention
from torch.nn.attention.flex_attention import BlockMask as BlockMaskType
from torch.nn.parameter import Buffer
import torch

from ..types import CausalDiffusionLMSettings, CDLMForwardArgs
from .misc import RMSNorm, RotaryEmbedding


def block_causal_mask_mod(
    b: object,
    h: object,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
    *,
    block_size: int,
) -> torch.Tensor:
    """Block-causal mask: causal across blocks, bidirectional within blocks.

    Returns:
        torch.Tensor: Mask values for the given query and key positions.
    """
    return q_idx // block_size >= kv_idx // block_size


def fast_create_block_causal_mask(
    attn_block_size: int,
    seq_length: int,
    mask_block_size: int,
    device: torch.device,
    *,
    calc_q: bool = True,
) -> BlockMaskType:
    """Fast block-causal mask creation for flex_attention.

    Analytically computes the sparse block structure instead of evaluating
    the mask function at every position. This enables flex_attention to use
    its efficient Triton kernel.

    Args:
        attn_block_size: The CausalDiffusionLM diffusion block size (e.g., 64)
        seq_length: Sequence length (e.g., 4096)
        mask_block_size: FlexAttention's internal block size (typically 128)
        device: CUDA device
        calc_q: if gradients can be calculated or not

    Returns:
        BlockMask with properly separated full vs partial blocks
    """
    num_mask_blocks = -(-seq_length // mask_block_size)  # Ceiling division
    attn_blocks_per_mask_block, rem = divmod(mask_block_size, attn_block_size)

    if rem != 0:
        raise ValueError(
            f"mask_block_size ({mask_block_size}) must be divisible by attn_block_size ({attn_block_size})"
        )

    num_attn_blocks = num_mask_blocks * attn_blocks_per_mask_block

    # Block-causal pattern at the attention block level
    lowres_attn_mask = torch.tril(
        torch.ones(num_attn_blocks, num_attn_blocks, dtype=torch.bool, device=device)
    )

    # Count attended positions per mask block
    block_attn_count = (
        lowres_attn_mask.reshape(
            num_mask_blocks, attn_blocks_per_mask_block, num_mask_blocks, attn_blocks_per_mask_block
        )
        .permute(0, 2, 1, 3)
        .sum(dim=[-2, -1])
    )

    max_count = attn_blocks_per_mask_block * attn_blocks_per_mask_block

    # Full blocks: 100% attended (can skip mask computation entirely)
    full_block_mask = block_attn_count == max_count
    if seq_length % mask_block_size > 0:
        full_block_mask[-1, :] = False  # Last row may be partial

    # Normal/partial blocks: need mask_mod evaluation
    normal_block_mask = (block_attn_count > 0) & (~full_block_mask)

    # Convert to sparse ordered format
    kv_num_blocks, kv_indices = _dense_to_ordered(normal_block_mask)
    full_kv_num_blocks, full_kv_indices = _dense_to_ordered(full_block_mask)

    q_num_blocks = None
    q_indices = None
    full_q_num_blocks = None
    full_q_indices = None
    if calc_q:
        q_num_blocks, q_indices = _dense_to_ordered(normal_block_mask.transpose(-2, -1))
        full_q_num_blocks, full_q_indices = _dense_to_ordered(full_block_mask.transpose(-2, -1))
        q_num_blocks = q_num_blocks[None, None, ...]
        q_indices = q_indices[None, None, ...]
        full_q_num_blocks = full_q_num_blocks[None, None, ...]
        full_q_indices = full_q_indices[None, None, ...]
    return BlockMask(
        seq_lengths=(seq_length, seq_length),
        kv_num_blocks=kv_num_blocks[None, None, ...],
        kv_indices=kv_indices[None, None, ...],
        full_kv_num_blocks=full_kv_num_blocks[None, None, ...],
        full_kv_indices=full_kv_indices[None, None, ...],
        q_num_blocks=q_num_blocks,
        q_indices=q_indices,
        full_q_num_blocks=full_q_num_blocks,
        full_q_indices=full_q_indices,
        mask_mod=partial(block_causal_mask_mod, block_size=attn_block_size),
        BLOCK_SIZE=(mask_block_size, mask_block_size),
    )


@torch.compile(
    dynamic=False,
    fullgraph=True,
    options={
        "triton.cudagraphs": True,
        "triton.cudagraph_trees": True,
        "autotune_num_choices_displayed": 1,
    },
)
def wrapper_flex_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: BlockMaskType,
) -> torch.Tensor:
    """Wrapper that uses FlexAttention.

    Returns:
        torch.Tensor: Attention output for the given query, key, and value tensors.
    """
    return flex_attention(
        q,
        k,
        v,
        block_mask=mask,
        enable_gqa=True,
        kernel_options={
            "BACKEND": "TRITON",
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "num_stages": 3,
            "num_warps": 4,
        },
    )


@final
class BlockCausalSelfAttention(nn.Module):
    """Block-causal self-attention with FlexAttention and optional GQA (n_kv_heads)."""

    def __init__(self, config: CausalDiffusionLMSettings, rope: RotaryEmbedding | None = None) -> None:
        super().__init__()

        assert config.n_embd % config.n_head == 0

        self.config = config
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.head_dim

        # GQA configuration
        self.n_kv_heads = config.kv_head_count

        if self.n_kv_heads <= 0:
            raise ValueError(f"n_kv_heads must be >= 1 (got {self.n_kv_heads})")
        if self.n_head % self.n_kv_heads != 0:
            raise ValueError(f"n_head ({self.n_head}) must be divisible by n_kv_heads ({self.n_kv_heads})")

        self.kv_repeat = self.n_head // self.n_kv_heads

        use_bias = config.use_bias

        # Linear projections: Q has n_embd outputs; K and V each have n_kv_heads * head_dim.
        kv_out = self.n_kv_heads * self.head_dim
        attn_out = self.n_embd + 2 * kv_out
        self.c_attn = nn.Linear(config.n_embd, attn_out, bias=use_bias, dtype=config.torch_dtype)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=use_bias, dtype=config.torch_dtype)

        # QK Norm (optional)
        if config.use_qk_norm:
            if config.use_rms_norm:
                self.q_norm: nn.Module | None = RMSNorm(config, size=self.head_dim)
                self.k_norm: nn.Module | None = RMSNorm(config, size=self.head_dim)
            else:
                self.q_norm = nn.LayerNorm(self.head_dim, dtype=config.torch_dtype)
                self.k_norm = nn.LayerNorm(self.head_dim, dtype=config.torch_dtype)
        else:
            self.q_norm = None
            self.k_norm = None
        # Empty placeholders until allocate_kv_cache; on meta, use CPU so
        # Module.to(device) can relocate them (meta tensors cannot be copied).
        cache_device = torch.get_default_device()
        if cache_device.type == "meta":
            cache_device = "cpu"
        self.k_cache = Buffer(
            torch.tensor([], dtype=config.torch_dtype, device=cache_device),
            persistent=False,
        )
        self.v_cache = Buffer(
            torch.tensor([], dtype=config.torch_dtype, device=cache_device),
            persistent=False,
        )
        # convert to a scalar tensor since torch.compile can specialize on float constants
        if config.clip_qkv is not None:
            self.clip_qkv = Buffer(
                torch.tensor(config.clip_qkv, dtype=config.torch_dtype, device=cache_device), persistent=False
            )
        else:
            self.clip_qkv = None

        # Optional rotary position embeddings.
        if rope is not None and not config.use_rope:
            raise ValueError("rope provided but config.use_rope is False")

        self.rope = None
        if config.use_rope:
            self.rope = rope if rope is not None else RotaryEmbedding(config)

    def release_kv_cache(self) -> None:
        """Drop KV cache tensors so GPU memory can be reclaimed."""
        self.k_cache = Buffer(
            torch.tensor([], dtype=self.config.torch_dtype, device=self.k_cache.device), persistent=False
        )
        self.v_cache = Buffer(
            torch.tensor([], dtype=self.config.torch_dtype, device=self.v_cache.device), persistent=False
        )

    def allocate_kv_cache(self, batch_size: int, max_seq_len: int) -> None:
        """Allocate paged KV cache tensors, keeping the current k_cache device."""
        self.k_cache = torch.zeros(
            1,
            self.config.kv_head_count,
            batch_size * max_seq_len,
            self.config.head_dim,
            dtype=self.config.torch_dtype,
            device=self.k_cache.device,
        )
        self.v_cache = torch.zeros(
            1,
            self.config.kv_head_count,
            batch_size * max_seq_len,
            self.config.head_dim,
            dtype=self.config.torch_dtype,
            device=self.v_cache.device,
        )

    def forward(self, x: torch.Tensor, f_args: CDLMForwardArgs) -> torch.Tensor:
        B, T, C = x.size()

        # QKV projection
        qkv = self.c_attn(x)
        if self.clip_qkv is not None:
            qkv = qkv.clamp(min=-self.clip_qkv, max=self.clip_qkv)

        kv_dim = self.n_kv_heads * self.head_dim
        q, k, v = qkv.split([self.n_embd, kv_dim, kv_dim], dim=2)

        # Keep batch and sequence adjacent through normalization and RoPE so paged
        # queries can flatten them without a copy.
        q = q.view(B, T, self.n_head, self.head_dim)  # [B, T, H, D]
        k = k.view(B, T, self.n_kv_heads, self.head_dim)  # [B, T, K, D]
        v = v.view(B, T, self.n_kv_heads, self.head_dim)  # [B, T, K, D]
        # QK Norm
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # RoPE
        if self.rope is not None:
            q, k = self.rope(f_args.positions, q, k)

        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if f_args.addr is not None and f_args.page is not None:
            k, v = f_args.page.assign(f_args.addr, k, v, self.k_cache, self.v_cache)
            q = q.view(B * T, self.n_head, self.head_dim).transpose(0, 1).unsqueeze(0)
        else:
            q = q.transpose(1, 2)

        # Output projection
        y = wrapper_flex_attention(q, k, v, mask=f_args.mask)
        if f_args.addr is not None and f_args.page is not None:
            y = y.view(self.n_head, B, T, self.head_dim).permute(1, 0, 2, 3)
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.c_proj(y)
