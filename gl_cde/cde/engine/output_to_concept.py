from typing import Self, final

from torch import Tensor
import torch
import torch.nn.functional as F

from ...cdlm import InterpretableCausalDiffusionLMBackbone, InterpretableOutput
from ...cdlm.layers import ConceptHead
from ..attribution_types import AttributionResult, ChunkedAttribution
from .attribution_cache import AttributionCache


@final
class Attributor:
    """Holds ConceptHeads and language modeling head weights."""

    def __init__(
        self,
        known_head: ConceptHead,
        unknown_head: ConceptHead,
        lm_head: torch.nn.Linear,
    ) -> None:
        """Initalizes modules needed for attribution.

        Args:
        known_head: Known concept head (for embeddings).
        unknown_head: Unknown concept head (for embeddings), or None.
        lm_head: [V, D] LM head weight matrix.
        """
        super().__init__()
        self.known_head = known_head
        self.unknown_head = unknown_head
        self.lm_head = lm_head

    def output_weights_for_tokens(self, token_ids: Tensor) -> Tensor:
        """Return LM head weight rows for the given token ids."""
        return F.embedding(token_ids, self.lm_head.weight).float()

    @classmethod
    def from_backbone(cls, backbone: InterpretableCausalDiffusionLMBackbone) -> Self:
        """Factory constructor from InterpretableCausalDiffusionLMBackbone.

        Args:
            backbone: InterpretableCausalDiffusionLMBackbone instance

        Returns:
            Attributor: Attributor built from backbone heads and LM weights.
        """
        return cls(
            known_head=backbone.known_head,
            unknown_head=backbone.unknown_head,
            lm_head=backbone.transformer.lm_head,
        )

    def _known_head_contrib(
        self,
        W_y: Tensor,
        topk_idx: Tensor,
        topk_logits: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute contribution from a dense concept head's sparse top-k.

        Returns:
            tuple[Tensor, Tensor]: Top-k concept indices and weighted dot-product contributions.
        """
        # P: Token positions (Batch * Num_Tokens), k: num top k, D: hidden dim
        emb = self.known_head._get_embedding(topk_idx).float()  # (P, k, D)
        dots = torch.einsum("pkd,pd->pk", emb, W_y)  # (P, k)
        w = torch.sigmoid(topk_logits.float())  # (P, k)
        return topk_idx, w * dots

    def _unknown_head_contrib(
        self,
        W_y: Tensor,
        topk_idx: Tensor,
        topk_logits: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute contribution from a factorized concept head's sparse top-k.

        Returns:
            tuple[Tensor, Tensor]: Top-k concept indices and weighted dot-product contributions.
        """
        # P: Token positions (Batch * Num_Tokens), k: num top k, D: hidden dim, r: low rank
        assert self.unknown_head.embedding_coef is not None
        assert self.unknown_head.embedding_basis is not None
        W_r = W_y @ self.unknown_head.embedding_basis.weight  # (P, r)
        emd_r = self.unknown_head.embedding_coef(topk_idx).float()  # (P, k, r)
        dots = torch.einsum("pkr,pr->pk", emd_r, W_r)  # (P, k)
        w = torch.sigmoid(topk_logits.float())  # (P, k)
        return topk_idx, w * dots

    @torch.no_grad()
    def __call__(
        self,
        outputs: InterpretableOutput,
        token_ids: torch.Tensor,
        batch_idx: torch.Tensor,
        block_pos: torch.Tensor,
    ) -> AttributionResult:
        pred_ids = token_ids[batch_idx, block_pos]
        pred_logits = torch.gather(
            outputs.logits[batch_idx, block_pos].float(), -1, pred_ids.unsqueeze(-1)
        ).squeeze(-1)  # (P)
        W_y = self.output_weights_for_tokens(pred_ids)  # (P, D)
        k_idx, k_c = self._known_head_contrib(
            W_y,
            topk_idx=outputs.known_topk_indices[batch_idx, block_pos],
            topk_logits=outputs.known_topk_logits[batch_idx, block_pos],
        )
        d_idx, d_c = self._unknown_head_contrib(
            W_y,
            topk_idx=outputs.unknown_topk_indices[batch_idx, block_pos],
            topk_logits=outputs.unknown_topk_logits[batch_idx, block_pos],
        )
        eps = pred_logits - k_c.sum(-1) - d_c.sum(-1)
        return AttributionResult(k_idx, k_c, d_idx, d_c, eps)


def chunk_ranges(
    start: int,
    token_ids: list[int],
    endofchunk_token_id: int,
    stop_token_ids: set[int] | None = None,
    *,
    prev_token_eos_or_eoc: bool = False,
    compute_last_chunk_idx: bool = False,
) -> tuple[list[tuple[int, int]], bool]:
    if stop_token_ids is None:
        stop_token_ids = set()
    chunks: list[tuple[int, int]] = []
    if len(token_ids) == 0 or start >= len(token_ids):
        return chunks, prev_token_eos_or_eoc
    chunk_start = start
    for pos in range(start, len(token_ids)):
        if token_ids[pos] == endofchunk_token_id:
            chunks.append((chunk_start, pos))
            chunk_start = pos + 1
            prev_token_eos_or_eoc = True
        elif token_ids[pos] in stop_token_ids:
            if not prev_token_eos_or_eoc:
                prev_token_eos_or_eoc = True
                if pos != start:
                    chunks.append((chunk_start, pos))
            chunk_start = pos + 1
        else:
            prev_token_eos_or_eoc = False
    if compute_last_chunk_idx and chunk_start < len(token_ids) and not prev_token_eos_or_eoc:
        chunks.append((chunk_start, len(token_ids)))
    return chunks, prev_token_eos_or_eoc


def index_sequence_positions(
    batch_idx: torch.Tensor, block_pos: torch.Tensor, seq_pos: torch.Tensor
) -> torch.Tensor:
    """Index global sequence positions with local block position(s).

    Args:
        batch_idx (torch.Tensor): batch indexes.
        block_pos (torch.Tensor): block position(s), local to the current block(s).
        seq_pos (torch.Tensor): sequence positions for the current block(s).

    Returns:
        seq_pos[block_pos] (torch.Tensor): sequence positions indexed at block position(s).
    """
    torch._assert_async(
        ((block_pos >= 0) & (block_pos < seq_pos.shape[-1])).all(),
        "block_pos out of bounds witf bounds (negative index or larger than seq_pos)",
    )

    torch._assert_async(
        ((batch_idx >= 0) & (batch_idx < seq_pos.shape[0])).all(),
        "batch_idx out of bounds with seq_pos (negative index or larger than seq_pos)",
    )

    if block_pos.ndim == 0:
        raise RuntimeError("block_pos is a scalar, number of dim is 0.")

    if seq_pos.ndim != 2:
        raise RuntimeError(f"seq_pos should have a batch and positional dimension. has only {seq_pos.ndim}")

    return seq_pos[batch_idx, block_pos]


@torch.no_grad()
def compute_concept_attribution(
    outputs: InterpretableOutput,
    token_ids: torch.Tensor,
    attributor: Attributor,
    batch_idx: torch.Tensor,
    batch_idx_list: list[torch.Tensor],
    block_pos: torch.Tensor,
    block_pos_list: list[torch.Tensor],
    seq_pos: torch.Tensor,
    attr_cache: list[AttributionCache],
) -> None:
    """Compute per-position concept contributions using sparse top-k indices.

    For each position, decomposes the predicted token logit into:
        - known concept contributions
        - discovered (unknown) concept contributions
        - epsilon residual

    Args:
        outputs: Backbone output object with known_topk_indices, known_topk_logits,
                 unknown_topk_indices, unknown_topk_logits and logits attributes.
        token_ids: token ids
        attributor: The backbone model with known_head, unknown_head, and lm_head.
        batch_idx: batch indexes.
        batch_idx_list: a list of batch_idxes.
        block_pos: tensor of local position indices.
        block_pos_list: a list of block_pos.
        seq_pos: tensor of global position indices.
        attr_cache: attribution cache which holds computed concept attributions.
    """
    torch._assert_async(
        ((block_pos >= 0) & (block_pos < token_ids.shape[1])).all(), "block_pos out of bounds with token_ids"
    )

    torch._assert_async(
        (block_pos < outputs.known_topk_indices.shape[1]).all(),
        "block_pos has out of bounds index with topk_indices",
    )

    torch._assert_async(
        ((batch_idx >= 0) & (batch_idx < token_ids.shape[0])).all(), "batch_idx out of bounds with token_ids"
    )

    torch._assert_async(
        (batch_idx < outputs.known_topk_indices.shape[0]).all(),
        "batch_idx has out of bounds index with topk_indices",
    )

    if block_pos.ndim == 0:
        raise RuntimeError("block_pos is a scalar, number of dim is 0.")

    if token_ids.ndim != 2:
        raise RuntimeError(f"token_ids is incorrect shape {token_ids.shape} with num dims not equal to 2")

    attributions = attributor(outputs, token_ids, batch_idx, block_pos)
    i, j = 0, 0
    for c, (batch_idx_t, block_pos_t) in enumerate(zip(batch_idx_list, block_pos_list, strict=True)):
        indexed_seq_pos = index_sequence_positions(batch_idx_t, block_pos_t, seq_pos)
        j += batch_idx_t.shape[0]
        attr_result = AttributionResult(
            known_indices=attributions.known_indices[i:j],
            known_contributions=attributions.known_contributions[i:j],
            disc_indices=attributions.disc_indices[i:j],
            disc_contributions=attributions.disc_contributions[i:j],
            epsilon=attributions.epsilon[i:j],
        )
        attr_cache[c].cache_write(attr_result, indexed_seq_pos)
        i = j


def chunk_attribution(
    attr: AttributionCache, start: int, end: int, k_known: int, k_unknown: int, num_known_concepts: int
) -> ChunkedAttribution:
    """Compute concept attribution over a chunk of tokens [start, end).

    Per position, each concept's contribution is normalized by the total absolute
    logit mass at that position (known + discovered + epsilon). This removes
    scale differences across positions so every token votes equally to the chunk.

    Args:
        attr: AttributionResult from compute_concept_attribution.
        start: Start token index (inclusive).
        end: End token index (exclusive).
        k_known: Number of k concepts for known
        k_unknown: Number of k concepts for unknown
        num_known_concepts: number of known concepts, used to offset unknown indexes

    Returns:
        entries: List of AttributionEntry dicts sorted by abs contribution.
        eps_pct: Epsilon as a percentage of total logit mass (mean over chunk).
    """
    if start == end:
        return ChunkedAttribution()
    k_idx, k_c, d_idx, d_c, eps = attr.index_cache(start, end)
    pos_total = (k_c.abs().sum(-1) + d_c.abs().sum(-1) + eps.abs()).clamp(min=1e-8)  # (T,)
    k_c_norm = k_c / pos_total.unsqueeze(-1)  # (T, K_known)
    d_c_norm = d_c / pos_total.unsqueeze(-1)  # (T, K_disc)

    def aggregate(idx: Tensor, contrib_norm: Tensor) -> tuple[Tensor, Tensor]:
        flat_idx = idx.reshape(-1)
        flat_norm = contrib_norm.reshape(-1)
        unique_ids, inverse = flat_idx.unique(return_inverse=True)
        mass = torch.zeros(len(unique_ids), device=flat_idx.device)
        mass.scatter_add_(0, inverse, flat_norm)
        return unique_ids, mass

    k_ids, k_mass = aggregate(k_idx, k_c_norm)
    d_ids, d_mass = aggregate(d_idx, d_c_norm)
    T = end - start
    if k_mass.shape[-1] < k_known:
        raise RuntimeError(
            f"number of unique concepts from k_mass {k_mass.shape[-1]}"
            f" is less than requested number of known concepts {k_known}."
        )
    if d_mass.shape[-1] < k_unknown:
        raise RuntimeError(
            f"number of unique concepts from d_mass {d_mass.shape[-1]}"
            f" is less than requested number of unknown concepts {k_unknown}."
        )
    cat_mass = torch.cat([k_mass, d_mass], dim=-1) / T
    cat_indices = torch.cat([k_ids, d_ids + num_known_concepts], dim=-1)
    _, sorted_indices = torch.sort(torch.abs(cat_mass), dim=-1, descending=True)
    concept_scores = cat_mass[sorted_indices]
    indices = cat_indices[sorted_indices]
    return ChunkedAttribution(concept_scores=concept_scores.tolist(), indices=indices.tolist())
