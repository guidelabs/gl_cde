"""Concept decomposition head with memory-efficient streaming operations.

Supports both known and unknown concepts with:
- Streaming feature computation (no (B, T, C) allocation)
- Streaming top-k selection (memory O(B * T * k))
- Sparse logit computation for loss functions
- Interventions (known concepts only)
"""

from dataclasses import dataclass
from typing import final

from torch import Tensor, nn
from torch.fx.experimental.symbolic_shapes import statically_known_true
import torch
import torch.nn.functional as F


@dataclass
class ConceptHeadOutput:
    """Output from ConceptHead forward pass.

    Attributes:
        features: Final concept features after intervention (B, T, D)
        predicted: Predicted features (B, T, D)
        topk_indices: Top-k concept indices (B, T, k). Set when using streaming top-k.
        topk_logits: Logits for top-k concepts (B, T, k). Set when using streaming top-k.
        hidden: Hidden states passed to this head (B, T, D). Stored for attribution.
    """

    features: Tensor
    predicted: Tensor
    topk_indices: Tensor
    topk_logits: Tensor
    hidden: Tensor | None = None


@final
class ConceptHead(nn.Module):
    """Concept decomposition head supporting both known and unknown concepts.

    Memory-efficient implementation that avoids (B, T, C) allocations by default.

    Modes:
    - Known (is_unknown=False): Supports top-k, interventions
    - Unknown (is_unknown=True):

    Architectures:
    - Query projection + sigmoid attention over embeddings

    Factorization (for large unknown heads):
    - factorize=False: Dense embeddings (C, D) and predictor (D, C)
    - factorize=True: Factorized embeddings (C, r) @ (r, D) where r << D
                      Reduces memory by ~10-20x for large C

    Memory Safety:
    - Unknown heads with n_concepts > 50k cannot use dense operations
    - Interventions are only supported for known heads
    - All tensor indexing uses F.embedding for DTensor safety

    Args:
        n_concepts: Number of concepts (C)
        concept_dim: Dimension of concept embeddings (should equal n_embd)
        n_embd: Model hidden dimension
        is_unknown: if model is unknown head
        topk: Top-k sparsity for concept weights. None = no sparsity.
        block_size: Block size for memory-efficient operations.
        pad_multiple: Pad n_concepts to a multiple of this for efficiency
        factorize: If True, use low-rank factorized embeddings
        factorize_rank: Rank for factorization (r). Lower = less memory, less expressivity.
    """

    def __init__(
        self,
        n_concepts: int,
        concept_dim: int,
        n_embd: int,
        topk: int = 16,
        block_size: int = 8192,
        *,
        is_unknown: bool = False,
        pad_multiple: int = 16,
        # Factorization options
        factorize: bool = False,
        factorize_rank: int = 256,
    ) -> None:
        super().__init__()

        self.n_concepts = n_concepts
        self.concept_dim = concept_dim
        self.n_embd = n_embd
        self.is_unknown = is_unknown
        self.topk = topk
        self.block_size = block_size

        # Flags
        self.pad_multiple = pad_multiple

        # Factorization options
        self.factorize = factorize
        self.factorize_rank = factorize_rank

        # Pad n_concepts to multiple of pad_multiple for efficiency
        self.n_concepts_padded = ((n_concepts + pad_multiple - 1) // pad_multiple) * pad_multiple

        # Weight initialization: Dense OR Factorized
        if factorize:
            # Factorized embeddings: E = coef @ basis
            # coef: (C, r), basis: (r, D) -> E: (C, D)
            self.embedding_coef = nn.Embedding(self.n_concepts_padded, factorize_rank, dtype=torch.float32)
            self.embedding_basis = nn.Linear(factorize_rank, concept_dim, bias=False, dtype=torch.float32)
            self.concept_embedding = None  # Don't allocate dense

            # Factorized predictor: logits = hidden @ down @ up.T
            # down: (D, r), up: (r, C) -> effective W: (D, C)
            self.predictor_down = nn.Linear(n_embd, factorize_rank, bias=False, dtype=torch.float32)
            self.predictor_up = nn.Linear(
                factorize_rank, self.n_concepts_padded, bias=False, dtype=torch.float32
            )
            self.concept_predictor = None
        else:
            # Dense embeddings (what we had before)
            self.concept_embedding = nn.Embedding(self.n_concepts_padded, concept_dim, dtype=torch.float32)
            self.embedding_coef = None
            self.embedding_basis = None

            self.concept_predictor = nn.Linear(
                n_embd, self.n_concepts_padded, bias=False, dtype=torch.float32
            )

            self.predictor_down = None
            self.predictor_up = None

    def init_weights(self) -> None:
        if self.factorize:
            if self.embedding_coef is not None:
                nn.init.normal_(self.embedding_coef.weight, mean=0.0, std=0.02)
            if self.embedding_basis is not None:
                nn.init.normal_(self.embedding_basis.weight, mean=0.0, std=0.02)
            if self.predictor_down is not None:
                nn.init.normal_(self.predictor_down.weight, mean=0.0, std=0.02)
            if self.predictor_up is not None:
                nn.init.normal_(self.predictor_up.weight, mean=0.0, std=0.02)
        else:
            if self.concept_embedding is not None:
                nn.init.normal_(self.concept_embedding.weight, mean=0.0, std=0.02)
            if self.concept_predictor is not None:
                nn.init.normal_(self.concept_predictor.weight, mean=0.0, std=0.02)

    def _safe_index(self, weight: Tensor, indices: Tensor) -> Tensor:
        """DTensor-safe indexing using F.embedding.

        Replaces weight[indices] which crashes under FSDP2/DTensor.

        Args:
            weight: (N, D) weight matrix
            indices: (...) indices to select

        Returns:
            (..., D) selected embeddings
        """
        if not torch.compiler.is_compiling():
            torch._assert_async(
                ((indices >= 0) & (indices < self.n_concepts)).all(),
                "indices out of bounds, negative or concept",
            )
        original_shape = indices.shape
        flat_indices = indices.reshape(-1)
        flat_result = F.embedding(flat_indices, weight)
        return flat_result.reshape(*original_shape, -1)

    def _get_embedding_weight(self) -> Tensor:  # pyright: ignore[reportUnusedFunction]
        """Get full embedding matrix.

        For dense: returns concept_embedding.weight
        For factorized: computes coef @ basis (materializes full matrix)

        Returns:
            (C, D) embedding matrix
        """
        if self.concept_embedding is not None:
            return self.concept_embedding.weight
        # Factorized: full embedding matrix from coef and basis.
        return self.embedding_basis(self.embedding_coef.weight)  # pyright: ignore[reportOptionalMemberAccess, reportOptionalCall]

    def _get_embedding(self, indices: Tensor) -> Tensor:  # pyright: ignore[reportUnusedFunction]
        """Get embeddings for specific indices (DTensor-safe).

        For dense: uses F.embedding
        For factorized: looks up coef, then applies basis

        Args:
            indices: (...) concept indices

        Returns:
            (..., D) embeddings
        """
        if not torch.compiler.is_compiling():
            torch._assert_async(
                ((indices >= 0) & (indices < self.n_concepts)).all(),
                "indices out of bounds, negative or concept",
            )
        if self.concept_embedding is not None:
            # Dense path - use nn.Embedding's forward (DTensor-safe)
            return self.concept_embedding(indices)
        # Factorized path: E[i] = basis(coef[i])
        assert self.embedding_coef is not None
        assert self.embedding_basis is not None
        coef = self.embedding_coef(indices)  # (..., r)
        return self.embedding_basis(coef)  # (..., D)

    def _get_predictor_weight(self) -> Tensor:
        """Get full predictor weight matrix (for linear path only).

        Returns:
            (C, D) predictor weight, or None if using attention
        """
        if self.concept_predictor is not None:
            return self.concept_predictor.weight
        assert self.predictor_down is not None
        assert self.predictor_up is not None
        # Factorized: multiply up (C×r) and down (r×D) to get full W (C×D).
        return self.predictor_up.weight @ self.predictor_down.weight

    @staticmethod
    def _merge_topk(
        topv: Tensor,
        topi: Tensor,
        v_blk: Tensor,
        i_blk: Tensor,
        k: int,
    ) -> tuple[Tensor, Tensor]:
        """Efficient merge of two top-k sets. Memory: O(BT × 2k).

        Returns:
            tuple[Tensor, Tensor]: Merged top-k values and indices.
        """
        cand_v = torch.cat([topv, v_blk], dim=1)  # (BT, 2k)
        cand_i = torch.cat([topi, i_blk], dim=1)  # (BT, 2k)
        new_v, sel = torch.topk(cand_v, k, dim=1)  # (BT, k)
        new_i = torch.gather(cand_i, 1, sel)  # (BT, k)
        return new_v, new_i

    # Dense Streaming Top-K Methods
    def linear_features_topk_streaming(self, hidden: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Memory-efficient linear prediction with streaming top-k.

        Uses merge-k-with-k to keep memory O(BT × k), not O(BT × block_size).

        Args:
            hidden: (B, T, D)

        Returns:
            features: (B, T, D) weighted concept features
            topk_indices: (B, T, k) indices of top-k concepts
            topk_logits: (B, T, k) logits for top-k concepts
        """
        assert self.concept_predictor is not None
        assert self.concept_embedding is not None
        B, T, D = hidden.shape
        C = self.n_concepts
        BT = B * T
        device = hidden.device
        k = min(self.topk, C)
        hidden_dtype = hidden.dtype
        hidden = hidden.float()
        flat_h = hidden.reshape(BT, D)
        W = self._get_predictor_weight()  # (C, D)
        # if compiled with dynamic=True, this expression will not be static.
        if statically_known_true(self.block_size >= BT):
            # In-place ops avoids an extra (BT, C) copy
            logits = F.linear(flat_h, W[:C]).clamp_(-15, 15).sigmoid_()
            _, topi = torch.topk(logits, k, dim=1)
        else:
            # Initialize top-k trackers
            topv = torch.full((BT, k), float("-inf"), device=device, dtype=torch.float32)
            topi = torch.zeros((BT, k), device=device, dtype=torch.long)

            # Pass 1: Find global top-k per token via streaming merge
            for start in range(0, C, self.block_size):
                end = min(start + self.block_size, C)
                W_chunk = W[start:end, :]
                # Compute logits for this block
                logits_blk = flat_h @ W_chunk.T.contiguous()
                logits_blk = logits_blk.clamp(-15, 15)

                # Values to rank by
                vals_blk = torch.sigmoid(logits_blk)

                # Get top-k within this block
                blk_k = min(k, end - start)
                v_blk, idx_blk = torch.topk(vals_blk, blk_k, dim=1)
                i_blk = idx_blk + start  # Offset to global indices

                # Pad if block smaller than k
                if blk_k < k:
                    pad_v = torch.full((BT, k - blk_k), float("-inf"), device=device, dtype=torch.float32)
                    pad_i = torch.zeros((BT, k - blk_k), device=device, dtype=torch.long)
                    v_blk = torch.cat([v_blk, pad_v], dim=1)
                    i_blk = torch.cat([i_blk, pad_i], dim=1)

                # Merge with running top-k
                topv, topi = self._merge_topk(topv, topi, v_blk, i_blk, k)

        # Pass 2: Compute features from top-k concepts only
        # Recompute logits for selected concepts (for accurate gradients)
        W_sel = self._safe_index(W, topi)  # (BT, k, D)
        logits_sel = torch.einsum("bd,bkd->bk", flat_h, W_sel)
        logits_sel = logits_sel.clamp(-15, 15)

        # Compute weights
        weights_sel = torch.sigmoid(logits_sel)

        # Compute features (DTensor-safe)
        E_sel = self.concept_embedding(topi)  # (BT, k, D)
        features = torch.einsum("bk,bkd->bd", weights_sel, E_sel)
        return (
            features.reshape(B, T, D).to(hidden_dtype),
            topi.reshape(B, T, k),
            logits_sel.reshape(B, T, k),
        )

    def linear_features_topk_factorized(self, hidden: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Memory-efficient factorized linear with streaming top-k.

        Args:
            hidden: (B, T, D) hidden states

        Returns:
            features: (B, T, D) weighted concept features
            topk_indices: (B, T, k) top-k concept indices
            topk_logits: (B, T, k) logits for top-k concepts
        """
        assert self.factorize, "Only valid for factorized head"
        assert self.predictor_down is not None, "Linear path requires predictor"
        assert self.predictor_up is not None, "Linear path requires predictor"
        assert self.embedding_basis is not None
        assert self.embedding_coef is not None

        B, T, D = hidden.shape
        BT = B * T
        C = self.n_concepts
        _ = self.factorize_rank
        device = hidden.device
        k = min(self.topk, C)
        hidden_dtype = hidden.dtype
        hidden = hidden.float()
        flat_h = hidden.reshape(BT, D)

        # Get factorized components
        up_weight = self.predictor_up.weight[:C]  # (C, r)

        # Compress hidden: (BT, D) @ (D, r) → (BT, r)
        h_compressed = self.predictor_down(flat_h)
        # if compiled with dynamic=True, this expression will not be static.
        if statically_known_true(self.block_size >= BT):
            # In-place ops avoids an extra (BT, C) copy
            logits = F.linear(h_compressed, up_weight).clamp_(-15, 15)
            _, topi = torch.topk(logits, k, dim=1)
        else:
            # Initialize top-k trackers
            topv = torch.full((BT, k), float("-inf"), device=device, dtype=torch.float32)
            topi = torch.zeros((BT, k), device=device, dtype=torch.long)

            # Pass 1: Find global top-k via streaming merge
            for start in range(0, C, self.block_size):
                end = min(start + self.block_size, C)
                up_chunk = up_weight[start:end]  # (chunk, r)

                # Factorized logits for chunk
                logits_chunk = F.linear(h_compressed, up_chunk)
                logits_chunk = logits_chunk.clamp(-15, 15)

                # Top-k within chunk
                blk_k = min(k, end - start)
                v_chunk, idx_chunk = torch.topk(logits_chunk, blk_k, dim=1)
                i_chunk = idx_chunk + start

                # Pad if needed
                if blk_k < k:
                    pad_v = torch.full((BT, k - blk_k), float("-inf"), device=device, dtype=torch.float32)
                    pad_i = torch.zeros((BT, k - blk_k), device=device, dtype=torch.long)
                    v_chunk = torch.cat([v_chunk, pad_v], dim=1)
                    i_chunk = torch.cat([i_chunk, pad_i], dim=1)

                # Merge
                topv, topi = self._merge_topk(topv, topi, v_chunk, i_chunk, k)

        # Pass 2: Compute features from top-k
        # Recompute logits for selected (full precision for gradients)
        up_sel = self._safe_index(up_weight, topi)  # (BT, k, r)
        logits_sel = torch.einsum("br,bkr->bk", h_compressed, up_sel)  # (BT, r)
        logits_sel = logits_sel.clamp(-15, 15)

        # Compute weighted features
        weights_sel = torch.sigmoid(logits_sel)

        coef_sel = self.embedding_coef(topi)  # (BT, k, r)
        weighted_coef = torch.einsum("bk,bkr->br", weights_sel, coef_sel)  # (BT, r)
        features = self.embedding_basis(weighted_coef)  # (BT, D)
        return (
            features.reshape(B, T, D).to(hidden_dtype),
            topi.reshape(B, T, k),
            logits_sel.reshape(B, T, k),
        )

    def forward(self, hidden: Tensor, *, store_hidden: bool = False) -> ConceptHeadOutput:
        """Forward pass for concept decomposition.

        Args:
            hidden: Transformer hidden states (B, T, n_embd)
            store_hidden: If True, store hidden in output for later attribution.

        Returns:
            ConceptHeadOutput with features, logits, predicted, weights
        """
        # Initialize outputs
        predicted: Tensor
        topk_indices: Tensor
        topk_logits: Tensor

        # Path selection based on: unknown or known
        if self.is_unknown:
            # FACTORIZED PATH (memory efficient)
            predicted, topk_indices, topk_logits = self.linear_features_topk_factorized(hidden)
        else:
            # DENSE STREAMING TOP-K PATH
            predicted, topk_indices, topk_logits = self.linear_features_topk_streaming(hidden)

        features = predicted

        return ConceptHeadOutput(
            features=features,
            predicted=predicted,
            topk_indices=topk_indices,
            topk_logits=topk_logits,
            hidden=hidden.detach() if store_hidden else None,
        )
