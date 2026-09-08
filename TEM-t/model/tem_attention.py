"""TEM-style attention and position/sensory projection modules.

Implements the core TEM-t attention mechanism where:
    - Queries and keys come from position representations (g)
    - Values come from sensory representations (x)

This enforces the TEM-t structural constraint:
    Q, K <- position   (position retrieves)
    V    <- sensory    (sensory is retrieved)

Includes:
    - FixedLayerNorm: z-score normalisation before attention projection
    - PositionProjector: g -> LN -> W_e -> query/key in d_k space
    - SensoryProjector: x -> W_x -> value in d_v space
    - TEMAttention: softmax(beta * qK^T / sqrt(d_k)) V with adaptive beta
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.types import AttentionOutput


# ---------------------------------------------------------------------------
# FixedLayerNorm
# ---------------------------------------------------------------------------
class FixedLayerNorm(nn.Module):
    """Parameter-free z-score normalisation along the last dimension.

    Applied to position encodings before they enter the attention
    projection. This normalisation is *not* fed back into the recurrent
    dynamics — it only affects Q/K computation.

    Formula:
        mu = mean(x, dim=-1)
        sigma2 = var(x, dim=-1, unbiased=False)
        x_norm = (x - mu) / sqrt(sigma2 + eps)

    Parameters
    ----------
    eps : float
        Small constant for numerical stability. Default 1e-5.
    """

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Apply fixed z-score normalisation.

        Parameters
        ----------
        x : FloatTensor [..., D]
            Input tensor (any leading dimensions, D last).

        Returns
        -------
        FloatTensor [..., D]
            Normalised tensor.
        """
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(var + self.eps)


# ---------------------------------------------------------------------------
# PositionProjector
# ---------------------------------------------------------------------------
class PositionProjector(nn.Module):
    """Project position encodings into the attention Q/K space.

    Applies optional fixed layer norm before a learned linear projection.

    Formula:
        g_tilde = LN_fixed(g) @ W_e

    where W_e is a [d_g, d_k] weight matrix.

    Parameters
    ----------
    d_g : int
        Input position encoding dimensionality.
    d_k : int
        Output key/query dimensionality.
    use_fixed_ln : bool
        Whether to apply FixedLayerNorm before projection. Default True.
    """

    def __init__(
        self,
        d_g: int,
        d_k: int,
        use_fixed_ln: bool = True,
    ) -> None:
        super().__init__()
        self.d_g = d_g
        self.d_k = d_k

        self.ln = FixedLayerNorm() if use_fixed_ln else nn.Identity()
        self.W_e = nn.Linear(d_g, d_k, bias=False)

    def forward(self, g: torch.FloatTensor) -> torch.FloatTensor:
        """Project position encoding to key/query space.

        Parameters
        ----------
        g : FloatTensor [B, d_g] or [B, M, d_g]
            Position encoding(s). Leading dims are preserved.

        Returns
        -------
        FloatTensor [B, d_k] or [B, M, d_k]
            Projected position encoding(s).
        """
        g_norm = self.ln(g)
        return self.W_e(g_norm)


# ---------------------------------------------------------------------------
# SensoryProjector
# ---------------------------------------------------------------------------
class SensoryProjector(nn.Module):
    """Project sensory observations into the attention value space.

    Supports two input modes:
        "onehot": x is a one-hot FloatTensor [..., N_x] -> x @ W_x
        "id":     x is a LongTensor [...] of class indices -> Embedding lookup

    For one-hot inputs, this is equivalent to an embedding table lookup
    since x @ W_x picks the row of W_x corresponding to the active index.

    Parameters
    ----------
    n_sensory : int
        Number of sensory classes (N_x).
    d_v : int
        Output value dimensionality.
    input_mode : str
        Either "onehot" or "id". Default "onehot".
    """

    def __init__(
        self,
        n_sensory: int,
        d_v: int,
        input_mode: str = "onehot",
    ) -> None:
        super().__init__()
        self.n_sensory = n_sensory
        self.d_v = d_v
        self.input_mode = input_mode

        if input_mode not in ("onehot", "id"):
            raise ValueError(
                f"input_mode must be 'onehot' or 'id', got '{input_mode}'"
            )

        # Weight matrix W_x: [N_x, d_v]
        # For onehot: x @ W_x. For id: acts as embedding table.
        self.W_x = nn.Linear(n_sensory, d_v, bias=False)

    def forward(self, x: torch.Tensor) -> torch.FloatTensor:
        """Project sensory observation to value space.

        Parameters
        ----------
        x : FloatTensor [..., N_x] (onehot mode) or LongTensor [...] (id mode)
            Sensory observation.

        Returns
        -------
        FloatTensor [..., d_v]
            Projected sensory value.
        """
        if self.input_mode == "onehot":
            return self.W_x(x.float())
        else:
            # id mode: W_x.weight has shape [d_v, N_x]; transpose for embedding
            # Embedding expects weight of shape [num_embeddings, embedding_dim]
            return F.embedding(x, self.W_x.weight.T)


# ---------------------------------------------------------------------------
# TEMAttention
# ---------------------------------------------------------------------------
class TEMAttention(nn.Module):
    """TEM-style memory attention with adaptive temperature.

    Computes:
        beta = beta0 * log(memory_size + 1)    [adaptive temperature]
        scores = beta * (q @ K^T) / sqrt(d_k)  [scaled dot product]
        weights = softmax(scores, mask=valid)  [masked softmax]
        read = weights @ V                      [value aggregation]

    Parameters
    ----------
    d_k : int
        Key/query dimensionality (for scaling).
    beta0 : float
        Base temperature. Default 1.0.
    adaptive_beta : bool
        Use log(m+1) temperature scaling. Default True.
    dropout : float
        Attention dropout probability. Default 0.0.
    """

    def __init__(
        self,
        d_k: int,
        beta0: float = 1.0,
        adaptive_beta: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_k = d_k
        self.beta0 = beta0
        self.adaptive_beta = adaptive_beta
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def compute_beta(
        self,
        memory_size: torch.LongTensor,
    ) -> torch.FloatTensor:
        """Compute the attention temperature.

        When adaptive: beta = beta0 * log(m + 1), which sharpens
        attention as the memory grows, preventing averaging over
        too many irrelevant slots.

        Parameters
        ----------
        memory_size : LongTensor [B]
            Current number of valid memory slots per batch item.

        Returns
        -------
        FloatTensor [B, 1]
            Per-batch-item attention temperature.
        """
        if self.adaptive_beta:
            beta = self.beta0 * torch.log(
                memory_size.float() + 1.0
            )
        else:
            beta = torch.full_like(
                memory_size, self.beta0, dtype=torch.float
            )
        return beta.unsqueeze(-1)  # [B, 1]

    def forward(
        self,
        query: torch.FloatTensor,
        keys: torch.FloatTensor,
        values: torch.FloatTensor,
        valid_mask: torch.BoolTensor,
        memory_size: torch.LongTensor,
    ) -> AttentionOutput:
        """Retrieve values from memory using position-based attention.

        Parameters
        ----------
        query : FloatTensor [B, d_k]
            Current query vector (from projected position).
        keys : FloatTensor [B, M, d_k]
            Memory key matrix (projected historical positions).
        values : FloatTensor [B, M, d_value]
            Memory value matrix (projected sensory observations).
        valid_mask : BoolTensor [B, M]
            True for valid memory slots. Invalid slots get -inf score.
        memory_size : LongTensor [B]
            Number of valid slots per batch item (at least 1).

        Returns
        -------
        AttentionOutput with read, weights, and scores.
        """
        B, M, d_k = keys.shape
        d_value = values.shape[-1]

        # Scaled dot-product scores: [B, 1, d_k] x [B, d_k, M] -> [B, 1, M] -> [B, M]
        scores = torch.bmm(
            query.unsqueeze(1),       # [B, 1, d_k]
            keys.transpose(1, 2),     # [B, d_k, M]
        ).squeeze(1)                  # [B, M]

        scores = scores / (d_k ** 0.5)

        # Adaptive temperature scaling
        beta = self.compute_beta(memory_size)  # [B, 1]
        scores = beta * scores

        # Mask invalid memory slots: set to very negative before softmax
        masked_scores = scores.masked_fill(~valid_mask, float("-inf"))

        # Softmax over memory dimension
        weights = F.softmax(masked_scores, dim=-1)  # [B, M]
        weights = weights * valid_mask.float()        # zero out invalid
        weights = self.dropout(weights)

        # Weighted sum of values: [B, M] x [B, M, d_value] -> [B, d_value]
        read = torch.bmm(
            weights.unsqueeze(1),  # [B, 1, M]
            values,                # [B, M, d_value]
        ).squeeze(1)               # [B, d_value]

        return AttentionOutput(
            read=read,
            weights=weights,
            scores=scores,  # raw scores before masking (for debugging)
        )
