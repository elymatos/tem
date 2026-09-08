"""Baseline TEM (Temporal Episodic Memory) model for comparison.

Implements a simpler associative memory model that uses Hebbian
fast weights stored in a conjunctive (position x sensory) memory
matrix. Unlike TEM-t which uses softmax attention, TEM uses
attractor dynamics to retrieve memories from a learned weight matrix.

Key differences from TEM-t:
  - Memory is a single Hebbian matrix, not individual slots.
  - Retrieval uses iterative attractor dynamics rather than one-shot attention.
  - Position and sensory are bound via outer product (conjunctive code).

This serves as a baseline for sample efficiency comparison
(Experiment 2 / 3).
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.batch import TrajectoryBatch


class TEMBaseline(nn.Module):
    """Original TEM-style model with Hebbian conjunctive memory.

    Parameters
    ----------
    n_sensory : int
        Number of sensory classes.
    n_actions : int
        Number of discrete actions.
    d_g : int
        Position representation dimensionality.
    d_x : int
        Sensory projection dimensionality.
    n_attractor_steps : int, optional
        Number of attractor iterations for memory retrieval. Default 5.
    activation : str, optional
        Recurrent activation. Default "identity".
    """

    def __init__(
        self,
        n_sensory: int,
        n_actions: int,
        d_g: int,
        d_x: int = 32,
        n_attractor_steps: int = 5,
        activation: str = "identity",
    ) -> None:
        super().__init__()

        self.n_sensory = n_sensory
        self.n_actions = n_actions
        self.d_g = d_g
        self.d_x = d_x
        self.n_attractor_steps = n_attractor_steps
        self.activation_name = activation

        # Action-conditioned transition matrices: [n_actions, d_g, d_g]
        self.W_a = nn.Parameter(torch.empty(n_actions, d_g, d_g))

        # Position projection
        self.g_proj = nn.Linear(d_g, d_g, bias=False)

        # Sensory projection
        self.x_proj = nn.Linear(n_sensory, d_x, bias=False)

        # Conjunctive code dimension
        d_p = d_g * d_x
        self.d_p = d_p

        # Prediction head: deconjunctify -> sensory logits
        # Reads out the sensory component from a retrieved conjunctive code.
        self.pred_head = nn.Sequential(
            nn.Linear(d_p, d_p // 2),
            nn.ReLU(),
            nn.Linear(d_p // 2, n_sensory),
        )

        # Initial position
        self.g0 = nn.Parameter(torch.zeros(d_g))

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with small random values."""
        nn.init.normal_(self.W_a, mean=0.0, std=0.01)

    def _activation(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Apply the configured activation function."""
        if self.activation_name == "identity":
            return x
        elif self.activation_name == "relu":
            return torch.relu(x)
        elif self.activation_name == "tanh":
            return torch.tanh(x)
        else:
            return x

    def _path_integrate(
        self,
        g_t: torch.FloatTensor,
        actions: torch.LongTensor,
    ) -> torch.FloatTensor:
        """Compute g_{t+1} = sigma(g_t @ W_{a_t}).

        Parameters
        ----------
        g_t : FloatTensor [B, d_g]
        actions : LongTensor [B]

        Returns
        -------
        FloatTensor [B, d_g]
        """
        W_sel = self.W_a[actions]
        g_next = torch.bmm(g_t.unsqueeze(1), W_sel).squeeze(1)
        return self._activation(g_next)

    def _make_conjunctive(
        self,
        g: torch.FloatTensor,
        x: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Form a conjunctive position-sensory code.

        p = flatten( g_proj(g)^T @ x_proj(x) )
        This is the outer product flattened into a vector.

        Parameters
        ----------
        g : FloatTensor [B, d_g]
            Position encoding.
        x : FloatTensor [B, N_x]
            Sensory observation (one-hot).

        Returns
        -------
        FloatTensor [B, d_g * d_x]
            Conjunctive code vector.
        """
        g_proj = self.g_proj(g)           # [B, d_g]
        x_proj = self.x_proj(x)           # [B, d_x]
        # Outer product: [B, d_g, 1] x [B, 1, d_x] -> [B, d_g, d_x]
        conj = torch.bmm(
            g_proj.unsqueeze(-1),
            x_proj.unsqueeze(1),
        )
        return conj.reshape(g.shape[0], -1)  # [B, d_g * d_x]

    def forward(
        self,
        batch: TrajectoryBatch,
    ) -> dict:
        """Execute TEM forward pass over a trajectory batch.

        Parameters
        ----------
        batch : TrajectoryBatch
            Trajectory batch.

        Returns
        -------
        dict with keys:
            logits : FloatTensor [B, T, N_x] — sensory predictions.
            g_seq : FloatTensor [B, T+1, d_g] — position trajectory.
        """
        B = batch.x.shape[0]
        T = batch.actions.shape[1]
        device = batch.x.device

        # Initialize
        g_t = self.g0.unsqueeze(0).expand(B, self.d_g).to(device)

        # Hebbian memory matrix: accumulates p @ p^T
        M = torch.zeros(B, self.d_p, self.d_p, device=device)

        # Write initial observation
        p_0 = self._make_conjunctive(g_t, batch.x[:, 0])
        M = M + torch.bmm(p_0.unsqueeze(-1), p_0.unsqueeze(1))  # outer product

        g_seq_list = [g_t.clone()]
        logits_list = []

        for t in range(T):
            # Path integrate
            g_pi = self._path_integrate(g_t, batch.actions[:, t])

            # Predict: query M with position-only query
            g_proj = self.g_proj(g_pi)  # [B, d_g]

            # Form query: we need conjunctive query [B, d_p].
            # Use a learned projection from g to d_p
            if not hasattr(self, "_query_proj"):
                self._query_proj = nn.Linear(self.d_g, self.d_p, bias=False).to(device)

            q = self._query_proj(g_proj)  # [B, d_p]

            # Attractor dynamics: q <- sigma(q @ M)
            for _ in range(self.n_attractor_steps):
                q = torch.bmm(q.unsqueeze(1), M).squeeze(1)  # [B, d_p]
                q = F.relu(q)  # attractor nonlinearity

            # Predict sensory from retrieved conjunctive code
            logits = self.pred_head(q)  # [B, N_x]
            logits_list.append(logits)

            # Observe x_{t+1} and update memory
            # For simplicity, we use g_pi as the position (no landmark correction in baseline)
            g_next = g_pi
            g_seq_list.append(g_next.clone())

            p_next = self._make_conjunctive(g_next, batch.x[:, t + 1])
            M = M + torch.bmm(p_next.unsqueeze(-1), p_next.unsqueeze(1))

            g_t = g_next

        g_seq = torch.stack(g_seq_list, dim=1)
        logits = torch.stack(logits_list, dim=1)

        return {
            "logits": logits,
            "g_seq": g_seq,
        }
