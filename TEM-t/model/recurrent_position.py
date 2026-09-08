"""Action-conditioned recurrent positional encoder.

Implements the core TEM-t path integration equation:
    g_{t+1}^{PI} = sigma(g_t @ W_{a_t})

where W_a is an action-dependent transition matrix of shape [d_g, d_g].
The activation sigma is configurable: identity, relu, or tanh.

This is the engine that learns spatial transition structure from
sequences of actions, producing structured positional representations
that can exhibit grid-like, band-like, or place-like tuning.
"""

from typing import Optional

import torch
import torch.nn as nn


class RecurrentPositionEncoder(nn.Module):
    """Action-conditioned recurrent position encoder.

    Maintains an internal position representation g_t that is updated
    at each step according to the action taken, without access to the
    upcoming sensory observation. This is the pure path-integration (PI)
    component of TEM-t.

    Parameters
    ----------
    d_g : int
        Dimensionality of the position representation.
    n_actions : int
        Number of discrete actions (e.g. 4 for N/E/S/W).
    activation : str, optional
        Nonlinearity applied after the linear transition.
        One of {"identity", "relu", "tanh"}. Default "identity".
    init_scale : float, optional
        Scale for weight initialization. Default 0.01.
    g0 : Optional torch.FloatTensor, optional
        Fixed initial position encoding [d_g]. If None, uses zeros.
        When learnable_g0=True this becomes a trainable parameter.
    learnable_g0 : bool, optional
        Whether g_0 is a learnable parameter. Default False.
    """

    def __init__(
        self,
        d_g: int,
        n_actions: int,
        activation: str = "identity",
        init_scale: float = 0.01,
        g0: Optional[torch.FloatTensor] = None,
        learnable_g0: bool = False,
    ) -> None:
        super().__init__()

        if activation not in ("identity", "relu", "tanh"):
            raise ValueError(
                f"Unsupported activation '{activation}'. "
                f"Expected one of: identity, relu, tanh."
            )

        self.d_g = d_g
        self.n_actions = n_actions
        self.activation_name = activation

        # Action-conditioned transition matrices: [n_actions, d_g, d_g]
        self.W_a = nn.Parameter(
            torch.empty(n_actions, d_g, d_g)
        )

        # Initial position encoding g_0
        if learnable_g0:
            self.g0 = nn.Parameter(torch.zeros(d_g))
        elif g0 is not None:
            self.register_buffer("g0", g0.clone().detach())
        else:
            self.register_buffer("g0", torch.zeros(d_g))

        self.learnable_g0 = learnable_g0

        self._init_weights(init_scale)

    def _init_weights(self, scale: float) -> None:
        """Initialize action transition matrices with small random values.

        Uses a normal distribution scaled so that repeated application
        does not immediately explode or vanish the position encoding.

        Parameters
        ----------
        scale : float
            Standard deviation of the normal distribution.
        """
        nn.init.normal_(self.W_a, mean=0.0, std=scale)

    # ------------------------------------------------------------------
    # Activation function (mapped from string config)
    # ------------------------------------------------------------------
    def _activation(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Apply the configured activation function.

        Parameters
        ----------
        x : FloatTensor [..., d_g]
            Input tensor.

        Returns
        -------
        FloatTensor [..., d_g]
            Activated tensor.
        """
        if self.activation_name == "identity":
            return x
        elif self.activation_name == "relu":
            return torch.relu(x)
        elif self.activation_name == "tanh":
            return torch.tanh(x)
        else:
            raise RuntimeError(f"Unreachable: {self.activation_name}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.FloatTensor:
        """Return the initial position encoding g_0 expanded to batch size.

        Parameters
        ----------
        batch_size : int
            Number of parallel trajectories.
        device : torch.device
            Target device.
        dtype : torch.dtype, optional
            Floating-point type. Default float32.

        Returns
        -------
        g_0 : FloatTensor [B, d_g]
            Initial position encoding replicated across the batch.
        """
        return self.g0.to(device=device, dtype=dtype).unsqueeze(0).expand(
            batch_size, self.d_g
        )

    def forward(
        self,
        g_t: torch.FloatTensor,
        actions: torch.LongTensor,
    ) -> torch.FloatTensor:
        """Execute one step of action-conditioned recurrence.

        Computes g_{t+1}^{PI} = sigma(g_t @ W_{a_t}) for each item
        in the batch.

        Parameters
        ----------
        g_t : FloatTensor [B, d_g]
            Current position encoding (before the action).
        actions : LongTensor [B]
            Action indices, one per batch item.
            Must satisfy 0 <= action < n_actions.

        Returns
        -------
        g_pi : FloatTensor [B, d_g]
            Path-integrated position encoding after the action.
        """
        # Select the per-action transition matrix: [B, d_g, d_g]
        W_selected = self.W_a[actions]  # [B, d_g, d_g]

        # Batched matrix-vector product: g_t [B, 1, d_g] @ W [B, d_g, d_g]
        # -> [B, 1, d_g] -> [B, d_g]
        g_pi = torch.bmm(
            g_t.unsqueeze(1),   # [B, 1, d_g]
            W_selected,         # [B, d_g, d_g]
        ).squeeze(1)            # [B, d_g]

        return self._activation(g_pi)
