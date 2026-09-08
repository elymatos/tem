"""Prediction and stabilisation heads for TEM-t.

Implements two key components:

1. SensoryPredictionHead
   Converts attention readout (retrieved sensory memory) into sensory
   class logits via a small MLP:
       logits = f_pred(r_t),  prob = softmax(logits)

2. LandmarkStabilizer
   Uses the observed sensory input to retrieve a position estimate from
   memory (sensory-as-landmark), then fuses it with the path-integrated
   position via a learned per-dimension gate:
       eta = sigmoid(f_mix([g_pi, g_retrieved]))
       g_next = g_pi + eta * (g_retrieved - g_pi)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from model.types import MemoryState, AttentionOutput
from model.tem_attention import TEMAttention
from model.tem_attention import SensoryProjector  # noqa: F401 — used in type hints


# ---------------------------------------------------------------------------
# SensoryPredictionHead
# ---------------------------------------------------------------------------
class SensoryPredictionHead(nn.Module):
    """MLP mapping attention readout to sensory class logits.

    Transforms the retrieved sensory memory representation r_t of shape
    [B, d_v] into unnormalised logits over the N_x sensory classes.

    Formula:
        logits = MLP(r_t)
        where MLP: d_v -> hidden -> ... -> N_x

    Parameters
    ----------
    d_v : int
        Input dimensionality (value space from attention).
    n_sensory : int
        Number of sensory classes (output dimensionality).
    hidden_dim : int, optional
        Hidden layer width. Default 128.
    n_layers : int, optional
        Number of hidden layers (0 = linear projection only). Default 1.
    """

    def __init__(
        self,
        d_v: int,
        n_sensory: int,
        hidden_dim: int = 128,
        n_layers: int = 1,
    ) -> None:
        super().__init__()
        self.d_v = d_v
        self.n_sensory = n_sensory

        layers = []
        in_dim = d_v

        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        # Final projection to N_x
        layers.append(nn.Linear(in_dim, n_sensory))

        self.mlp = nn.Sequential(*layers)

    def forward(self, read_x: torch.FloatTensor) -> torch.FloatTensor:
        """Predict sensory logits from the attention readout.

        Parameters
        ----------
        read_x : FloatTensor [B, d_v]
            Retrieved sensory value from attention.

        Returns
        -------
        logits : FloatTensor [B, N_x]
            Unnormalised sensory class logits.
        """
        return self.mlp(read_x)


# ---------------------------------------------------------------------------
# LandmarkStabilizer
# ---------------------------------------------------------------------------
class LandmarkStabilizer(nn.Module):
    """Position stabilisation using sensory landmarks.

    After observing the next sensory input x_{t+1}, this module:
    1. Retrieves a position estimate from memory based on sensory similarity
       (query = x_{t+1}, keys = historical x, values = historical g).
    2. Fuses the retrieved position with the path-integrated position via
       a learned per-dimension sigmoid gate.

    Formula (sensory_only mode):
        g_retrieved = f_g_ret( Attention(x_{t+1}, memory.sensory_keys, memory.values_g) )
        eta = sigmoid( f_mix([g_pi, g_retrieved]) )
        g_next = g_pi + eta * (g_retrieved - g_pi)

    Parameters
    ----------
    d_g : int
        Position representation dimensionality.
    d_v : int
        Sensory value dimensionality.
    d_k : int
        Key/query dimensionality for the landmark attention.
    use_landmark : bool, optional
        If False, g_next = g_pi (no correction). Default True.
    mode : str, optional
        Retrieval mode: "sensory_only" or "sensory_pi_disambig".
        Default "sensory_only".
    """

    def __init__(
        self,
        d_g: int,
        d_v: int,
        d_k: int,
        use_landmark: bool = True,
        mode: str = "sensory_only",
    ) -> None:
        super().__init__()
        self.d_g = d_g
        self.d_v = d_v
        self.d_k = d_k
        self.use_landmark = use_landmark
        self.mode = mode

        if mode not in ("sensory_only", "sensory_pi_disambig"):
            raise ValueError(
                f"Unsupported landmark mode '{mode}'. "
                f"Expected 'sensory_only' or 'sensory_pi_disambig'."
            )

        # Projects sensory values (d_v) into the attention key space (d_k)
        # so the TEMAttention module can process sensory-as-query/keys.
        self.sensory_to_key = nn.Linear(d_v, d_k, bias=False)

        # TEMAttention module for sensory-to-position retrieval.
        # Query = projected x, Keys = projected historical x, Values = historical g.
        self.landmark_attention = TEMAttention(
            d_k=d_k,
            beta0=1.0,
            adaptive_beta=True,
        )

        # Maps the attention readout (position value space, d_g) back to d_g.
        self.g_retrieval_mlp = nn.Sequential(
            nn.Linear(d_g, d_g),
            nn.ReLU(),
            nn.Linear(d_g, d_g),
        )

        # Fusion gate MLP f_mix: [g_pi, g_retrieved] -> eta (per-dimension gate).
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * d_g, d_g),
            nn.ReLU(),
            nn.Linear(d_g, d_g),
        )

    # ------------------------------------------------------------------
    # retrieve_position
    # ------------------------------------------------------------------
    def retrieve_position(
        self,
        x_next: torch.FloatTensor,
        g_pi: torch.FloatTensor,
        memory: MemoryState,
        sensory_value_x: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, Optional[AttentionOutput]]:
        """Retrieve a position estimate from memory using the current sensory input.

        In sensory_only mode:
            - Project x_next through W_x to get sensory value [B, d_v].
            - Map sensory value to d_k key space via sensory_to_key.
            - Query = projected sensory (d_k).
            - Keys = projected historical sensory values (d_k).
            - Values = historical position encodings (d_g, from memory.values_g).
            - Pass attention readout through g_retrieval_mlp.

        Parameters
        ----------
        x_next : FloatTensor [B, N_x]
            Current sensory observation (one-hot).
        g_pi : FloatTensor [B, d_g]
            Path-integrated position (used in disambiguation mode only).
        memory : MemoryState
            Current episodic memory.
        sensory_value_x : FloatTensor [B, d_v]
            Pre-computed sensory projection of x_next
            (= sensory_proj(x_next)) passed in for efficiency.

        Returns
        -------
        g_retrieved : FloatTensor [B, d_g]
            Position estimate retrieved from memory.
        attn_landmark : Optional AttentionOutput
            Attention produced during retrieval.
        """
        if not self.use_landmark:
            B = x_next.shape[0]
            M = memory.valid_mask.shape[1]
            device = x_next.device
            return g_pi.clone(), AttentionOutput(
                read=torch.zeros(B, self.d_g, device=device),
                weights=torch.zeros(B, M, device=device),
                scores=torch.zeros(B, M, device=device),
            )

        # Project sensory to key space
        query_k = self.sensory_to_key(sensory_value_x)  # [B, d_k]

        # Build key projections for all historical sensory observations.
        # memory.raw_x: [B, M, N_x]. Project through sensory embedder
        # to get [B, M, d_v], then to [B, M, d_k].
        # We handle this externally: the caller passes in the sensory
        # values for the entire memory.
        # Actually, compute keys from memory.raw_x.
        # memory.raw_x @ W_x gives sensory values. We embed via the same
        # W_x linear operator. Then map to d_k.

        # For efficiency, compute sensory values for all memory slots
        # using the linear layer: memory.raw_x [B, M, N_x] @ W_x^T
        # W_x in SensoryProjector is nn.Linear(N_x, d_v), so weight is [d_v, N_x].
        # We compute this outside via sensory_projector, but here
        # we accept pre-computed memory_values_x: actually memory.values_x is
        # [B, M, d_v] and is the projected sensory.

        keys_k = self.sensory_to_key(memory.values_x)  # [B, M, d_k]

        # Attention: query from sensory, keys from sensory, values from position
        attn_output = self.landmark_attention(
            query=query_k,
            keys=keys_k,
            values=memory.values_g,  # historical positions [B, M, d_g]
            valid_mask=memory.valid_mask,
            memory_size=memory.size,
        )

        # Map readout back to d_g
        g_retrieved = self.g_retrieval_mlp(attn_output.read)  # [B, d_g]

        return g_retrieved, attn_output

    # ------------------------------------------------------------------
    # forward: fuse PI and retrieved position
    # ------------------------------------------------------------------
    def forward(
        self,
        g_pi: torch.FloatTensor,
        g_retrieved: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Fuse path-integrated and landmark-retrieved position encodings.

        Formula:
            eta = sigmoid( f_mix([g_pi, g_retrieved]) )
            g_next = g_pi + eta * (g_retrieved - g_pi)

        where eta is a per-dimension gate controlling how much the
        landmark-retrieved position corrects the path-integrated estimate.

        Parameters
        ----------
        g_pi : FloatTensor [B, d_g]
            Path-integrated position g_{t+1}^{PI}.
        g_retrieved : FloatTensor [B, d_g]
            Position retrieved from sensory landmarks.

        Returns
        -------
        g_next : FloatTensor [B, d_g]
            Stabilized position encoding g_{t+1}.
        eta : FloatTensor [B, d_g]
            Per-dimension fusion gate in (0, 1).
            0 = fully trust PI, 1 = fully trust landmark.
        """
        if not self.use_landmark:
            return g_pi, torch.zeros_like(g_pi)

        # Concatenate and compute gate
        combined = torch.cat([g_pi, g_retrieved], dim=-1)  # [B, 2*d_g]
        eta = torch.sigmoid(self.fusion_mlp(combined))       # [B, d_g]

        # Interpolate between PI and retrieved position
        g_next = g_pi + eta * (g_retrieved - g_pi)           # [B, d_g]

        return g_next, eta
