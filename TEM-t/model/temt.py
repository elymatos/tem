"""TEM-t main model.

Implements the full TEM-t architecture that combines:
  1. Recurrent positional encoding via action-conditioned transitions.
  2. TEM-style attention with position-based queries and sensory-based values.
  3. Landmark-based position stabilisation.
  4. Episodic memory with deduplication.

The core online prediction protocol:
  g_0 = init_state()
  memory_0 = init_memory(g_0, x_0)

  for t in range(T):
      # Phase A: predict next sensory WITHOUT seeing x_{t+1}
      prediction = predict_next(g_t, memory_t, a_t)

      # Phase B: observe x_{t+1}, correct position, write memory
      observation = observe_next(prediction.g_pi, memory_t, x_{t+1})

      g_t = observation.g_next
      memory_t = observation.memory_next

The strict separation ensures zero-shot evaluation uses only
path-integration predictions (logits_pi), which cannot leak
information from future observations.
"""

from typing import Optional, Dict, Any

import torch
import torch.nn as nn

from training.batch import TrajectoryBatch
from model.types import (
    MemoryState,
    AttentionOutput,
    PredictionState,
    ObservationState,
    TEMTStepOutput,
    TEMTForwardOutput,
)
from model.recurrent_position import RecurrentPositionEncoder
from model.tem_attention import (
    PositionProjector,
    SensoryProjector,
    TEMAttention,
)
from model.memory import TEMMemory
from model.heads import SensoryPredictionHead, LandmarkStabilizer


class TEMTModel(nn.Module):
    """TEM-t: Temporal Episodic Memory Transformer.

    A transformer model for studying hippocampal-like spatial
    representations. Uses action-conditioned recurrent positions,
    TEM-style separated Q/K/V projections, and episodic memory
    with landmark-based stabilisation.

    Parameters
    ----------
    n_sensory : int
        Number of sensory classes (N_x).
    n_actions : int
        Number of discrete actions (N_a).
    d_g : int
        Position representation dimensionality.
    d_k : int
        Attention key/query dimensionality.
    d_v : int
        Sensory value dimensionality.
    max_memory : int
        Maximum memory slots (M). Default T + 1.
    activation : str, optional
        Recurrent activation: "identity", "relu", or "tanh".
    beta0 : float, optional
        Base attention temperature. Default 1.0.
    use_landmark_stabilization : bool, optional
        Enable landmark-based g stabilisation. Default True.
    memory_dedup : bool, optional
        Enable memory write deduplication. Default True.
    use_fixed_layer_norm : bool, optional
        Use FixedLayerNorm before attention projection. Default True.
    prediction_hidden_dim : int, optional
        Hidden dim for sensory prediction MLP. Default 128.
    landmark_mode : str, optional
        Landmark retrieval mode. Default "sensory_only".
    init_scale : float, optional
        Scale for action-matrix initialisation. Default 0.01.
    """

    def __init__(
        self,
        n_sensory: int,
        n_actions: int,
        d_g: int,
        d_k: int,
        d_v: int,
        max_memory: int,
        activation: str = "identity",
        beta0: float = 1.0,
        use_landmark_stabilization: bool = True,
        memory_dedup: bool = True,
        use_fixed_layer_norm: bool = True,
        prediction_hidden_dim: int = 128,
        landmark_mode: str = "sensory_only",
        init_scale: float = 0.01,
    ) -> None:
        super().__init__()

        self.n_sensory = n_sensory
        self.n_actions = n_actions
        self.d_g = d_g
        self.d_k = d_k
        self.d_v = d_v
        self.max_memory = max_memory
        self.use_landmark_stabilization = use_landmark_stabilization

        # ---- Submodules ----

        # Recurrent position encoder: g_{t+1}^{PI} = sigma(g_t @ W_a)
        self.position_encoder = RecurrentPositionEncoder(
            d_g=d_g,
            n_actions=n_actions,
            activation=activation,
            init_scale=init_scale,
            learnable_g0=True,
        )

        # Position -> Q/K projection
        self.position_proj = PositionProjector(
            d_g=d_g,
            d_k=d_k,
            use_fixed_ln=use_fixed_layer_norm,
        )

        # Sensory -> V projection
        self.sensory_proj = SensoryProjector(
            n_sensory=n_sensory,
            d_v=d_v,
            input_mode="onehot",
        )

        # TEM attention for sensory prediction (Q from g, K from g, V from x)
        self.sensory_attention = TEMAttention(
            d_k=d_k,
            beta0=beta0,
            adaptive_beta=True,
        )

        # Sensory prediction head
        self.pred_head = SensoryPredictionHead(
            d_v=d_v,
            n_sensory=n_sensory,
            hidden_dim=prediction_hidden_dim,
            n_layers=1,
        )

        # Landmark stabilizer for position correction
        self.landmark_stabilizer = LandmarkStabilizer(
            d_g=d_g,
            d_v=d_v,
            d_k=d_k,
            use_landmark=use_landmark_stabilization,
            mode=landmark_mode,
        )

        # Episodic memory
        self.memory_module = TEMMemory(
            max_memory=max_memory,
            d_g=d_g,
            d_k=d_k,
            d_v=d_v,
            n_sensory=n_sensory,
            dedup=memory_dedup,
        )

        # TEM attention for stable-position prediction (optional consistency check)
        self.stable_attention = TEMAttention(
            d_k=d_k,
            beta0=beta0,
            adaptive_beta=True,
        )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.FloatTensor:
        """Return initial position encoding g_0.

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
        """
        return self.position_encoder.init_state(batch_size, device, dtype)

    def init_memory(
        self,
        g_0: torch.FloatTensor,
        x_0: torch.FloatTensor,
        states_0: Optional[torch.LongTensor] = None,
    ) -> MemoryState:
        """Initialize memory with the first observation-position pair.

        Parameters
        ----------
        g_0 : FloatTensor [B, d_g]
            Initial position encoding.
        x_0 : FloatTensor [B, N_x]
            Initial sensory observation (one-hot).
        states_0 : Optional LongTensor [B]
            Ground-truth state indices.

        Returns
        -------
        MemoryState with size = 1.
        """
        key_g_0 = self.position_proj(g_0)
        value_x_0 = self.sensory_proj(x_0)

        return self.memory_module.init_from_observation(
            g_0=g_0,
            x_0=x_0,
            key_g_0=key_g_0,
            value_x_0=value_x_0,
            states_0=states_0,
        )

    # ------------------------------------------------------------------
    # predict_next: prediction phase (NO x_{t+1})
    # ------------------------------------------------------------------
    def predict_next(
        self,
        g_t: torch.FloatTensor,
        memory_t: MemoryState,
        actions_t: torch.LongTensor,
    ) -> PredictionState:
        """Predict the next sensory observation using only g_t and action.

        This is the STRICT online prediction step. It MUST NOT access
        x_{t+1} in any form. The prediction is made by:
          1. Path-integrating g_t with the action to get g_{t+1}^{PI}.
          2. Using g_{t+1}^{PI} as an attention query over memory.
          3. Reading out the retrieved sensory value and classifying it.

        Parameters
        ----------
        g_t : FloatTensor [B, d_g]
            Current stabilised position encoding.
        memory_t : MemoryState
            Current episodic memory (contains up to time t).
        actions_t : LongTensor [B]
            Action taken at time t.

        Returns
        -------
        PredictionState
        """
        # Step A: path integration
        # g_{t+1}^{PI} = sigma(g_t @ W_{a_t})
        g_pi = self.position_encoder(g_t, actions_t)  # [B, d_g]

        # Project as attention query
        q_pi = self.position_proj(g_pi)  # [B, d_k]

        # Attention: Q from g_pi, K from historical g, V from historical x
        attn_pi = self.sensory_attention(
            query=q_pi,
            keys=memory_t.keys_g,
            values=memory_t.values_x,
            valid_mask=memory_t.valid_mask,
            memory_size=memory_t.size,
        )

        # Sensory prediction
        logits_pi = self.pred_head(attn_pi.read)  # [B, N_x]
        probs_pi = torch.softmax(logits_pi, dim=-1)

        return PredictionState(
            g_pi=g_pi,
            q_pi=q_pi,
            attn_pi=attn_pi,
            logits_pi=logits_pi,
            probs_pi=probs_pi,
        )

    # ------------------------------------------------------------------
    # observe_next: observation / correction phase (uses x_{t+1})
    # ------------------------------------------------------------------
    def observe_next(
        self,
        g_pi: torch.FloatTensor,
        memory_t: MemoryState,
        x_next: torch.FloatTensor,
        states_next: Optional[torch.LongTensor] = None,
        compute_stable_prediction: bool = True,
    ) -> ObservationState:
        """Process the observed sensory input to correct position and write memory.

        1. Retrieve position from sensory landmark (optional).
        2. Fuse PI and retrieved position to get stabilised g_{t+1}.
        3. Optionally re-query memory with stabilised g for consistency.
        4. Write new observation-position pair to memory.

        Parameters
        ----------
        g_pi : FloatTensor [B, d_g]
            Path-integrated position g_{t+1}^{PI} from predict_next.
        memory_t : MemoryState
            Memory before observing x_{t+1}.
        x_next : FloatTensor [B, N_x]
            Observed sensory input at time t+1 (one-hot).
        states_next : Optional LongTensor [B]
            Ground-truth state for the new observation.
        compute_stable_prediction : bool, optional
            Whether to compute the stable-position prediction. Default True.

        Returns
        -------
        ObservationState
        """
        B = g_pi.shape[0]
        device = g_pi.device

        # Project x_next to sensory value space
        value_x_next = self.sensory_proj(x_next)  # [B, d_v]

        # ---- Landmark-based position retrieval ----
        g_retrieved, attn_landmark = self.landmark_stabilizer.retrieve_position(
            x_next=x_next,
            g_pi=g_pi,
            memory=memory_t,
            sensory_value_x=value_x_next,
        )

        # ---- Fuse PI and retrieved position ----
        g_next, eta = self.landmark_stabilizer(g_pi, g_retrieved)

        # ---- Optional stable-position prediction ----
        attn_stable = None
        logits_stable = None
        probs_stable = None

        if compute_stable_prediction:
            q_stable = self.position_proj(g_next)
            attn_stable = self.stable_attention(
                query=q_stable,
                keys=memory_t.keys_g,
                values=memory_t.values_x,
                valid_mask=memory_t.valid_mask,
                memory_size=memory_t.size,
            )
            logits_stable = self.pred_head(attn_stable.read)
            probs_stable = torch.softmax(logits_stable, dim=-1)

        # ---- Write to memory ----
        key_g_next = self.position_proj(g_next)

        memory_next, wrote_mask = self.memory_module.write(
            memory=memory_t,
            key_g_new=key_g_next,
            value_x_new=value_x_next,
            g_new=g_next,
            x_new=x_next,
            states_new=states_next,
        )

        return ObservationState(
            g_retrieved=g_retrieved,
            eta=eta,
            g_next=g_next,
            attn_landmark=attn_landmark,
            attn_stable=attn_stable,
            logits_stable=logits_stable,
            probs_stable=probs_stable,
            memory_next=memory_next,
            wrote_mask=wrote_mask,
        )

    # ------------------------------------------------------------------
    # step: single-step predict + observe
    # ------------------------------------------------------------------
    def step(
        self,
        g_t: torch.FloatTensor,
        memory_t: MemoryState,
        actions_t: torch.LongTensor,
        x_next: torch.FloatTensor,
        states_next: Optional[torch.LongTensor] = None,
        compute_stable_prediction: bool = True,
    ) -> TEMTStepOutput:
        """Execute one full step of the TEM-t online protocol.

        Internally calls predict_next then observe_next in sequence.

        Parameters
        ----------
        g_t : FloatTensor [B, d_g]
            Current position encoding.
        memory_t : MemoryState
            Current memory.
        actions_t : LongTensor [B]
            Action taken at time t.
        x_next : FloatTensor [B, N_x]
            Sensory observation at time t+1.
        states_next : Optional LongTensor [B]
            Ground-truth state at t+1.
        compute_stable_prediction : bool, optional
            Whether to compute stable prediction. Default True.

        Returns
        -------
        TEMTStepOutput
        """
        prediction = self.predict_next(g_t, memory_t, actions_t)
        observation = self.observe_next(
            g_pi=prediction.g_pi,
            memory_t=memory_t,
            x_next=x_next,
            states_next=states_next,
            compute_stable_prediction=compute_stable_prediction,
        )
        return TEMTStepOutput(prediction=prediction, observation=observation)

    # ------------------------------------------------------------------
    # forward: full sequence
    # ------------------------------------------------------------------
    def forward(
        self,
        batch: TrajectoryBatch,
        return_traces: bool = True,
        compute_stable_prediction: bool = True,
    ) -> TEMTForwardOutput:
        """Execute the full TEM-t forward pass over a trajectory batch.

        Iterates t = 0 .. T-1:
          prediction[t] = predict_next(g_t, memory_t, actions[:, t])
          observation[t] = observe_next(prediction.g_pi, memory_t, x[:, t+1])
          g_t = observation.g_next
          memory_t = observation.memory_next

        Parameters
        ----------
        batch : TrajectoryBatch
            Batch of trajectories.
        return_traces : bool, optional
            If True, collect intermediate tensors for analysis. Default True.
        compute_stable_prediction : bool, optional
            Whether to compute stable-position predictions. Default True.

        Returns
        -------
        TEMTForwardOutput
        """
        B = batch.x.shape[0]
        T = batch.actions.shape[1]
        device = batch.x.device

        # Initialize
        g_t = self.init_state(B, device, batch.x.dtype)
        states_0 = batch.states[:, 0] if batch.states is not None else None
        memory_t = self.init_memory(g_t, batch.x[:, 0], states_0)

        # Accumulators
        g_seq_list = [g_t]                                    # g_0
        g_pi_seq_list = []
        logits_pi_list = []
        probs_pi_list = []
        logits_stable_list = []
        probs_stable_list = []
        attn_pi_list = []
        attn_stable_list = []
        eta_seq_list = []
        memory_sizes_list = [memory_t.size.clone()]            # size after step 0

        for t in range(T):
            a_t = batch.actions[:, t]                          # [B]
            x_next = batch.x[:, t + 1]                          # [B, N_x]
            states_next = (
                batch.states[:, t + 1]
                if batch.states is not None
                else None
            )

            step_out = self.step(
                g_t=g_t,
                memory_t=memory_t,
                actions_t=a_t,
                x_next=x_next,
                states_next=states_next,
                compute_stable_prediction=compute_stable_prediction,
            )

            # Collect
            g_pi_seq_list.append(step_out.prediction.g_pi)
            logits_pi_list.append(step_out.prediction.logits_pi)
            probs_pi_list.append(step_out.prediction.probs_pi)

            if return_traces:
                attn_pi_list.append(step_out.prediction.attn_pi.weights)
                eta_seq_list.append(step_out.observation.eta)

                if step_out.observation.attn_stable is not None:
                    attn_stable_list.append(step_out.observation.attn_stable.weights)

            if step_out.observation.logits_stable is not None:
                logits_stable_list.append(step_out.observation.logits_stable)
                probs_stable_list.append(step_out.observation.probs_stable)

            # Advance state
            g_t = step_out.observation.g_next
            memory_t = step_out.observation.memory_next

            g_seq_list.append(g_t)
            memory_sizes_list.append(memory_t.size.clone())

        # Stack accumulated tensors
        g_seq = torch.stack(g_seq_list, dim=1)                  # [B, T+1, d_g]
        g_pi_seq = torch.stack(g_pi_seq_list, dim=1)            # [B, T, d_g]
        logits_pi = torch.stack(logits_pi_list, dim=1)          # [B, T, N_x]
        probs_pi = torch.stack(probs_pi_list, dim=1)            # [B, T, N_x]
        memory_sizes = torch.stack(memory_sizes_list, dim=1)    # [B, T+1]

        # Optional tensors
        logits_stable = (
            torch.stack(logits_stable_list, dim=1)
            if logits_stable_list
            else None
        )
        probs_stable = (
            torch.stack(probs_stable_list, dim=1)
            if probs_stable_list
            else None
        )
        attn_pi = torch.stack(attn_pi_list, dim=1) if attn_pi_list else None
        attn_stable = (
            torch.stack(attn_stable_list, dim=1) if attn_stable_list else None
        )
        eta_seq = torch.stack(eta_seq_list, dim=1) if eta_seq_list else None

        # Extra debug info
        extra: Dict[str, Any] = {}

        return TEMTForwardOutput(
            logits_pi=logits_pi,
            probs_pi=probs_pi,
            logits_stable=logits_stable,
            probs_stable=probs_stable,
            g_seq=g_seq,
            g_pi_seq=g_pi_seq,
            eta_seq=eta_seq,
            attn_pi=attn_pi,
            attn_stable=attn_stable,
            memory_sizes=memory_sizes,
            final_memory=memory_t,
            extra=extra,
        )
