"""TEM-t model internal data structures.

Defines the core dataclasses used throughout the TEM-t pipeline:
memory state, attention output, prediction/observation states,
and the full forward-output container.

All tensors follow the batch-first convention: [B, ...].
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch


@dataclass
class AttentionOutput:
    """Result of a single TEM attention (memory-retrieval) operation.

    Attributes
    ----------
    read : FloatTensor [B, d_value]
        The retrieved value vector, computed as weighted sum of values.
    weights : FloatTensor [B, M]
        Attention weights over memory slots (softmax output).
        Valid memory slots sum to 1; invalid slots are zero.
    scores : FloatTensor [B, M]
        Raw attention logits before softmax.
    """

    read: torch.FloatTensor
    weights: torch.FloatTensor
    scores: torch.FloatTensor


@dataclass
class MemoryState:
    """TEM-t episodic memory state at a given time step.

    Stores the history of position-sensory bindings. Memory is
    implemented as fixed-size tensors of max length M = T + 1
    with a validity mask, enabling batched forward passes.

    Attributes
    ----------
    keys_g : FloatTensor [B, M, d_k]
        Projected position keys used for attention retrieval.
    values_x : FloatTensor [B, M, d_v]
        Projected sensory values read by attention.
    values_g : FloatTensor [B, M, d_g]
        Raw position representations for landmark stabilization.
    raw_x : FloatTensor [B, M, N_x]
        Raw one-hot sensory observations (for dedup or debugging).
    valid_mask : BoolTensor [B, M]
        True for occupied memory slots, False for empty ones.
    size : LongTensor [B]
        Number of currently occupied memory slots per batch item.
    raw_states : Optional LongTensor [B, M]
        Ground-truth state indices, only set during evaluation.
    """

    keys_g: torch.FloatTensor
    values_x: torch.FloatTensor
    values_g: torch.FloatTensor
    raw_x: torch.FloatTensor
    valid_mask: torch.BoolTensor
    size: torch.LongTensor
    raw_states: Optional[torch.LongTensor] = None


@dataclass
class PredictionState:
    """Output of predict_next — prediction BEFORE observing x_{t+1}.

    This is the strict online prediction: it only uses g_t, memory_t,
    and a_t. It MUST NOT access x_{t+1} in any form.

    Attributes
    ----------
    g_pi : FloatTensor [B, d_g]
        Path-integrated position g_{t+1}^{PI}.
    q_pi : FloatTensor [B, d_k]
        Projected query from g_{t+1}^{PI}.
    attn_pi : AttentionOutput
        Attention result using g_{t+1}^{PI} as query on existing memory.
    logits_pi : FloatTensor [B, N_x]
        Sensory prediction logits (before softmax).
    probs_pi : FloatTensor [B, N_x]
        Sensory prediction probabilities (after softmax).
    """

    g_pi: torch.FloatTensor
    q_pi: torch.FloatTensor
    attn_pi: AttentionOutput
    logits_pi: torch.FloatTensor
    probs_pi: torch.FloatTensor


@dataclass
class ObservationState:
    """Output of observe_next — position correction AFTER observing x_{t+1}.

    Uses the observed sensory input as a landmark to stabilize the
    path-integrated position, then optionally re-queries memory with
    the stabilized position for a consistency prediction.

    Attributes
    ----------
    g_retrieved : FloatTensor [B, d_g]
        Position retrieved from sensory landmark lookup.
    eta : FloatTensor [B, d_g]
        Per-dimension gate between g_pi and g_retrieved.
    g_next : FloatTensor [B, d_g]
        Stabilized position g_{t+1} after landmark correction.
    attn_landmark : Optional AttentionOutput
        Attention produced during sensory landmark lookup.
    attn_stable : Optional AttentionOutput
        Attention using stabilized g_{t+1} to re-read sensory memory.
    logits_stable : Optional FloatTensor [B, N_x]
        Consistency sensory prediction logits.
    probs_stable : Optional FloatTensor [B, N_x]
        Consistency sensory prediction probabilities.
    memory_next : MemoryState
        Updated memory after writing current observation-position pair.
    wrote_mask : BoolTensor [B]
        Whether a new memory slot was written at this step.
    """

    g_retrieved: torch.FloatTensor
    eta: torch.FloatTensor
    g_next: torch.FloatTensor
    attn_landmark: Optional[AttentionOutput]
    attn_stable: Optional[AttentionOutput]
    logits_stable: Optional[torch.FloatTensor]
    probs_stable: Optional[torch.FloatTensor]
    memory_next: MemoryState
    wrote_mask: torch.BoolTensor


@dataclass
class TEMTStepOutput:
    """Complete single-step TEM-t output.

    Encapsulates both phases of the online prediction protocol:
    first predict_next (without x_{t+1}), then observe_next (with x_{t+1}).

    Attributes
    ----------
    prediction : PredictionState
        predict_next output — uses g_t, memory_t, a_t only.
    observation : ObservationState
        observe_next output — uses x_{t+1} for correction.
    """

    prediction: PredictionState
    observation: ObservationState


@dataclass
class TEMTForwardOutput:
    """TEM-t full-sequence forward pass result.

    Collected outputs across all T time steps of a trajectory.

    Attributes
    ----------
    logits_pi : FloatTensor [B, T, N_x]
        PI-based sensory prediction logits (one per step).
    probs_pi : FloatTensor [B, T, N_x]
        PI-based sensory prediction probabilities.
    logits_stable : Optional FloatTensor [B, T, N_x]
        Stabilized-position prediction logits.
    probs_stable : Optional FloatTensor [B, T, N_x]
        Stabilized-position prediction probabilities.
    g_seq : FloatTensor [B, T+1, d_g]
        Stabilized position encodings g_0 ... g_T.
    g_pi_seq : FloatTensor [B, T, d_g]
        Path-integrated position encodings g_1^{PI} ... g_T^{PI}.
    eta_seq : Optional FloatTensor [B, T, d_g]
        Landmark correction gates per step.
    attn_pi : Optional FloatTensor [B, T, M]
        Attention weights from PI predictions.
    attn_stable : Optional FloatTensor [B, T, M]
        Attention weights from stabilized predictions.
    memory_sizes : LongTensor [B, T+1]
        Memory size after each time step.
    final_memory : MemoryState
        Memory state after processing the full sequence.
    extra : Dict[str, Any]
        Debugging / auxiliary tensors.
    """

    logits_pi: torch.FloatTensor
    probs_pi: torch.FloatTensor
    logits_stable: Optional[torch.FloatTensor]
    probs_stable: Optional[torch.FloatTensor]
    g_seq: torch.FloatTensor
    g_pi_seq: torch.FloatTensor
    eta_seq: Optional[torch.FloatTensor]
    attn_pi: Optional[torch.FloatTensor]
    attn_stable: Optional[torch.FloatTensor]
    memory_sizes: torch.LongTensor
    final_memory: MemoryState
    extra: Dict[str, Any]
