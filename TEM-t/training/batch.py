"""Standardised batch data structure for TEM-t training and evaluation.

Defines TrajectoryBatch — the single input/output container consumed
by TEMTModel.forward and produced by TrajectorySampler.
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class TrajectoryBatch:
    """A minibatch of agent trajectories in 2D grid environments.

    Each trajectory is a sequence of sensory observations and actions.
    Ground-truth state labels are carried for evaluation only and MUST
    NOT be used by the model core (predict_next, recurrent_position, etc.).

    Attributes
    ----------
    x_ids : LongTensor [B, T+1]
        Sensory class index per time step.
    x : FloatTensor [B, T+1, N_x]
        One-hot encoding of sensory observation per time step.
    actions : LongTensor [B, T]
        Action index per transition (t -> t+1).
    states : Optional LongTensor [B, T+1]
        Ground-truth graph node index per time step.
        Used only for zero-shot evaluation, rate maps, and gridness.
    env_ids : Optional LongTensor [B]
        Environment index for each trajectory in the batch.
    valid_mask : Optional BoolTensor [B, T]
        True for valid transitions; None if no padding.

    Protocol
    --------
    x[:, 0]          — initial observation (no preceding action).
    actions[:, t]    — action taken from state at time t to time t+1.
    x[:, t+1]        — observation at time t+1, target for prediction.
    logits_pi[:, t]  — prediction for x[:, t+1] using g_{t+1}^{PI}.
    zero_shot[:, t]  — whether transition (states[t], actions[t]) is novel
                        but states[t+1] was already visited.
    """

    x_ids: torch.LongTensor
    x: torch.FloatTensor
    actions: torch.LongTensor

    states: Optional[torch.LongTensor] = None
    env_ids: Optional[torch.LongTensor] = None
    valid_mask: Optional[torch.BoolTensor] = None
