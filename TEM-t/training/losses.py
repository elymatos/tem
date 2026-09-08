"""Loss functions for TEM-t training.

Implements the composite training objective:

    L = lambda_pi * L_pi
      + lambda_stable * L_stable
      + lambda_g * L_g
      + lambda_weight * L_weight
      + lambda_g_l2 * L_g_l2

where:
    L_pi     — cross-entropy of PI-based prediction (core online loss)
    L_stable — cross-entropy of stabilised-position prediction
    L_g      — MSE between stabilised g and PI g (consistency)
    L_weight — L2 weight decay
    L_g_l2   — L2 penalty on position encoding norm
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.batch import TrajectoryBatch
from model.types import TEMTForwardOutput


# ---------------------------------------------------------------------------
# LossOutput
# ---------------------------------------------------------------------------
@dataclass
class LossOutput:
    """Aggregated loss computation result.

    Attributes
    ----------
    total : Tensor []
        Scalar total loss (sum of weighted terms).
    terms : Dict[str, Tensor]
        Individual loss terms keyed by name (each a scalar Tensor).
    metrics : Dict[str, Tensor]
        Auxiliary metrics computed during loss evaluation (e.g. accuracy).
    """

    total: torch.Tensor
    terms: Dict[str, torch.Tensor]
    metrics: Dict[str, torch.Tensor]


# ---------------------------------------------------------------------------
# TEMTLoss
# ---------------------------------------------------------------------------
class TEMTLoss(nn.Module):
    """Composite training loss for TEM-t.

    Parameters
    ----------
    lambda_pi : float, optional
        Weight for PI-based prediction CE loss. Default 1.0.
    lambda_stable : float, optional
        Weight for stabilised-position prediction CE loss. Default 1.0.
    lambda_g : float, optional
        Weight for g-consistency MSE loss. Default 0.1.
    lambda_weight : float, optional
        Weight for L2 parameter decay. Default 1e-5.
    lambda_g_l2 : float, optional
        Weight for g-norm penalty. Default 1e-4.
    """

    def __init__(
        self,
        lambda_pi: float = 1.0,
        lambda_stable: float = 1.0,
        lambda_g: float = 0.1,
        lambda_weight: float = 1e-5,
        lambda_g_l2: float = 1e-4,
    ) -> None:
        super().__init__()
        self.lambda_pi = lambda_pi
        self.lambda_stable = lambda_stable
        self.lambda_g = lambda_g
        self.lambda_weight = lambda_weight
        self.lambda_g_l2 = lambda_g_l2

    def forward(
        self,
        output: TEMTForwardOutput,
        batch: TrajectoryBatch,
        model: Optional[nn.Module] = None,
    ) -> LossOutput:
        """Compute the composite TEM-t loss.

        Parameters
        ----------
        output : TEMTForwardOutput
            Model forward pass result.
        batch : TrajectoryBatch
            Input batch (provides target x_ids).
        model : Optional nn.Module
            Model whose parameters are penalised (for L_weight).

        Returns
        -------
        LossOutput
        """
        # Target sensory IDs: x_ids[:, 1:] — the sensory class at t+1
        targets = batch.x_ids[:, 1:]  # [B, T]

        B, T = targets.shape
        device = targets.device

        # ---- L_pi: PI-based prediction cross-entropy ----
        # logits_pi [B, T, N_x] -> flatten to [B*T, N_x]
        logits_pi_flat = output.logits_pi.reshape(-1, output.logits_pi.shape[-1])
        targets_flat = targets.reshape(-1)

        # Apply valid mask if present
        if batch.valid_mask is not None:
            mask_flat = batch.valid_mask.reshape(-1)
            ce_pi = F.cross_entropy(logits_pi_flat, targets_flat, reduction="none")
            loss_pi = (ce_pi * mask_flat.float()).sum() / (mask_flat.sum() + 1e-8)
        else:
            loss_pi = F.cross_entropy(logits_pi_flat, targets_flat)
        loss_pi = self.lambda_pi * loss_pi

        # ---- L_stable: stabilised-position prediction CE ----
        loss_stable = torch.tensor(0.0, device=device)
        if output.logits_stable is not None:
            logits_st_flat = output.logits_stable.reshape(-1, output.logits_stable.shape[-1])
            if batch.valid_mask is not None:
                ce_st = F.cross_entropy(logits_st_flat, targets_flat, reduction="none")
                loss_stable = (ce_st * mask_flat.float()).sum() / (mask_flat.sum() + 1e-8)
            else:
                loss_stable = F.cross_entropy(logits_st_flat, targets_flat)
        loss_stable = self.lambda_stable * loss_stable

        # ---- L_g: consistency between stabilised g and PI g ----
        # g_seq[:, 1:] [B, T, d_g] vs g_pi_seq [B, T, d_g]
        diff = output.g_seq[:, 1:] - output.g_pi_seq  # [B, T, d_g]
        if batch.valid_mask is not None:
            mask_expanded = batch.valid_mask.unsqueeze(-1).float()  # [B, T, 1]
            loss_g = (diff.pow(2) * mask_expanded).sum() / (mask_expanded.sum() * diff.shape[-1] + 1e-8)
        else:
            loss_g = diff.pow(2).mean()
        loss_g = self.lambda_g * loss_g

        # ---- L_weight: L2 weight decay ----
        loss_weight = torch.tensor(0.0, device=device)
        if model is not None and self.lambda_weight > 0:
            l2_sum = torch.tensor(0.0, device=device)
            for p in model.parameters():
                l2_sum = l2_sum + p.pow(2).sum()
            loss_weight = self.lambda_weight * l2_sum

        # ---- L_g_l2: penalty on position encoding norm ----
        loss_g_l2 = self.lambda_g_l2 * output.g_seq.pow(2).mean()

        # ---- Total ----
        total = loss_pi + loss_stable + loss_g + loss_weight + loss_g_l2

        # ---- Auxiliary metrics ----
        with torch.no_grad():
            pred_pi = logits_pi_flat.argmax(dim=-1)  # [B*T]
            acc_pi = (pred_pi == targets_flat).float().mean()

        return LossOutput(
            total=total,
            terms={
                "loss_pi": loss_pi.detach(),
                "loss_stable": loss_stable.detach(),
                "loss_g": loss_g.detach(),
                "loss_weight": loss_weight.detach(),
                "loss_g_l2": loss_g_l2.detach(),
            },
            metrics={
                "acc_pi": acc_pi.detach(),
            },
        )
