"""Evaluation utilities for TEM-t.

Provides the TEMTEvaluator class that computes:
  - Standard sensory prediction accuracy
  - Zero-shot prediction accuracy (novel edges to known nodes)
"""

from dataclasses import dataclass
from typing import Dict

import torch
from tqdm import tqdm

from model.temt import TEMTModel
from training.envs import TrajectorySampler
from training.metrics import categorical_accuracy, compute_zero_shot_mask


# ---------------------------------------------------------------------------
# EvaluationOutput
# ---------------------------------------------------------------------------
@dataclass
class EvaluationOutput:
    """Result of an evaluation run.

    Attributes
    ----------
    metrics : Dict[str, float]
        Scalar metrics (accuracy, loss, etc.).
    tensors : Dict[str, torch.Tensor]
        Collected tensors for further analysis.
    """

    metrics: Dict[str, float]
    tensors: Dict[str, torch.Tensor]


# ---------------------------------------------------------------------------
# TEMTEvaluator
# ---------------------------------------------------------------------------
class TEMTEvaluator:
    """Evaluator for TEM-t model predictions.

    Parameters
    ----------
    model : TEMTModel
        The trained TEM-t model.
    sampler : TrajectorySampler
        Sampler for evaluation trajectories (should use test environments).
    device : torch.device
        Computation device.
    """

    def __init__(
        self,
        model: TEMTModel,
        sampler: TrajectorySampler,
        device: torch.device,
    ) -> None:
        self.model = model
        self.sampler = sampler
        self.device = device

    # ------------------------------------------------------------------
    # evaluate_prediction: standard accuracy
    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate_prediction(
        self,
        batch_size: int,
        n_batches: int,
    ) -> EvaluationOutput:
        """Evaluate standard sensory prediction accuracy.

        Uses logits_pi (path-integration prediction) only.

        Parameters
        ----------
        batch_size : int
            Batch size for evaluation.
        n_batches : int
            Number of evaluation batches.

        Returns
        -------
        EvaluationOutput
        """
        self.model.eval()

        total_correct = 0
        total_steps = 0
        total_loss = 0.0

        for _ in tqdm(range(n_batches), desc="Evaluating", unit="batch", dynamic_ncols=True):
            batch = self.sampler.sample_batch(batch_size, self.device)

            output = self.model(
                batch, return_traces=False, compute_stable_prediction=False
            )

            # Accuracy
            preds = output.logits_pi.argmax(dim=-1)  # [B, T]
            targets = batch.x_ids[:, 1:]               # [B, T]
            correct = (preds == targets).float()
            total_correct += correct.sum().item()
            total_steps += correct.numel()

            # Approximate loss
            logits_flat = output.logits_pi.reshape(-1, output.logits_pi.shape[-1])
            targets_flat = targets.reshape(-1)
            loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
            total_loss += loss.item()

        acc_all = total_correct / total_steps if total_steps > 0 else 0.0
        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0

        return EvaluationOutput(
            metrics={
                "acc_pi": acc_all,
                "loss_pi": avg_loss,
            },
            tensors={},
        )

    # ------------------------------------------------------------------
    # evaluate_zero_shot: zero-shot accuracy
    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate_zero_shot(
        self,
        batch_size: int,
        n_batches: int,
    ) -> EvaluationOutput:
        """Evaluate zero-shot sensory prediction accuracy.

        Zero-shot transitions are those where the edge (s_t, a_t) has
        not been traversed before, but the destination s_{t+1} has been
        visited. Correct prediction on these steps requires the model
        to have learned the abstract transition structure.

        Uses logits_pi (pre-observation prediction) to avoid leakage.

        Parameters
        ----------
        batch_size : int
            Batch size for evaluation.
        n_batches : int
            Number of evaluation batches.

        Returns
        -------
        EvaluationOutput
            Metrics include acc_all, acc_zero_shot, n_zero_shot.
        """
        self.model.eval()

        total_zs_correct = 0
        total_zs_steps = 0
        total_all_correct = 0
        total_all_steps = 0

        for _ in tqdm(range(n_batches), desc="Eval zero-shot", unit="batch", dynamic_ncols=True):
            batch = self.sampler.sample_batch(batch_size, self.device)

            output = self.model(
                batch, return_traces=False, compute_stable_prediction=False
            )

            # Predictions
            preds = output.logits_pi.argmax(dim=-1)  # [B, T]
            targets = batch.x_ids[:, 1:]               # [B, T]

            # All-step accuracy
            correct = (preds == targets)
            total_all_correct += correct.sum().item()
            total_all_steps += correct.numel()

            # Zero-shot mask
            if batch.states is None:
                continue

            zs_mask = compute_zero_shot_mask(batch.states, batch.actions)  # [B, T]
            zs_correct = correct & zs_mask

            total_zs_correct += zs_correct.sum().item()
            total_zs_steps += zs_mask.sum().item()

        acc_all = total_all_correct / total_all_steps if total_all_steps > 0 else 0.0
        acc_zs = total_zs_correct / total_zs_steps if total_zs_steps > 0 else 0.0

        return EvaluationOutput(
            metrics={
                "acc_all": acc_all,
                "acc_zero_shot": acc_zs,
                "n_zero_shot": total_zs_steps,
            },
            tensors={},
        )
