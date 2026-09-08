"""TEM-t training loop with checkpointing.

Provides the Trainer class that orchestrates:
  - batch sampling
  - forward pass + loss computation
  - backpropagation and optimizer step
  - periodic evaluation and logging
  - checkpoint save/load
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import os
import json
import time

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from training.batch import TrajectoryBatch
from training.losses import TEMTLoss, LossOutput
from training.envs import TrajectorySampler
from training.evaluator import TEMTEvaluator, EvaluationOutput

from model.temt import TEMTModel


# ---------------------------------------------------------------------------
# TrainState
# ---------------------------------------------------------------------------
@dataclass
class TrainState:
    """Mutable training progress tracker.

    Attributes
    ----------
    global_step : int
        Number of gradient updates performed.
    epoch : int
        Number of full dataset passes (conceptual; not strictly enforced).
    best_metric : Optional float
        Best validation metric observed so far.
    """

    global_step: int = 0
    epoch: int = 0
    best_metric: Optional[float] = None


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class Trainer:
    """TEM-t training loop.

    Parameters
    ----------
    model : TEMTModel
        The TEM-t model to train.
    loss_fn : TEMTLoss
        Composite loss function.
    optimizer : torch.optim.Optimizer
        Optimizer instance.
    train_sampler : TrajectorySampler
        Sampler for training trajectories.
    val_sampler : Optional TrajectorySampler
        Sampler for validation trajectories.
    evaluator : Optional TEMTEvaluator
        Evaluator for computing metrics.
    device : torch.device
        Computation device.
    exp_dir : str
        Experiment directory for checkpoints and logs.
    grad_clip_norm : Optional float
        Maximum gradient norm (clipping). None disables clipping.
    """

    def __init__(
        self,
        model: TEMTModel,
        loss_fn: TEMTLoss,
        optimizer: torch.optim.Optimizer,
        train_sampler: TrajectorySampler,
        val_sampler: Optional[TrajectorySampler],
        evaluator: Optional[TEMTEvaluator],
        device: torch.device,
        exp_dir: str,
        grad_clip_norm: Optional[float] = 1.0,
    ) -> None:
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.evaluator = evaluator
        self.device = device
        self.exp_dir = exp_dir
        self.grad_clip_norm = grad_clip_norm

        # Ensure directories
        os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)

        self._train_log_path = os.path.join(exp_dir, "logs", "train_metrics.jsonl")
        self._eval_log_path = os.path.join(exp_dir, "logs", "eval_metrics.jsonl")

        # Mixed-precision scaler (GPU only; no-op on CPU)
        self._use_amp = (device.type == "cuda")
        self.scaler = GradScaler() if self._use_amp else None

    # ------------------------------------------------------------------
    # Single training step
    # ------------------------------------------------------------------
    def train_step(self, batch: TrajectoryBatch) -> LossOutput:
        """Execute one gradient update.

        1. Forward pass through the model.
        2. Compute composite loss.
        3. Backpropagate.
        4. Clip gradients (if configured).
        5. Step the optimizer.

        Parameters
        ----------
        batch : TrajectoryBatch
            Training batch.

        Returns
        -------
        LossOutput
            Loss values and metrics for logging.
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Mixed-precision forward (autocast off for CPU)
        with autocast('cuda') if self._use_amp else torch.enable_grad():
            output = self.model(
                batch, return_traces=True, compute_stable_prediction=True
            )
            loss_out = self.loss_fn(output, batch, self.model)

        # Backward with gradient scaling
        if self._use_amp:
            self.scaler.scale(loss_out.total).backward()
            if self.grad_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss_out.total.backward()
            if self.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()

        return loss_out

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _evaluate(
        self,
        batch_size: int,
        n_batches: int,
    ) -> EvaluationOutput:
        """Run evaluation on validation data.

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
        if self.evaluator is None:
            return EvaluationOutput(metrics={}, tensors={})

        self.model.eval()
        return self.evaluator.evaluate_prediction(batch_size, n_batches)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    def train(
        self,
        num_updates: int,
        batch_size: int,
        eval_interval: int = 500,
        log_interval: int = 100,
        save_interval: int = 1000,
    ) -> TrainState:
        """Run the main training loop.

        Parameters
        ----------
        num_updates : int
            Total number of gradient updates.
        batch_size : int
            Training batch size.
        eval_interval : int
            Evaluate every N gradient steps.
        log_interval : int
            Log metrics every N gradient steps.
        save_interval : int
            Save checkpoint every N gradient steps.

        Returns
        -------
        TrainState
            Final training state.
        """
        state = TrainState()

        # Use tqdm for a compact progress bar; print eval/save events as they occur.
        pbar = tqdm(
            total=num_updates,
            desc="Training",
            unit="step",
            dynamic_ncols=True,
        )

        for step in range(num_updates):
            t_start = time.time()

            # Sample training batch
            batch = self.train_sampler.sample_batch(batch_size, self.device)

            # Train step
            loss_out = self.train_step(batch)

            state.global_step += 1
            dt = time.time() - t_start

            # Update progress bar
            acc = loss_out.metrics.get("acc_pi", torch.tensor(0.0))
            pbar.set_postfix({
                "loss": f"{loss_out.total.item():.3f}",
                "acc": f"{acc.item():.3f}",
            })
            pbar.update(1)

            # Logging (detailed JSONL)
            if state.global_step % log_interval == 0:
                log_entry = {
                    "global_step": state.global_step,
                    **{k: v.item() for k, v in loss_out.terms.items()},
                    **{k: v.item() for k, v in loss_out.metrics.items()},
                    "total": loss_out.total.item(),
                    "dt": dt,
                }
                os.makedirs(os.path.dirname(self._train_log_path), exist_ok=True)
                with open(self._train_log_path, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

            # Evaluation
            if state.global_step % eval_interval == 0:
                pbar.write(f"--- Eval @ step {state.global_step} ---")
                eval_out = self._evaluate(batch_size, n_batches=10)
                eval_log = {
                    "global_step": state.global_step,
                    **eval_out.metrics,
                }
                os.makedirs(os.path.dirname(self._eval_log_path), exist_ok=True)
                with open(self._eval_log_path, "a") as f:
                    f.write(json.dumps(eval_log) + "\n")

                # Track best
                if "acc_pi" in eval_out.metrics:
                    acc_val = eval_out.metrics["acc_pi"]
                    if state.best_metric is None or acc_val > state.best_metric:
                        state.best_metric = acc_val
                        self.save_checkpoint(
                            os.path.join(self.exp_dir, "checkpoints", "best.pt"),
                            state,
                        )

                pbar.write(
                    "  " + " ".join(f"{k}={v:.4f}" for k, v in eval_out.metrics.items())
                )

            # Save checkpoint
            if state.global_step % save_interval == 0:
                self.save_checkpoint(
                    os.path.join(
                        self.exp_dir, "checkpoints", f"step_{state.global_step:06d}.pt"
                    ),
                    state,
                )
                self.save_checkpoint(
                    os.path.join(self.exp_dir, "checkpoints", "latest.pt"),
                    state,
                )

        pbar.close()

        # Final save
        self.save_checkpoint(
            os.path.join(self.exp_dir, "checkpoints", "latest.pt"),
            state,
        )

        return state

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def save_checkpoint(
        self,
        path: str,
        train_state: TrainState,
    ) -> None:
        """Save model, optimizer, and training state to disk.

        Parameters
        ----------
        path : str
            File path for the checkpoint (.pt file).
        train_state : TrainState
            Current training state.
        """
        rng_state = {
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_state["cuda"] = torch.cuda.get_rng_state_all()

        checkpoint: Dict[str, Any] = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": train_state.global_step,
            "epoch": train_state.epoch,
            "best_metric": train_state.best_metric,
            "rng_state": rng_state,
        }
        if self._use_amp and self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)

    def load_checkpoint(
        self,
        path: str,
    ) -> TrainState:
        """Load model, optimizer, and training state from disk.

        Parameters
        ----------
        path : str
            File path for the checkpoint (.pt file).

        Returns
        -------
        TrainState
            Restored training state.
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self._use_amp and self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        state = TrainState(
            global_step=checkpoint["global_step"],
            epoch=checkpoint.get("epoch", 0),
            best_metric=checkpoint.get("best_metric", None),
        )

        return state
