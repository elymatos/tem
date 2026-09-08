"""End-to-end integration tests for TEM-t.

Tests the full pipeline: environment sampling -> model forward ->
loss computation -> gradient update.
"""

import torch

from model.temt import TEMTModel
from training.batch import TrajectoryBatch
from training.envs import GridWorldSpec, TrajectorySampler
from training.losses import TEMTLoss
from training.evaluator import TEMTEvaluator
from training.metrics import categorical_accuracy


class TestIntegration:
    """Integration tests for the full TEM-t pipeline."""

    @staticmethod
    def test_full_pipeline() -> None:
        """End-to-end: sample -> forward -> loss -> backward -> step."""
        # Setup
        spec = GridWorldSpec(height=5, width=5, n_actions=5, boundary="stay")
        sampler = TrajectorySampler(
            spec=spec,
            n_sensory=10,
            episode_length=20,
            n_envs=10,
            seed=0,
        )

        model = TEMTModel(
            n_sensory=10,
            n_actions=5,
            d_g=16,
            d_k=8,
            d_v=12,
            max_memory=25,
            memory_dedup=False,
        )

        loss_fn = TEMTLoss(
            lambda_pi=1.0,
            lambda_stable=1.0,
            lambda_g=0.1,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Sample batch
        batch = sampler.sample_batch(4, torch.device("cpu"))

        # Forward
        output = model(batch, return_traces=True, compute_stable_prediction=True)

        # Loss
        loss_out = loss_fn(output, batch, model)

        # Backward
        loss_out.total.backward()

        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No grad for {name}"

        # Optimizer step
        optimizer.step()

        # Check loss is finite
        assert torch.isfinite(loss_out.total), f"Loss is not finite: {loss_out.total}"

    @staticmethod
    def test_evaluator_integration() -> None:
        """Evaluator should work with a trained (random) model."""
        spec = GridWorldSpec(height=5, width=5)
        sampler = TrajectorySampler(
            spec=spec, n_sensory=10, episode_length=10, n_envs=5, seed=0,
        )

        model = TEMTModel(
            n_sensory=10, n_actions=5,
            d_g=16, d_k=8, d_v=12,
            max_memory=15, memory_dedup=False,
        )
        model.eval()

        evaluator = TEMTEvaluator(model, sampler, torch.device("cpu"))

        result = evaluator.evaluate_prediction(batch_size=4, n_batches=5)

        assert "acc_pi" in result.metrics
        assert "loss_pi" in result.metrics
        assert 0.0 <= result.metrics["acc_pi"] <= 1.0

    @staticmethod
    def test_training_loop_converges() -> None:
        """A few training steps should reduce loss."""
        spec = GridWorldSpec(height=5, width=5, n_actions=5, boundary="stay")
        sampler = TrajectorySampler(
            spec=spec, n_sensory=10, episode_length=10, n_envs=20, seed=42,
        )

        model = TEMTModel(
            n_sensory=10, n_actions=5,
            d_g=16, d_k=8, d_v=12,
            max_memory=15, memory_dedup=False,
        )

        loss_fn = TEMTLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        device = torch.device("cpu")

        losses = []
        for _ in range(20):
            model.train()
            optimizer.zero_grad()

            batch = sampler.sample_batch(8, device)
            output = model(batch, return_traces=True, compute_stable_prediction=True)
            loss_out = loss_fn(output, batch, model)
            loss_out.total.backward()
            optimizer.step()

            losses.append(loss_out.total.item())

        # Loss should generally decrease (first > last on average)
        # Allow some noise by comparing first 5 avg vs last 5 avg
        first_avg = sum(losses[:5]) / 5
        last_avg = sum(losses[-5:]) / 5

        assert last_avg < first_avg, (
            f"Loss should decrease: first_avg={first_avg:.4f}, last_avg={last_avg:.4f}"
        )
