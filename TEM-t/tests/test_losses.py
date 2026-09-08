"""Unit tests for TEMTLoss.

Tests:
  - LossOutput structure
  - loss shapes (scalar)
  - loss composition when some terms disabled
"""

import torch

from model.temt import TEMTModel
from training.batch import TrajectoryBatch
from training.losses import TEMTLoss, LossOutput


def _make_model_and_batch():
    """Create a small model and batch for loss testing."""
    B, T = 2, 5
    model = TEMTModel(
        n_sensory=10, n_actions=4,
        d_g=16, d_k=8, d_v=12,
        max_memory=10,
        use_landmark_stabilization=True,
        memory_dedup=False,
    )

    x_ids = torch.randint(0, 10, (B, T + 1))
    x = torch.nn.functional.one_hot(x_ids, 10).float()
    actions = torch.randint(0, model.n_actions, (B, T))
    states = torch.randint(0, 100, (B, T + 1))

    batch = TrajectoryBatch(
        x_ids=x_ids, x=x, actions=actions, states=states,
    )

    return model, batch


class TestTEMTLoss:
    """Tests for TEMTLoss."""

    @staticmethod
    def test_output_is_loss_output() -> None:
        """Output should be a LossOutput dataclass."""
        model, batch = _make_model_and_batch()

        output = model(batch, return_traces=True, compute_stable_prediction=True)
        loss_fn = TEMTLoss()
        loss_out = loss_fn(output, batch, model)

        assert isinstance(loss_out, LossOutput)
        assert isinstance(loss_out.total, torch.Tensor)
        assert loss_out.total.ndim == 0  # scalar

    @staticmethod
    def test_terms_dict_keys() -> None:
        """LossOutput.terms should contain expected keys."""
        model, batch = _make_model_and_batch()

        output = model(batch, return_traces=True, compute_stable_prediction=True)
        loss_fn = TEMTLoss()
        loss_out = loss_fn(output, batch, model)

        expected_keys = {"loss_pi", "loss_stable", "loss_g", "loss_weight", "loss_g_l2"}
        assert set(loss_out.terms.keys()) == expected_keys

    @staticmethod
    def test_metrics_contains_accuracy() -> None:
        """LossOutput.metrics should contain acc_pi."""
        model, batch = _make_model_and_batch()

        output = model(batch, return_traces=True, compute_stable_prediction=True)
        loss_fn = TEMTLoss()
        loss_out = loss_fn(output, batch, model)

        assert "acc_pi" in loss_out.metrics
        assert 0.0 <= loss_out.metrics["acc_pi"].item() <= 1.0

    @staticmethod
    def test_lambda_zero_disables_loss() -> None:
        """Setting all lambdas to zero should give total loss ~0."""
        model, batch = _make_model_and_batch()

        output = model(batch, return_traces=True, compute_stable_prediction=True)
        loss_fn = TEMTLoss(
            lambda_pi=0.0,
            lambda_stable=0.0,
            lambda_g=0.0,
            lambda_weight=0.0,
            lambda_g_l2=0.0,
        )
        loss_out = loss_fn(output, batch, None)

        assert torch.allclose(loss_out.total, torch.tensor(0.0), atol=1e-6)

    @staticmethod
    def test_requires_grad() -> None:
        """Total loss should require grad."""
        model, batch = _make_model_and_batch()

        output = model(batch, return_traces=True, compute_stable_prediction=True)
        loss_fn = TEMTLoss()
        loss_out = loss_fn(output, batch, model)

        assert loss_out.total.requires_grad
