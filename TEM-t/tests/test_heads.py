"""Unit tests for prediction heads (SensoryPredictionHead, LandmarkStabilizer).

Tests:
  - SensoryPredictionHead output shapes
  - LandmarkStabilizer forward shapes
  - LandmarkStabilizer: disabled mode returns g_pi
"""

import torch

from model.heads import SensoryPredictionHead, LandmarkStabilizer
from model.types import MemoryState


class TestSensoryPredictionHead:
    """Tests for SensoryPredictionHead."""

    @staticmethod
    def test_output_shape() -> None:
        """Output should be [B, N_x]."""
        B, d_v, N_x = 4, 128, 45
        head = SensoryPredictionHead(d_v=d_v, n_sensory=N_x, hidden_dim=64, n_layers=1)

        read = torch.randn(B, d_v)
        logits = head(read)
        assert logits.shape == (B, N_x)

    @staticmethod
    def test_zero_layers() -> None:
        """0 hidden layers should also work (linear projection)."""
        B, d_v, N_x = 4, 32, 10
        head = SensoryPredictionHead(d_v=d_v, n_sensory=N_x, n_layers=0)

        read = torch.randn(B, d_v)
        logits = head(read)
        assert logits.shape == (B, N_x)

    @staticmethod
    def test_multiple_layers() -> None:
        """Multiple hidden layers should work."""
        B, d_v, N_x = 4, 64, 20
        head = SensoryPredictionHead(d_v=d_v, n_sensory=N_x, hidden_dim=32, n_layers=3)

        read = torch.randn(B, d_v)
        logits = head(read)
        assert logits.shape == (B, N_x)


class TestLandmarkStabilizer:
    """Tests for LandmarkStabilizer."""

    @staticmethod
    def test_forward_shapes() -> None:
        """g_next and eta should have shape [B, d_g]."""
        B, d_g, d_v, d_k = 4, 64, 32, 32
        stabilizer = LandmarkStabilizer(
            d_g=d_g, d_v=d_v, d_k=d_k, use_landmark=True,
        )

        g_pi = torch.randn(B, d_g)
        g_retrieved = torch.randn(B, d_g)

        g_next, eta = stabilizer(g_pi, g_retrieved)

        assert g_next.shape == (B, d_g)
        assert eta.shape == (B, d_g)

    @staticmethod
    def test_disabled_returns_pi() -> None:
        """When use_landmark=False, g_next should equal g_pi."""
        B, d_g, d_v, d_k = 4, 64, 32, 32
        stabilizer = LandmarkStabilizer(
            d_g=d_g, d_v=d_v, d_k=d_k, use_landmark=False,
        )

        g_pi = torch.randn(B, d_g)
        g_retrieved = torch.randn(B, d_g)

        g_next, eta = stabilizer(g_pi, g_retrieved)

        assert torch.allclose(g_next, g_pi, atol=1e-6)
        assert (eta == 0).all()

    @staticmethod
    def test_eta_in_range() -> None:
        """eta values should be in (0, 1) due to sigmoid."""
        B, d_g, d_v, d_k = 4, 64, 32, 32
        stabilizer = LandmarkStabilizer(
            d_g=d_g, d_v=d_v, d_k=d_k, use_landmark=True,
        )

        g_pi = torch.randn(B, d_g)
        g_retrieved = torch.randn(B, d_g)

        _, eta = stabilizer(g_pi, g_retrieved)

        assert (eta >= 0).all() and (eta <= 1).all(), \
            f"eta should be in [0, 1], got [{eta.min():.4f}, {eta.max():.4f}]"

    @staticmethod
    def test_retrieve_position_no_landmark() -> None:
        """With use_landmark=False, retrieve_position returns g_pi."""
        B, M, d_g, d_v, d_k, N_x = 2, 5, 64, 32, 32, 10
        stabilizer = LandmarkStabilizer(
            d_g=d_g, d_v=d_v, d_k=d_k, use_landmark=False,
        )

        g_pi = torch.randn(B, d_g)
        x_next = torch.zeros(B, N_x)
        x_next[0, 3] = 1.0
        x_next[1, 7] = 1.0

        # Create a minimal memory
        memory = MemoryState(
            keys_g=torch.randn(B, M, d_k),
            values_x=torch.randn(B, M, d_v),
            values_g=torch.randn(B, M, d_g),
            raw_x=torch.zeros(B, M, N_x),
            valid_mask=torch.ones(B, M, dtype=torch.bool),
            size=torch.tensor([M, M], dtype=torch.long),
        )

        sensory_value_x = torch.randn(B, d_v)

        g_retrieved, _ = stabilizer.retrieve_position(
            x_next, g_pi, memory, sensory_value_x
        )

        assert torch.allclose(g_retrieved, g_pi, atol=1e-6)
