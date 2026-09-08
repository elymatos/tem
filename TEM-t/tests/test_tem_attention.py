"""Unit tests for TEM attention modules.

Tests:
  - FixedLayerNorm: zero mean, unit variance
  - PositionProjector and SensoryProjector: shape correctness
  - TEMAttention: output shapes, mask enforcement, adaptive beta
"""

import torch

from model.tem_attention import (
    FixedLayerNorm,
    PositionProjector,
    SensoryProjector,
    TEMAttention,
)
from model.types import AttentionOutput


class TestFixedLayerNorm:
    """Tests for FixedLayerNorm."""

    @staticmethod
    def test_output_shape() -> None:
        """Output shape matches input shape."""
        ln = FixedLayerNorm()
        x = torch.randn(4, 8, 64)
        y = ln(x)
        assert y.shape == x.shape

    @staticmethod
    def test_zero_mean() -> None:
        """Normalised output should have near-zero mean per sample."""
        ln = FixedLayerNorm()
        x = torch.randn(16, 32)
        y = ln(x)
        mu = y.mean(dim=-1)
        assert (mu.abs() < 1e-5).all(), f"Mean should be ~0, got {mu}"

    @staticmethod
    def test_unit_variance() -> None:
        """Normalised output should have variance ~1 per sample."""
        ln = FixedLayerNorm()
        x = torch.randn(16, 32)
        y = ln(x)
        var = y.var(dim=-1, unbiased=False)
        assert (var - 1.0).abs().max() < 1e-4

    @staticmethod
    def test_handles_batch_dimensions() -> None:
        """Should work with arbitrary leading dimensions."""
        ln = FixedLayerNorm()
        x = torch.randn(2, 5, 3, 16)
        y = ln(x)
        assert y.shape == x.shape
        mu = y.mean(dim=-1)
        assert (mu.abs() < 1e-5).all()


class TestPositionProjector:
    """Tests for PositionProjector."""

    @staticmethod
    def test_single_output_shape() -> None:
        """Project single vector [B, d_g] -> [B, d_k]."""
        B, d_g, d_k = 4, 128, 64
        proj = PositionProjector(d_g, d_k)
        g = torch.randn(B, d_g)
        out = proj(g)
        assert out.shape == (B, d_k)

    @staticmethod
    def test_batch_memory_output_shape() -> None:
        """Project memory batch [B, M, d_g] -> [B, M, d_k]."""
        B, M, d_g, d_k = 4, 10, 128, 64
        proj = PositionProjector(d_g, d_k)
        g = torch.randn(B, M, d_g)
        out = proj(g)
        assert out.shape == (B, M, d_k)


class TestSensoryProjector:
    """Tests for SensoryProjector."""

    @staticmethod
    def test_onehot_mode() -> None:
        """Project one-hot sensory input."""
        N_x, d_v = 10, 32
        proj = SensoryProjector(N_x, d_v, input_mode="onehot")

        x = torch.zeros(3, N_x)
        x[0, 2] = 1.0
        x[1, 5] = 1.0
        x[2, 9] = 1.0

        out = proj(x)
        assert out.shape == (3, d_v)

    @staticmethod
    def test_id_mode() -> None:
        """Project sensory ID input."""
        N_x, d_v = 10, 32
        proj = SensoryProjector(N_x, d_v, input_mode="id")

        x_ids = torch.tensor([2, 5, 9])
        out = proj(x_ids)
        assert out.shape == (3, d_v)

    @staticmethod
    def test_onehot_equals_id() -> None:
        """One-hot mode and ID mode should give same results for equivalent inputs."""
        N_x, d_v = 10, 32

        # Need same weights
        proj_onehot = SensoryProjector(N_x, d_v, input_mode="onehot")
        proj_id = SensoryProjector(N_x, d_v, input_mode="id")

        # Copy weights
        proj_id.W_x.load_state_dict(proj_onehot.W_x.state_dict())

        x_ids = torch.tensor([2, 5, 9])
        x_onehot = torch.nn.functional.one_hot(x_ids, N_x).float()

        out_onehot = proj_onehot(x_onehot)
        out_id = proj_id(x_ids)

        assert torch.allclose(out_onehot, out_id, atol=1e-6)


class TestTEMAttention:
    """Tests for TEMAttention."""

    @staticmethod
    def test_output_shapes() -> None:
        """Attention returns correct shapes."""
        B, M, d_k, d_v = 2, 5, 64, 128
        attn = TEMAttention(d_k=d_k)

        query = torch.randn(B, d_k)
        keys = torch.randn(B, M, d_k)
        values = torch.randn(B, M, d_v)
        valid_mask = torch.ones(B, M, dtype=torch.bool)
        mem_size = torch.tensor([3, 5], dtype=torch.long)  # different sizes

        out = attn(query, keys, values, valid_mask, mem_size)

        assert isinstance(out, AttentionOutput)
        assert out.read.shape == (B, d_v)
        assert out.weights.shape == (B, M)
        assert out.scores.shape == (B, M)

    @staticmethod
    def test_weights_sum_to_one() -> None:
        """Attention weights should sum to ~1 over valid slots."""
        B, M, d_k, d_v = 2, 10, 64, 32
        attn = TEMAttention(d_k=d_k)

        query = torch.randn(B, d_k)
        keys = torch.randn(B, M, d_k)
        values = torch.randn(B, M, d_v)

        # Only first 3 slots valid for batch 0, first 5 for batch 1
        valid_mask = torch.zeros(B, M, dtype=torch.bool)
        valid_mask[0, :3] = True
        valid_mask[1, :5] = True
        mem_size = torch.tensor([3, 5], dtype=torch.long)

        out = attn(query, keys, values, valid_mask, mem_size)

        # Weights on valid slots should sum to ~1
        sum_b0 = out.weights[0, valid_mask[0]].sum()
        sum_b1 = out.weights[1, valid_mask[1]].sum()
        assert torch.allclose(sum_b0, torch.tensor(1.0), atol=1e-5)
        assert torch.allclose(sum_b1, torch.tensor(1.0), atol=1e-5)

    @staticmethod
    def test_invalid_slots_zero_weight() -> None:
        """Attention weights should be zero on invalid slots."""
        B, M, d_k, d_v = 2, 5, 64, 32
        attn = TEMAttention(d_k=d_k)

        query = torch.randn(B, d_k)
        keys = torch.randn(B, M, d_k)
        values = torch.randn(B, M, d_v)

        valid_mask = torch.zeros(B, M, dtype=torch.bool)
        valid_mask[0, :2] = True
        valid_mask[1, :3] = True
        mem_size = torch.tensor([2, 3], dtype=torch.long)

        out = attn(query, keys, values, valid_mask, mem_size)

        assert (out.weights[0, ~valid_mask[0]] == 0).all()
        assert (out.weights[1, ~valid_mask[1]] == 0).all()

    @staticmethod
    def test_adaptive_beta_grows() -> None:
        """Beta should increase with memory size."""
        d_k = 64
        attn = TEMAttention(d_k=d_k, beta0=1.0, adaptive_beta=True)

        beta_small = attn.compute_beta(torch.tensor([1]))
        beta_large = attn.compute_beta(torch.tensor([10]))

        assert beta_large.item() > beta_small.item(), \
            "Adaptive beta should increase with memory size"
