"""Unit tests for RecurrentPositionEncoder.

Tests:
  - output shape correctness
  - action-dependent transitions produce different outputs for different actions
  - activation functions (identity, relu, tanh)
  - initial state shape
"""

import torch
import torch.nn as nn

from model.recurrent_position import RecurrentPositionEncoder


class TestRecurrentPositionEncoder:
    """Test suite for RecurrentPositionEncoder."""

    @staticmethod
    def test_output_shape() -> None:
        """forward output has shape [B, d_g]."""
        B, d_g, n_actions = 4, 128, 4
        encoder = RecurrentPositionEncoder(d_g=d_g, n_actions=n_actions)

        g_t = torch.randn(B, d_g)
        actions = torch.randint(0, n_actions, (B,))

        g_pi = encoder(g_t, actions)
        assert g_pi.shape == (B, d_g), f"Expected {(B, d_g)}, got {g_pi.shape}"

    @staticmethod
    def test_init_state_shape() -> None:
        """init_state returns [B, d_g]."""
        B, d_g, n_actions = 4, 64, 4
        encoder = RecurrentPositionEncoder(d_g=d_g, n_actions=n_actions)

        g0 = encoder.init_state(B, torch.device("cpu"))
        assert g0.shape == (B, d_g), f"Expected {(B, d_g)}, got {g0.shape}"

    @staticmethod
    def test_action_dependence() -> None:
        """Different actions should produce different position updates."""
        B, d_g, n_actions = 2, 64, 4
        encoder = RecurrentPositionEncoder(d_g=d_g, n_actions=n_actions)
        encoder.eval()

        g_t = torch.randn(B, d_g)
        g_pi_0 = encoder(g_t, torch.tensor([0, 0]))
        g_pi_1 = encoder(g_t, torch.tensor([1, 1]))

        # With small init weights they might be close, but not exactly equal
        # Reseed to ensure different outputs
        assert not torch.allclose(
            g_pi_0, g_pi_1, atol=1e-6
        ), "Different actions should produce different outputs"

    @staticmethod
    def test_identity_activation() -> None:
        """Identity activation: g_pi should be a linear function of g_t."""
        B, d_g, n_actions = 2, 16, 4
        encoder = RecurrentPositionEncoder(
            d_g=d_g, n_actions=n_actions, activation="identity"
        )
        encoder.eval()

        # With identity activation, g_pi = g_t @ W_a for the selected action.
        g_t = torch.randn(B, d_g)
        actions = torch.zeros(B, dtype=torch.long)

        g_pi = encoder(g_t, actions)

        # Manual computation: g_t @ W_a[0]
        expected = g_t @ encoder.W_a[0]
        assert torch.allclose(g_pi, expected, atol=1e-6), \
            "Identity activation should match linear transform"

    @staticmethod
    def test_relu_nonnegative() -> None:
        """ReLU activation should produce nonnegative outputs for some inputs."""
        B, d_g, n_actions = 2, 16, 4
        encoder = RecurrentPositionEncoder(
            d_g=d_g, n_actions=n_actions, activation="relu"
        )
        encoder.eval()

        # Use positive g_t and positive weights to get positive pre-activation
        # But ReLU on potentially negative values should still be >= 0
        g_t = torch.randn(B, d_g)
        actions = torch.randint(0, n_actions, (B,))

        g_pi = encoder(g_t, actions)
        assert (g_pi >= 0).all(), "ReLU output should be nonnegative"

    @staticmethod
    def test_tanh_bounded() -> None:
        """Tanh activation should produce outputs in (-1, 1)."""
        B, d_g, n_actions = 4, 16, 4
        encoder = RecurrentPositionEncoder(
            d_g=d_g, n_actions=n_actions, activation="tanh"
        )
        encoder.eval()

        g_t = torch.randn(B, d_g)
        actions = torch.randint(0, n_actions, (B,))

        g_pi = encoder(g_t, actions)
        assert (g_pi >= -1.0).all() and (g_pi <= 1.0).all(), \
            "Tanh output should be in (-1, 1)"

    @staticmethod
    def test_learnable_g0() -> None:
        """Learnable g0 should be a parameter."""
        B, d_g, n_actions = 2, 16, 4
        encoder = RecurrentPositionEncoder(
            d_g=d_g, n_actions=n_actions, learnable_g0=True
        )
        assert isinstance(encoder.g0, nn.Parameter), \
            "g0 should be a nn.Parameter when learnable_g0=True"
