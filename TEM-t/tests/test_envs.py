"""Unit tests for environment and trajectory sampling.

Tests:
  - GridWorldSpec properties
  - step_grid boundary behaviours
  - TrajectorySampler sample_batch shapes and value ranges
"""

import torch

from training.envs import (
    GridWorldSpec,
    EnvironmentInstance,
    step_grid,
    TrajectorySampler,
)
from training.batch import TrajectoryBatch


class TestGridWorldSpec:
    """Tests for GridWorldSpec."""

    @staticmethod
    def test_n_states() -> None:
        """n_states should be height * width."""
        spec = GridWorldSpec(height=5, width=7)
        assert spec.n_states == 35

    @staticmethod
    def test_invalid_boundary_raises() -> None:
        """Invalid boundary should raise ValueError."""
        try:
            GridWorldSpec(height=5, width=5, boundary="invalid")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


class TestStepGrid:
    """Tests for step_grid."""

    @staticmethod
    def test_basic_movement() -> None:
        """Moving east should increment column."""
        spec = GridWorldSpec(height=10, width=10)
        state = 0  # (0, 0)
        next_state = step_grid(state, 1, spec)  # east
        assert next_state == 1  # (0, 1)

    @staticmethod
    def test_north_movement() -> None:
        """Moving north should decrement row."""
        spec = GridWorldSpec(height=10, width=10)
        state = 10  # (1, 0)
        next_state = step_grid(state, 0, spec)  # north
        assert next_state == 0  # (0, 0)

    @staticmethod
    def test_stay_boundary() -> None:
        """With stay boundary, moving off-grid keeps position."""
        spec = GridWorldSpec(height=5, width=5, boundary="stay")
        state = 0  # (0, 0)
        next_state = step_grid(state, 0, spec)  # north (off-grid)
        assert next_state == 0  # stays at (0, 0)

    @staticmethod
    def test_wrap_boundary() -> None:
        """With wrap boundary, moving off-grid wraps around."""
        spec = GridWorldSpec(height=5, width=5, boundary="wrap")
        state = 0  # (0, 0)
        next_state = step_grid(state, 0, spec)  # north -> wraps to (4, 0)
        assert next_state == 20  # row 4, col 0


class TestTrajectorySampler:
    """Tests for TrajectorySampler."""

    @staticmethod
    def _make_sampler(**overrides) -> TrajectorySampler:
        """Create a sampler with small dimensions."""
        spec = GridWorldSpec(height=5, width=5, n_actions=4, boundary="stay")
        defaults = dict(
            spec=spec,
            n_sensory=8,
            episode_length=10,
            n_envs=20,
            seed=42,
        )
        defaults.update(overrides)
        return TrajectorySampler(**defaults)

    @staticmethod
    def test_sample_batch_shapes() -> None:
        """Sample batch should return correct shapes."""
        sampler = TestTrajectorySampler._make_sampler()
        B, T = 4, 10

        batch = sampler.sample_batch(B, torch.device("cpu"))

        assert isinstance(batch, TrajectoryBatch)
        assert batch.x_ids.shape == (B, T + 1)
        assert batch.x.shape == (B, T + 1, 8)
        assert batch.actions.shape == (B, T)
        assert batch.states.shape == (B, T + 1)
        assert batch.env_ids.shape == (B,)

    @staticmethod
    def test_sensory_ids_in_range() -> None:
        """Sensory IDs should be in [0, n_sensory)."""
        sampler = TestTrajectorySampler._make_sampler(n_sensory=8)

        batch = sampler.sample_batch(4, torch.device("cpu"))
        assert batch.x_ids.min() >= 0
        assert batch.x_ids.max() < 8

    @staticmethod
    def test_states_in_range() -> None:
        """States should be in [0, n_states)."""
        sampler = TestTrajectorySampler._make_sampler()
        spec = sampler.spec

        batch = sampler.sample_batch(4, torch.device("cpu"))
        assert batch.states.min() >= 0
        assert batch.states.max() < spec.n_states

    @staticmethod
    def test_actions_in_range() -> None:
        """Actions should be in [0, n_actions)."""
        sampler = TestTrajectorySampler._make_sampler()

        batch = sampler.sample_batch(4, torch.device("cpu"))
        assert batch.actions.min() >= 0
        assert batch.actions.max() < 4

    @staticmethod
    def test_one_hot_valid() -> None:
        """x should be valid one-hot vectors (one 1 per row)."""
        sampler = TestTrajectorySampler._make_sampler()

        batch = sampler.sample_batch(4, torch.device("cpu"))
        # Sum over class dimension should be 1 for each position
        sums = batch.x.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums))
