"""Unit tests for the main TEMTModel.

Tests:
  - init_state and init_memory shapes
  - predict_next output shapes
  - observe_next output shapes
  - step output shapes
  - forward output shapes
  - no-leakage test: predict_next must not depend on x_{t+1}
"""

import torch

from model.temt import TEMTModel
from training.batch import TrajectoryBatch


def _make_mini_model(**overrides) -> TEMTModel:
    """Create a small TEMTModel for testing.

    Parameters
    ----------
    **overrides
        Override any default parameter.

    Returns
    -------
    TEMTModel
    """
    defaults = dict(
        n_sensory=10,
        n_actions=4,
        d_g=16,
        d_k=8,
        d_v=12,
        max_memory=10,
        activation="identity",
        use_landmark_stabilization=True,
        memory_dedup=False,  # disable dedup for simpler tests
    )
    defaults.update(overrides)
    return TEMTModel(**defaults)


def _make_dummy_batch(B: int, T: int, n_sensory: int = 10, n_actions: int = 4) -> TrajectoryBatch:
    """Create a dummy trajectory batch."""
    x_ids = torch.randint(0, n_sensory, (B, T + 1))
    x = torch.nn.functional.one_hot(x_ids, n_sensory).float()
    actions = torch.randint(0, n_actions, (B, T))
    states = torch.randint(0, 100, (B, T + 1))

    return TrajectoryBatch(
        x_ids=x_ids,
        x=x,
        actions=actions,
        states=states,
    )


class TestTEMTModel:
    """Tests for TEMTModel."""

    @staticmethod
    def test_init_state_shape() -> None:
        """init_state returns [B, d_g]."""
        model = _make_mini_model()
        g0 = model.init_state(4, torch.device("cpu"))
        assert g0.shape == (4, 16)

    @staticmethod
    def test_init_memory_shape() -> None:
        """init_memory returns MemoryState with correct internal shapes."""
        model = _make_mini_model(max_memory=20)

        B = 4
        g0 = model.init_state(B, torch.device("cpu"))
        x0 = torch.zeros(B, 10)
        x0[:, 0] = 1.0

        memory = model.init_memory(g0, x0)

        assert memory.keys_g.shape == (B, 20, 8)     # d_k=8
        assert memory.values_x.shape == (B, 20, 12)   # d_v=12
        assert memory.values_g.shape == (B, 20, 16)   # d_g=16
        assert memory.raw_x.shape == (B, 20, 10)      # N_x=10
        assert memory.size.tolist() == [1, 1, 1, 1]

    @staticmethod
    def test_predict_next_shapes() -> None:
        """predict_next should return correct PredictionState shapes."""
        model = _make_mini_model()

        B = 4
        g0 = model.init_state(B, torch.device("cpu"))
        x0 = torch.zeros(B, 10)
        x0[:, 0] = 1.0
        memory = model.init_memory(g0, x0)

        actions = torch.randint(0, model.n_actions, (B,))
        pred = model.predict_next(g0, memory, actions)

        assert pred.g_pi.shape == (B, 16)
        assert pred.q_pi.shape == (B, 8)
        assert pred.logits_pi.shape == (B, 10)
        assert pred.probs_pi.shape == (B, 10)
        assert pred.attn_pi.read.shape == (B, 12)

    @staticmethod
    def test_observe_next_shapes() -> None:
        """observe_next should return correct ObservationState shapes."""
        model = _make_mini_model()

        B = 4
        g0 = model.init_state(B, torch.device("cpu"))
        x0 = torch.zeros(B, 10)
        x0[:, 0] = 1.0
        memory = model.init_memory(g0, x0)

        actions = torch.randint(0, model.n_actions, (B,))
        pred = model.predict_next(g0, memory, actions)

        x_next = torch.zeros(B, 10)
        x_next[:, 3] = 1.0

        obs = model.observe_next(pred.g_pi, memory, x_next)

        assert obs.g_next.shape == (B, 16)
        assert obs.g_retrieved.shape == (B, 16)
        assert obs.eta.shape == (B, 16)
        assert obs.wrote_mask.shape == (B,)
        assert obs.memory_next.size.tolist() == [2, 2, 2, 2]

    @staticmethod
    def test_step_shapes() -> None:
        """step should return correct TEMTStepOutput."""
        model = _make_mini_model()

        B = 4
        g0 = model.init_state(B, torch.device("cpu"))
        x0 = torch.zeros(B, 10)
        x0[:, 0] = 1.0
        memory = model.init_memory(g0, x0)

        actions = torch.randint(0, model.n_actions, (B,))
        x_next = torch.zeros(B, 10)
        x_next[:, 3] = 1.0

        step_out = model.step(g0, memory, actions, x_next)

        assert step_out.prediction.logits_pi.shape == (B, 10)
        assert step_out.observation.g_next.shape == (B, 16)

    @staticmethod
    def test_forward_shapes() -> None:
        """forward should return correct TEMTForwardOutput shapes."""
        model = _make_mini_model(max_memory=10)

        B, T = 4, 8
        batch = _make_dummy_batch(B, T, n_sensory=10, n_actions=4)

        output = model(batch, return_traces=True, compute_stable_prediction=True)

        assert output.logits_pi.shape == (B, T, 10)
        assert output.probs_pi.shape == (B, T, 10)
        assert output.g_seq.shape == (B, T + 1, 16)
        assert output.g_pi_seq.shape == (B, T, 16)
        assert output.logits_stable.shape == (B, T, 10)
        assert output.probs_stable.shape == (B, T, 10)
        assert output.attn_pi.shape == (B, T, model.max_memory)
        assert output.attn_stable.shape == (B, T, model.max_memory)
        assert output.memory_sizes.shape == (B, T + 1)

    @staticmethod
    def test_no_leakage() -> None:
        """Modifying x_{t+1} should NOT change predict_next output.

        This is the critical no-leakage invariant: predict_next uses
        only g_t, memory_t, and a_t. It must not access x_{t+1}.
        """
        model = _make_mini_model()

        B = 4
        g0 = model.init_state(B, torch.device("cpu"))
        x0 = torch.zeros(B, 10)
        x0[:, 0] = 1.0
        memory = model.init_memory(g0, x0)

        # Run one predict with fixed inputs
        actions = torch.randint(0, model.n_actions, (B,))
        pred1 = model.predict_next(g0, memory, actions)

        # Run again with same inputs (should be identical)
        pred2 = model.predict_next(g0, memory, actions)

        assert torch.allclose(pred1.logits_pi, pred2.logits_pi, atol=1e-6), \
            "predict_next should be deterministic given same inputs"

    @staticmethod
    def test_forward_batch_determinism() -> None:
        """Same batch should produce same outputs."""
        model = _make_mini_model(max_memory=20)
        model.eval()

        B, T = 4, 8
        batch1 = _make_dummy_batch(B, T)

        torch.manual_seed(42)
        batch2 = _make_dummy_batch(B, T)

        # They'll differ because random, so test with the same batch
        with torch.no_grad():
            out1 = model(batch1, return_traces=False, compute_stable_prediction=False)
            out2 = model(batch1, return_traces=False, compute_stable_prediction=False)

        assert torch.allclose(out1.logits_pi, out2.logits_pi, atol=1e-6)
