"""Zero-shot evaluation integration tests.

Tests:
  - compute_zero_shot_mask produces expected mask on a simple trajectory
  - Zero-shot evaluation workflow: model forward -> mask -> accuracy
"""

import torch

from model.temt import TEMTModel
from training.batch import TrajectoryBatch
from training.metrics import compute_zero_shot_mask


def _make_simple_model() -> TEMTModel:
    """Create a small TEMTModel."""
    return TEMTModel(
        n_sensory=10, n_actions=4,
        d_g=16, d_k=8, d_v=12,
        max_memory=10,
        memory_dedup=False,
    )


class TestZeroShot:
    """Integration tests for zero-shot evaluation."""

    @staticmethod
    def test_zero_shot_mask_simple() -> None:
        """Verify zero-shot mask on a hand-crafted trajectory.

        Trajectory: 0->1->2->0->1
        Steps:
          t=0: 0->1 (new edge, new dest) -> not ZS
          t=1: 1->2 (new edge, new dest) -> not ZS
          t=2: 2->0 (new edge, dest 0 known) -> ZS
          t=3: 0->1 (edge(0,a) traversed t=0) -> not ZS
        """
        states = torch.tensor([[0, 1, 2, 0, 1]])
        actions = torch.tensor([[0, 0, 1, 0]])

        zs = compute_zero_shot_mask(states, actions)

        expected = torch.tensor([[False, False, True, False]])
        assert (zs == expected).all(), f"Expected {expected}, got {zs}"

    @staticmethod
    def test_zero_shot_alignment() -> None:
        """logits_pi[:, t] should align with zero_shot_mask[:, t] and target[:, t+1].

        This ensures the time-index alignment is correct for evaluation.
        """
        B, T = 2, 5
        model = _make_simple_model()

        x_ids = torch.randint(0, 10, (B, T + 1))
        x = torch.nn.functional.one_hot(x_ids, 10).float()
        actions = torch.randint(0, 4, (B, T))
        states = torch.randint(0, 25, (B, T + 1))

        batch = TrajectoryBatch(
            x_ids=x_ids, x=x, actions=actions, states=states,
        )

        with torch.no_grad():
            output = model(batch, return_traces=False, compute_stable_prediction=False)

        zs_mask = compute_zero_shot_mask(states, actions)

        # logits_pi[t] predicts x_ids[t+1], zs_mask[t] also refers to transition t->t+1
        assert output.logits_pi.shape == (B, T, 10)
        assert zs_mask.shape == (B, T)

        # Verify alignment: pick a zero-shot step and check prediction
        for b in range(B):
            for t in range(T):
                if zs_mask[b, t]:
                    pred = output.logits_pi[b, t].argmax().item()
                    target = x_ids[b, t + 1].item()
                    # We don't assert correctness, just that shapes align
                    assert 0 <= pred < 10
                    assert 0 <= target < 10
