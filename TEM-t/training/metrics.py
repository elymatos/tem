"""Evaluation metrics and analysis tools for TEM-t.

Includes:
  - categorical accuracy with optional mask
  - zero-shot transition mask computation
  - rate map computation for position units and memory neurons
  - gridness score for detecting grid-like spatial tuning
  - place score (peakiness) and remapping score
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# categorical_accuracy
# ---------------------------------------------------------------------------
def categorical_accuracy(
    logits: torch.FloatTensor,
    targets: torch.LongTensor,
    mask: Optional[torch.BoolTensor] = None,
) -> torch.Tensor:
    """Compute categorical prediction accuracy.

    Parameters
    ----------
    logits : FloatTensor [B, T, N_x]
        Prediction logits per time step.
    targets : LongTensor [B, T]
        Ground-truth class indices.
    mask : Optional BoolTensor [B, T]
        If provided, only count masked-in entries.

    Returns
    -------
    scalar Tensor
        Accuracy as a fraction in [0, 1].
    """
    preds = logits.argmax(dim=-1)  # [B, T]
    correct = (preds == targets).float()

    if mask is not None:
        correct = correct * mask.float()
        denom = mask.float().sum() + 1e-8
    else:
        denom = correct.numel()

    return correct.sum() / denom


# ---------------------------------------------------------------------------
# zero_shot_mask
# ---------------------------------------------------------------------------
def compute_zero_shot_mask(
    states: torch.LongTensor,
    actions: torch.LongTensor,
) -> torch.BoolTensor:
    """Compute zero-shot transition mask.

    A transition at time t is zero-shot iff:
      1. The edge (s_t, a_t) was NOT traversed before in this trajectory.
      2. The destination state s_{t+1} WAS visited before.

    This tests whether the model can infer the correct destination
    via learned spatial structure rather than memorising transitions.

    Parameters
    ----------
    states : LongTensor [B, T+1]
        Graph node indices along the trajectory.
    actions : LongTensor [B, T]
        Action indices at each step.

    Returns
    -------
    BoolTensor [B, T]
        True for zero-shot transitions.
    """
    B, T_plus_1 = states.shape
    T = actions.shape[1]
    device = states.device

    zero_shot = torch.zeros(B, T, device=device, dtype=torch.bool)

    for b in range(B):
        visited_nodes = set()
        visited_edges = set()

        visited_nodes.add(int(states[b, 0].item()))

        for t in range(T):
            s_t = int(states[b, t].item())
            a_t = int(actions[b, t].item())
            s_tp1 = int(states[b, t + 1].item())

            edge = (s_t, a_t)

            # Check zero-shot condition
            novel_edge = edge not in visited_edges
            known_dest = s_tp1 in visited_nodes

            if novel_edge and known_dest:
                zero_shot[b, t] = True

            # Update visit sets
            visited_nodes.add(s_tp1)
            visited_edges.add(edge)

    return zero_shot


# ---------------------------------------------------------------------------
# rate_map
# ---------------------------------------------------------------------------
def compute_rate_maps(
    activations: torch.FloatTensor,
    states: torch.LongTensor,
    n_states: int,
    mask: Optional[torch.BoolTensor] = None,
) -> torch.FloatTensor:
    """Compute spatial rate maps for hidden units.

    For each unit i and graph node v, the rate map is the average
    activation of unit i when the agent is at node v.

    Formula:
        R_i(v) = sum_t 1[states[t]==v] * activations[t,i]
               / (sum_t 1[states[t]==v] + eps)

    Parameters
    ----------
    activations : FloatTensor [B, T, D]
        Unit activations per time step (e.g. g_seq or attn weights).
    states : LongTensor [B, T]
        Ground-truth graph node indices.
    n_states : int
        Total number of distinct graph nodes.
    mask : Optional BoolTensor [B, T]
        Optional validity mask.

    Returns
    -------
    FloatTensor [D, N_s]
        Rate map for each of the D units across all N_s states.
    """
    B, T_total, D = activations.shape
    device = activations.device

    rate_maps = torch.zeros(D, n_states, device=device)
    visit_counts = torch.zeros(n_states, device=device)

    for b in range(B):
        for t in range(T_total):
            if mask is not None and not mask[b, t]:
                continue

            s = int(states[b, t].item())
            if s < 0 or s >= n_states:
                continue

            visit_counts[s] += 1
            rate_maps[:, s] += activations[b, t]

    # Normalise by visit counts
    rate_maps = rate_maps / (visit_counts.unsqueeze(0) + 1e-8)

    return rate_maps


# ---------------------------------------------------------------------------
# memory_rate_map
# ---------------------------------------------------------------------------
def compute_memory_rate_maps(
    attn_weights: torch.FloatTensor,
    states: torch.LongTensor,
    n_states: int,
    mask: Optional[torch.BoolTensor] = None,
) -> torch.FloatTensor:
    """Compute spatial rate maps for memory neurons (attention slots).

    Each memory neuron j (attention slot j) has its activation (= attention
    weight) averaged across visits to each graph node.

    Formula:
        R_j^{mem}(v) = sum_t 1[states[t]==v] * attn[t,j]
                     / (sum_t 1[states[t]==v] + eps)

    Parameters
    ----------
    attn_weights : FloatTensor [B, T, M]
        Attention weights over memory slots per time step.
    states : LongTensor [B, T]
        Ground-truth graph node indices.
    n_states : int
        Total number of distinct graph nodes.
    mask : Optional BoolTensor [B, T]
        Optional validity mask.

    Returns
    -------
    FloatTensor [M, N_s]
        Rate map for each of the M memory slots across all N_s states.
    """
    return compute_rate_maps(attn_weights, states, n_states, mask)


# ---------------------------------------------------------------------------
# gridness_score
# ---------------------------------------------------------------------------
def gridness_score(
    rate_map_2d: torch.FloatTensor,
    angles: Tuple[int, ...] = (30, 60, 90, 120, 150),
) -> float:
    """Compute the gridness score for a 2D rate map.

    Gridness quantifies hexagonal periodic structure:
    1. Compute the 2D spatial autocorrelation of the rate map.
    2. Rotate the autocorrelogram by each angle.
    3. Compute Pearson correlation between original and rotated maps.
    4. gridness = min(corr_60, corr_120) - max(corr_30, corr_90, corr_150).

    Positive gridness indicates a 6-fold symmetric (grid-like) pattern.

    Parameters
    ----------
    rate_map_2d : FloatTensor [H, W]
        Spatial rate map for a single unit.
    angles : tuple of int
        Rotation angles to test. Default (30, 60, 90, 120, 150).

    Returns
    -------
    float
        Gridness score.
    """
    import torch.fft

    H, W = rate_map_2d.shape
    device = rate_map_2d.device

    # Detrend: subtract the mean
    rm = rate_map_2d - rate_map_2d.mean()

    # 2D autocorrelation via FFT
    # Pad to avoid circular wrap-around effects
    pad_h, pad_w = H // 2, W // 2

    # Use rfft2 for real input
    rm_pad = F.pad(rm.unsqueeze(0).unsqueeze(0), (pad_w, pad_w, pad_h, pad_h))
    # Shift to centre
    # FFT-based autocorrelation: ifft2(|fft2(x)|^2)
    f = torch.fft.rfft2(rm_pad)
    power = f.real.pow(2) + f.imag.pow(2)
    autocorr_full = torch.fft.irfft2(power).squeeze(0).squeeze(0)

    # Take the central (H, W) region
    ac = autocorr_full[
        pad_h : pad_h + H,
        pad_w : pad_w + W,
    ]

    # Normalise
    ac = ac / (ac.max() + 1e-8)

    # Helper: rotate a 2D tensor by an angle using affine grid
    def _rotate(t: torch.Tensor, angle_deg: float) -> torch.Tensor:
        """Rotate a 2D tensor by angle_deg degrees about its centre."""
        theta = angle_deg * (3.1415926535 / 180.0)
        cos_a, sin_a = torch.cos(torch.tensor(theta)), torch.sin(torch.tensor(theta))
        rot_mat = torch.tensor(
            [[cos_a, -sin_a, 0], [sin_a, cos_a, 0]],
            device=device, dtype=torch.float32,
        ).unsqueeze(0)

        grid = F.affine_grid(
            rot_mat,
            [1, 1, H, W],
            align_corners=False,
        )
        rotated = F.grid_sample(
            t.unsqueeze(0).unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return rotated.squeeze(0).squeeze(0)

    def _pearson_corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute Pearson correlation between flattened tensors."""
        a_f = a.flatten()
        b_f = b.flatten()
        a_c = a_f - a_f.mean()
        b_c = b_f - b_f.mean()
        return (a_c * b_c).sum() / (a_c.norm() * b_c.norm() + 1e-8)

    # Compute correlations at each angle
    cors = {}
    for ang in angles:
        rotated = _rotate(ac, float(ang))
        cors[ang] = _pearson_corr(ac, rotated).item()

    # Gridness formula
    gridness = min(cors[60], cors[120]) - max(cors[30], cors[90], cors[150])

    return float(gridness)


# ---------------------------------------------------------------------------
# place_score (peakiness)
# ---------------------------------------------------------------------------
def place_score(
    rate_map_2d: torch.FloatTensor,
) -> float:
    """Compute the peakiness / place selectivity score.

    Higher values indicate that activation is concentrated in fewer
    spatial locations (place-cell-like tuning).

    Formula:
        peakiness = max(rate_map) / (mean(rate_map) + eps)

    Parameters
    ----------
    rate_map_2d : FloatTensor [H, W]
        Spatial rate map for a single unit.

    Returns
    -------
    float
        Peakiness score.
    """
    r = rate_map_2d
    return float(r.max().item() / (r.mean().item() + 1e-8))


# ---------------------------------------------------------------------------
# remapping_score
# ---------------------------------------------------------------------------
def remapping_score(
    rate_map_a: torch.FloatTensor,
    rate_map_b: torch.FloatTensor,
) -> float:
    """Compute the cross-environment correlation of two rate maps.

    Low values (near 0) indicate remapping: the unit fires in
    different locations across environments.

    Formula:
        rho = PearsonCorr(flatten(R_a), flatten(R_b))

    Parameters
    ----------
    rate_map_a : FloatTensor [H, W]
        Rate map in environment A.
    rate_map_b : FloatTensor [H, W]
        Rate map in environment B.

    Returns
    -------
    float
        Pearson correlation coefficient in [-1, 1].
    """
    a = rate_map_a.flatten()
    b = rate_map_b.flatten()
    a_c = a - a.mean()
    b_c = b - b.mean()
    return float(
        (a_c * b_c).sum().item() / (a_c.norm().item() * b_c.norm().item() + 1e-8)
    )
