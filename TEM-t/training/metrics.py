"""Evaluation metrics and analysis tools for TEM-t.

Includes:
  - categorical accuracy with optional mask
  - zero-shot transition mask computation
  - rate map computation for position units and memory neurons
  - gridness score for detecting grid-like spatial tuning
  - place score (peakiness) and remapping score
"""

from typing import Optional, Tuple
import math

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
    smooth_sigma: float = 1.0,
    inner_frac: float = 0.15,
) -> float:
    """Compute the gridness score for a 2D rate map.

    Standard Sargolini/Hafting-style gridness:
    1. Optionally smooth the rate map spatially.
    2. Compute the 2D spatial autocorrelogram, centred on zero lag.
    3. Mask out the central peak and the far field, keeping an annulus.
    4. Rotate the annulus by each angle and correlate with the original.
    5. gridness = min(corr_60, corr_120) - max(corr_30, corr_90, corr_150).

    Positive gridness indicates 6-fold (hexagonal) symmetry. A score of
    0.3-0.5 is conventionally taken as the grid-cell threshold.

    Notes
    -----
    Three details are essential and easy to get wrong:

    - The FFT autocorrelation puts zero lag at index [0, 0], so the result
      MUST be fftshift-ed before the central region is taken. Without this
      the rotational correlations are computed on a misaligned map and even
      a perfect hexagonal grid scores negative.
    - The central peak must be excluded. It is rotationally symmetric at
      every angle and otherwise dominates all correlations.
    - Correlations are computed only over pixels that remain in-bounds after
      rotation, so that zero-fill from the rotation does not bias the result.

    Parameters
    ----------
    rate_map_2d : FloatTensor [H, W]
        Spatial rate map for a single unit.
    angles : tuple of int
        Rotation angles to test. Default (30, 60, 90, 120, 150).
    smooth_sigma : float
        Gaussian smoothing sigma in bins; 0 disables. Default 1.0.
    inner_frac : float
        Inner annulus radius as a fraction of the outer radius, excluding
        the central peak. Default 0.15.

    Returns
    -------
    float
        Gridness score. Returns 0.0 for a map with no variance.
    """
    rm = rate_map_2d.float()
    device = rm.device

    if rm.std() < 1e-8:
        return 0.0

    # 1. Spatial smoothing (separable Gaussian)
    if smooth_sigma and smooth_sigma > 0:
        r = max(1, int(3 * smooth_sigma))
        x = torch.arange(-r, r + 1, dtype=torch.float32, device=device)
        k = torch.exp(-x.pow(2) / (2 * smooth_sigma ** 2))
        k = k / k.sum()
        rm = F.conv2d(rm[None, None], k.view(1, 1, -1, 1), padding=(r, 0))
        rm = F.conv2d(rm, k.view(1, 1, 1, -1), padding=(0, r))[0, 0]

    rm = rm - rm.mean()
    H, W = rm.shape

    # 2. Linear autocorrelation via zero-padded FFT, then shift zero lag to centre
    padded = F.pad(rm[None, None], (W, W, H, H))
    f = torch.fft.fft2(padded)
    ac = torch.fft.ifft2(f * f.conj()).real[0, 0]
    ac = torch.fft.fftshift(ac)

    cy, cx = ac.shape[0] // 2, ac.shape[1] // 2
    radius = min(H, W) - 1
    ac = ac[cy - radius : cy + radius + 1, cx - radius : cx + radius + 1]
    ac = ac / (ac.max() + 1e-8)

    # 3. Annulus mask: drop the central peak and anything beyond the outer radius
    n = ac.shape[0]
    c = n // 2
    coords = torch.arange(n, dtype=torch.float32, device=device) - c
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    rad = torch.sqrt(yy.pow(2) + xx.pow(2))
    annulus = (rad > inner_frac * radius) & (rad <= radius)

    def _rotate(t: torch.Tensor, angle_deg: float) -> torch.Tensor:
        """Rotate a 2D tensor by angle_deg degrees about its centre."""
        theta = math.radians(angle_deg)
        rot_mat = torch.tensor(
            [[math.cos(theta), -math.sin(theta), 0.0],
             [math.sin(theta), math.cos(theta), 0.0]],
            device=device, dtype=torch.float32,
        ).unsqueeze(0)
        grid = F.affine_grid(rot_mat, [1, 1, n, n], align_corners=False)
        return F.grid_sample(
            t[None, None], grid, mode="bilinear",
            padding_mode="zeros", align_corners=False,
        )[0, 0]

    def _pearson_corr(a: torch.Tensor, b: torch.Tensor, m: torch.Tensor) -> float:
        a_c = a[m]
        b_c = b[m]
        if a_c.numel() < 2:
            return 0.0
        a_c = a_c - a_c.mean()
        b_c = b_c - b_c.mean()
        return float((a_c * b_c).sum() / (a_c.norm() * b_c.norm() + 1e-8))

    ones = torch.ones_like(ac)
    cors = {}
    for ang in angles:
        rotated = _rotate(ac, float(ang))
        # only compare pixels still in-bounds after rotation
        valid = annulus & (_rotate(ones, float(ang)) > 0.5)
        cors[ang] = _pearson_corr(ac, rotated, valid)

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
