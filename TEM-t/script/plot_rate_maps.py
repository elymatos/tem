#!/usr/bin/env python3
"""Plot rate maps extracted from a trained TEM-t model.

Generates:
  - g_unit_rate_maps.png  : rate maps for position-encoding units
  - memory_rate_maps.png   : rate maps for memory neurons (if available)
  - gridness_histogram.png : distribution of gridness scores

Usage:
    python script/plot_rate_maps.py \
        --rate_map_dir experiments/exp01/run001/rate_maps \
        --output_dir experiments/exp01/run001/figures
"""

import argparse
import os
import json
import math

import torch
import numpy as np

# Matplotlib is optional; graceful degradation if missing.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    plt = None


def plot_unit_grid(
    rate_maps: torch.FloatTensor,
    output_path: str,
    title: str = "Rate Maps",
    max_plots: int = 64,
) -> None:
    """Plot a grid of rate maps.

    Parameters
    ----------
    rate_maps : FloatTensor [D, H, W] or [D, N_s]
        2D rate maps for D units.
    output_path : str
        Output image file path.
    title : str
        Figure title.
    max_plots : int
        Maximum number of units to show.
    """
    if not HAS_MPL:
        print("matplotlib not installed; skipping plot.")
        return

    D = rate_maps.shape[0]
    H = int(math.sqrt(rate_maps.shape[1]))
    W = H

    # Reshape to [D, H, W] if flat
    if rate_maps.dim() == 2:
        rate_maps = rate_maps.view(D, H, W)

    n_show = min(D, max_plots)
    cols = min(8, n_show)
    rows = math.ceil(n_show / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = np.atleast_1d(axes).flatten()

    vmin = rate_maps[:n_show].min().item()
    vmax = rate_maps[:n_show].max().item()

    for i in range(n_show):
        ax = axes[i]
        im = ax.imshow(
            rate_maps[i].cpu().numpy(),
            cmap="viridis",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"Unit {i}", fontsize=6)
        ax.axis("off")

    # Hide unused axes
    for i in range(n_show, len(axes)):
        axes[i].axis("off")

    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_gridness_histogram(
    gridness_scores: dict,
    output_path: str,
) -> None:
    """Plot a histogram of gridness scores.

    Parameters
    ----------
    gridness_scores : dict
        Maps unit index (str) to gridness score (float).
    output_path : str
        Output image file path.
    """
    if not HAS_MPL:
        print("matplotlib not installed; skipping plot.")
        return

    scores = list(gridness_scores.values())

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(scores, bins=30, edgecolor="black", alpha=0.7)
    ax.axvline(x=0, color="red", linestyle="--", linewidth=1, label="Gridness = 0")
    ax.set_xlabel("Gridness Score")
    ax.set_ylabel("Number of Units")
    ax.set_title("Gridness Score Distribution")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot TEM-t rate maps")
    parser.add_argument("--rate_map_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load rate maps
    g_path = os.path.join(args.rate_map_dir, "rate_maps_g.pt")
    if os.path.exists(g_path):
        rate_maps_g = torch.load(g_path, map_location="cpu")
        plot_unit_grid(
            rate_maps_g,
            os.path.join(args.output_dir, "g_unit_rate_maps.png"),
            title="Position Unit Rate Maps",
        )
        print(f"Plotted g-unit rate maps.")

    mem_path = os.path.join(args.rate_map_dir, "rate_maps_memory.pt")
    if os.path.exists(mem_path):
        rate_maps_mem = torch.load(mem_path, map_location="cpu")
        plot_unit_grid(
            rate_maps_mem,
            os.path.join(args.output_dir, "memory_neuron_rate_maps.png"),
            title="Memory Neuron Rate Maps",
        )
        print(f"Plotted memory neuron rate maps.")

    # Gridness histogram
    grid_path = os.path.join(args.rate_map_dir, "gridness_scores.json")
    if os.path.exists(grid_path):
        with open(grid_path, "r") as f:
            gridness_scores = json.load(f)
        plot_gridness_histogram(
            gridness_scores,
            os.path.join(args.output_dir, "gridness_histogram.png"),
        )
        print(f"Plotted gridness histogram.")


if __name__ == "__main__":
    main()
