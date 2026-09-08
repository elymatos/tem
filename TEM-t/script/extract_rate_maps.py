#!/usr/bin/env python3
"""Extract spatial rate maps from a trained TEM-t model.

Rolls out the model on a test grid, collecting position encodings
and attention weights, then computes per-unit and per-memory-slot
rate maps.

Usage:
    python script/extract_rate_maps.py \
        --ckpt experiments/exp01/run001/checkpoints/latest.pt \
        --dataset_config config/dataset_2d_grid.yaml \
        --model_config config/model_temt.yaml \
        --output_dir experiments/exp01/run001/rate_maps \
        --n_steps 50000
"""

import argparse
import os
import sys
import json
import yaml

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from script.train_temt import build_model
from training.envs import GridWorldSpec, TrajectorySampler
from training.metrics import compute_rate_maps, compute_memory_rate_maps, gridness_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract TEM-t rate maps")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--dataset_config", type=str, required=True)
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_steps", type=int, default=50000)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # Load configs
    with open(args.model_config, "r") as f:
        model_cfg = yaml.safe_load(f)
    with open(args.dataset_config, "r") as f:
        dataset_cfg = yaml.safe_load(f)

    device = torch.device(args.device)

    # Build model and load weights
    model = build_model(model_cfg).to(device)
    checkpoint = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Build sampler with a single environment for clean rate maps
    ds = dataset_cfg["dataset"]
    spec = GridWorldSpec(
        height=ds["height"],
        width=ds["width"],
        n_actions=ds.get("n_actions", 4),
        boundary=ds.get("boundary", "stay"),
    )
    sampler = TrajectorySampler(
        spec=spec,
        n_sensory=ds["n_sensory"],
        episode_length=ds.get("episode_length", 200),
        n_envs=1,  # single environment for rate maps
        seed=0,
    )

    # Roll out and collect activations/states
    all_g_seq = []
    all_states = []
    all_attn = []

    total_steps = 0
    batch_size = 1

    pbar = tqdm(total=args.n_steps, desc="Rollout", unit="step", dynamic_ncols=True)

    with torch.no_grad():
        while total_steps < args.n_steps:
            batch = sampler.sample_batch(batch_size, device)
            output = model(batch, return_traces=True, compute_stable_prediction=False)

            # output.g_seq: [B, T+1, d_g] -> use [B, 1:] (positions after each step)
            T = batch.actions.shape[1]
            gs = output.g_seq[:, 1:]  # [B, T, d_g] - g_1 .. g_T
            st = batch.states[:, 1:]  # [B, T]

            all_g_seq.append(gs.cpu())
            all_states.append(st.cpu())

            if output.attn_pi is not None:
                all_attn.append(output.attn_pi.cpu())

            total_steps += T
            pbar.update(T)
            if total_steps >= args.n_steps:
                break

    pbar.close()

    # Concatenate
    g_seq_all = torch.cat(all_g_seq, dim=1)[:, : args.n_steps]      # [1, N, d_g]
    states_all = torch.cat(all_states, dim=1)[:, : args.n_steps]    # [1, N]

    # Compute rate maps
    rate_maps_g = compute_rate_maps(g_seq_all, states_all, spec.n_states)
    # rate_maps_g: [d_g, N_s] -> reshape to [d_g, H, W]
    rate_maps_g_2d = rate_maps_g.view(model.d_g, spec.height, spec.width)

    # Memory neuron rate maps
    rate_maps_memory = None
    if all_attn:
        attn_all = torch.cat(all_attn, dim=1)[:, : args.n_steps]
        rate_maps_memory = compute_memory_rate_maps(attn_all, states_all, spec.n_states)

    # Gridness scores for g units
    gridness_scores = {}
    n_units = min(model.d_g, 50)
    for i in tqdm(range(n_units), desc="Gridness", unit="unit", dynamic_ncols=True):
        rm_2d = rate_maps_g_2d[i]
        score = gridness_score(rm_2d)
        gridness_scores[str(i)] = score

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(rate_maps_g, os.path.join(args.output_dir, "rate_maps_g.pt"))
    if rate_maps_memory is not None:
        torch.save(rate_maps_memory, os.path.join(args.output_dir, "rate_maps_memory.pt"))
    with open(os.path.join(args.output_dir, "gridness_scores.json"), "w") as f:
        json.dump(gridness_scores, f, indent=2)
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump({
            "checkpoint": args.ckpt,
            "n_steps": args.n_steps,
            "height": spec.height,
            "width": spec.width,
        }, f, indent=2)

    print(f"Saved rate maps to {args.output_dir}")
    print(f"Gridness scores (first 10 units):")
    for k, v in list(gridness_scores.items())[:10]:
        print(f"  unit {k}: {v:.4f}")


if __name__ == "__main__":
    main()
