#!/usr/bin/env python3
"""Evaluate zero-shot prediction accuracy of a trained TEM-t model.

Usage:
    python script/eval_zero_shot.py \
        --ckpt experiments/exp01/run001/checkpoints/latest.pt \
        --dataset_config config/dataset_2d_grid.yaml \
        --model_config config/model_temt.yaml \
        --batch_size 64 \
        --n_batches 100
"""

import argparse
import os
import sys
import json
import yaml

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from script.train_temt import build_model, build_sampler
from training.evaluator import TEMTEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TEM-t zero-shot accuracy")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--dataset_config", type=str, required=True)
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_batches", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # Load configs
    with open(args.model_config, "r") as f:
        model_cfg = yaml.safe_load(f)
    with open(args.dataset_config, "r") as f:
        dataset_cfg = yaml.safe_load(f)

    # Device
    device = torch.device(args.device)

    # Build model and load checkpoint
    model = build_model(model_cfg).to(device)
    checkpoint = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Build test sampler
    sampler = build_sampler(dataset_cfg, args.seed)

    # Evaluate
    evaluator = TEMTEvaluator(model, sampler, device)
    result = evaluator.evaluate_zero_shot(
        batch_size=args.batch_size,
        n_batches=args.n_batches,
    )

    print(json.dumps(result.metrics, indent=2))


if __name__ == "__main__":
    main()
