#!/usr/bin/env python3
"""Train a TEM-t model on 2D grid world navigation.

Usage:
    python script/train_temt.py \
        --model_config config/model_temt.yaml \
        --dataset_config config/dataset_2d_grid.yaml \
        --train_config config/train.yaml \
        --exp_dir experiments/exp01_entorhinal_representations/run001 \
        --seed 0
"""

import argparse
import os
import sys
import json
import yaml

import torch

# Ensure project root is on the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.temt import TEMTModel
from training.envs import GridWorldSpec, TrajectorySampler
from training.losses import TEMTLoss
from training.trainer import Trainer
from training.evaluator import TEMTEvaluator


def load_yaml(path: str) -> dict:
    """Load a YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge_configs(*dicts: dict) -> dict:
    """Naively merge configuration dictionaries (later keys override)."""
    result: dict = {}
    for d in dicts:
        result.update(d)
    return result


def build_model(model_config: dict) -> TEMTModel:
    """Construct a TEMTModel from configuration dict.

    Parameters
    ----------
    model_config : dict
        Parsed model configuration.

    Returns
    -------
    TEMTModel
    """
    cfg = model_config["model"]
    mem_cfg = cfg.get("memory", {})
    pred_cfg = cfg.get("prediction_head", {})

    return TEMTModel(
        n_sensory=cfg["n_sensory"],
        n_actions=cfg["n_actions"],
        d_g=cfg["d_g"],
        d_k=cfg["d_k"],
        d_v=cfg["d_v"],
        max_memory=mem_cfg.get("max_memory", 201),
        activation=cfg.get("activation", "identity"),
        beta0=cfg.get("beta0", 1.0),
        use_landmark_stabilization=cfg.get("use_landmark_stabilization", True),
        memory_dedup=mem_cfg.get("dedup", True),
        use_fixed_layer_norm=cfg.get("use_fixed_layer_norm", True),
        prediction_hidden_dim=pred_cfg.get("hidden_dim", 128),
        landmark_mode=cfg.get("landmark_mode", "sensory_only"),
        init_scale=cfg.get("init_scale", 0.01),
    )


def build_sampler(
    dataset_config: dict,
    seed: int,
    episode_length: int | None = None,
) -> TrajectorySampler:
    """Construct a trajectory sampler from configuration.

    Parameters
    ----------
    dataset_config : dict
        Parsed dataset configuration.
    seed : int
        Random seed.
    episode_length : int or None
        Override the dataset's episode_length. If None, uses the config value.
        Used for BPTT truncation: training uses short (25) episodes,
        evaluation uses full (200) episodes.

    Returns
    -------
    TrajectorySampler
    """
    cfg = dataset_config["dataset"]
    ep_len = episode_length if episode_length is not None else cfg["episode_length"]

    spec = GridWorldSpec(
        height=cfg["height"],
        width=cfg["width"],
        n_actions=cfg.get("n_actions", 4),
        boundary=cfg.get("boundary", "stay"),
    )

    return TrajectorySampler(
        spec=spec,
        n_sensory=cfg["n_sensory"],
        episode_length=ep_len,
        n_envs=cfg["n_train_envs"],
        seed=seed,
    )


def build_val_sampler(
    dataset_config: dict,
    seed: int,
    episode_length: int | None = None,
    n_envs: int | None = None,
) -> TrajectorySampler:
    """Construct a validation trajectory sampler.

    Parameters
    ----------
    dataset_config : dict
        Parsed dataset configuration.
    seed : int
        Random seed (offset to avoid overlap with train).
    episode_length : int or None
        Override episode length for validation.
    n_envs : int or None
        Override number of evaluation environments.

    Returns
    -------
    TrajectorySampler
    """
    cfg = dataset_config["dataset"]
    ep_len = episode_length if episode_length is not None else cfg["episode_length"]
    n_env = n_envs if n_envs is not None else cfg.get("n_val_envs", 100)

    spec = GridWorldSpec(
        height=cfg["height"],
        width=cfg["width"],
        n_actions=cfg.get("n_actions", 4),
        boundary=cfg.get("boundary", "stay"),
    )

    return TrajectorySampler(
        spec=spec,
        n_sensory=cfg["n_sensory"],
        episode_length=ep_len,
        n_envs=n_env,
        seed=seed + 1000000,  # offset seed
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TEM-t model")
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--dataset_config", type=str, required=True)
    parser.add_argument("--train_config", type=str, required=True)
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Load configurations
    model_cfg = load_yaml(args.model_config)
    dataset_cfg = load_yaml(args.dataset_config)
    train_cfg = load_yaml(args.train_config)

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Determine device
    device_str = train_cfg["train"].get("device", "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    # Build components
    model = build_model(model_cfg)
    loss_fn = TEMTLoss(
        lambda_pi=train_cfg["train"]["loss"].get("lambda_pi", 1.0),
        lambda_stable=train_cfg["train"]["loss"].get("lambda_stable", 1.0),
        lambda_g=train_cfg["train"]["loss"].get("lambda_g", 0.1),
        lambda_weight=train_cfg["train"]["loss"].get("lambda_weight", 1e-5),
        lambda_g_l2=train_cfg["train"]["loss"].get("lambda_g_l2", 1e-4),
    )

    optimizer_cfg = train_cfg["train"]["optimizer"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=optimizer_cfg.get("lr", 0.0003),
        weight_decay=optimizer_cfg.get("weight_decay", 0.0),
    )

    # BPTT truncation: training uses short episodes, eval uses full length
    trunc_len = train_cfg["train"].get("truncation_len", 25)
    full_len = dataset_cfg["dataset"].get("episode_length", 200)

    train_sampler = build_sampler(dataset_cfg, args.seed, episode_length=trunc_len)
    val_sampler = build_val_sampler(dataset_cfg, args.seed, episode_length=full_len)
    evaluator = TEMTEvaluator(model, val_sampler, device)

    # Build trainer
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_sampler=train_sampler,
        val_sampler=val_sampler,
        evaluator=evaluator,
        device=device,
        exp_dir=args.exp_dir,
        grad_clip_norm=train_cfg["train"].get("grad_clip_norm", 1.0),
    )

    # Log configs
    os.makedirs(os.path.join(args.exp_dir, "config"), exist_ok=True)
    for name, cfg in [
        ("model_temt.yaml", model_cfg),
        ("dataset_2d_grid.yaml", dataset_cfg),
        ("train.yaml", train_cfg),
    ]:
        with open(os.path.join(args.exp_dir, "config", name), "w") as f:
            yaml.dump(cfg, f)

    # Metadata
    metadata = {
        "experiment_name": os.path.basename(os.path.dirname(args.exp_dir)),
        "run_name": os.path.basename(args.exp_dir),
        "seed": args.seed,
        "model": "TEMT",
    }
    with open(os.path.join(args.exp_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Train
    logging_cfg = train_cfg["train"]["logging"]
    state = trainer.train(
        num_updates=train_cfg["train"]["num_updates"],
        batch_size=train_cfg["train"]["batch_size"],
        eval_interval=logging_cfg.get("eval_interval", 500),
        log_interval=logging_cfg.get("log_interval", 100),
        save_interval=logging_cfg.get("save_interval", 1000),
    )

    print(f"Training complete. Final step: {state.global_step}")
    print(f"Best metric: {state.best_metric}")


if __name__ == "__main__":
    main()
