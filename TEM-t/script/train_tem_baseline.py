#!/usr/bin/env python3
"""Train the TEM baseline model (Hebbian conjunctive memory) for comparison.

Usage:
    python script/train_tem_baseline.py \
        --model_config config/model_temt.yaml \
        --dataset_config config/dataset_2d_grid.yaml \
        --train_config config/train.yaml \
        --exp_dir experiments/exp02_sample_efficiency_vs_tem/tem_run001 \
        --seed 0
"""

import argparse
import os
import sys
import json
import time
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.baseline_tem import TEMBaseline
from training.envs import GridWorldSpec, TrajectorySampler
from training.metrics import categorical_accuracy, compute_zero_shot_mask
from training.batch import TrajectoryBatch


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model(model_config: dict) -> TEMBaseline:
    """Construct a TEMBaseline from configuration.

    TEM uses different parameters than TEM-t:
      - d_x: sensory projection dim (instead of d_v).
             MUST be small (≤16): the Hebbian matrix M has shape
             [B, d_g*d_x, d_g*d_x]. With d_g=128, d_x=32, each
             matrix per batch item is 4096² × 4B = 67 MB, and
             B=32 pushes it >2 GB. Reduce d_x to keep B × (d_g*d_x)²
             within VRAM. We override the config value for safety.
      - n_attractor_steps: attractor iterations (default 5)
      - No d_k, no landmark stabilizer, no dedup memory.
    """
    cfg = model_config["model"]
    # Use a smaller d_x than the config default to keep d_p manageable.
    # d_x=8 → d_p=1024 → M [B,1024,1024] ≈ 4 MB per batch item.
    d_x = min(cfg.get("d_x", 32), 8)
    # Use smaller d_g than TEM-t to keep Hebbian matrix d_p = d_g * d_x tractable.
    # d_g=64, d_x=8 → d_p=512 → M [B,512,512] ≈ 1 MB per batch item.
    d_g = min(cfg["d_g"], 64)
    d_p = d_g * d_x
    print(f"TEM baseline: d_g={d_g}, d_x={d_x}, d_p={d_p} "
          f"(Hebbian M: {d_p}×{d_p} = {d_p*d_p*4/1024:.0f} KB per batch item)")
    return TEMBaseline(
        n_sensory=cfg["n_sensory"],
        n_actions=cfg["n_actions"],
        d_g=d_g,
        d_x=d_x,
        n_attractor_steps=cfg.get("n_attractor_steps", 5),
        activation=cfg.get("activation", "identity"),
    )


def build_sampler(
    dataset_config: dict,
    seed: int,
    episode_length: int | None = None,
) -> TrajectorySampler:
    cfg = dataset_config["dataset"]
    ep_len = episode_length if episode_length is not None else cfg["episode_length"]

    spec = GridWorldSpec(
        height=cfg["height"],
        width=cfg["width"],
        n_actions=cfg.get("n_actions", 5),
        boundary=cfg.get("boundary", "stay"),
    )

    return TrajectorySampler(
        spec=spec,
        n_sensory=cfg["n_sensory"],
        episode_length=ep_len,
        n_envs=cfg["n_train_envs"],
        seed=seed,
    )


@torch.no_grad()
def evaluate_zero_shot(
    model: TEMBaseline,
    sampler: TrajectorySampler,
    device: torch.device,
    batch_size: int = 64,
    n_batches: int = 20,
) -> dict:
    """Compute zero-shot prediction accuracy for the TEM baseline.

    Uses the model's online prediction (no x_{t+1} leakage) since
    TEM's forward loop predicts BEFORE observing the next sensory.
    """
    model.eval()

    total_correct = 0
    total_steps = 0
    total_zs_correct = 0
    total_zs_steps = 0

    for _ in tqdm(range(n_batches), desc="Eval TEM zero-shot", unit="batch", dynamic_ncols=True):
        batch = sampler.sample_batch(batch_size, device)

        # TEMBaseline.forward returns {"logits": [B,T,N_x], "g_seq": [B,T+1,d_g]}
        output = model(batch)

        preds = output["logits"].argmax(dim=-1)      # [B, T]
        targets = batch.x_ids[:, 1:]                  # [B, T]
        correct = (preds == targets)

        total_correct += correct.sum().item()
        total_steps += correct.numel()

        if batch.states is not None:
            zs_mask = compute_zero_shot_mask(batch.states, batch.actions)
            zs_correct = correct & zs_mask
            total_zs_correct += zs_correct.sum().item()
            total_zs_steps += zs_mask.sum().item()

    return {
        "acc_all": total_correct / total_steps if total_steps > 0 else 0.0,
        "acc_zero_shot": total_zs_correct / total_zs_steps if total_zs_steps > 0 else 0.0,
        "n_zero_shot": total_zs_steps,
    }


def save_checkpoint(model, optimizer, path, global_step):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
    }, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TEM baseline model")
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--dataset_config", type=str, required=True)
    parser.add_argument("--train_config", type=str, required=True)
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Load config
    model_cfg = load_yaml(args.model_config)
    dataset_cfg = load_yaml(args.dataset_config)
    train_cfg = load_yaml(args.train_config)
    train = train_cfg["train"]

    # Seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device
    device_str = train.get("device", "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    # Build
    model = build_model(model_cfg).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train["optimizer"].get("lr", 0.001),
        weight_decay=train["optimizer"].get("weight_decay", 0.0),
    )

    trunc_len = train.get("truncation_len", 25)
    full_len = dataset_cfg["dataset"].get("episode_length", 200)

    train_sampler = build_sampler(dataset_cfg, args.seed, episode_length=trunc_len)
    eval_sampler = build_sampler(dataset_cfg, args.seed + 1000000, episode_length=full_len)

    # Loss weights
    loss_cfg = train.get("loss", {})
    lambda_pi = loss_cfg.get("lambda_pi", 1.0)
    lambda_weight = loss_cfg.get("lambda_weight", 1e-5)
    lambda_g_l2 = loss_cfg.get("lambda_g_l2", 1e-4)

    # Directories
    os.makedirs(args.exp_dir, exist_ok=True)
    os.makedirs(os.path.join(args.exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.exp_dir, "logs"), exist_ok=True)

    log_path = os.path.join(args.exp_dir, "logs", "train_metrics.jsonl")
    eval_path = os.path.join(args.exp_dir, "logs", "eval_metrics.jsonl")

    # Save configs
    os.makedirs(os.path.join(args.exp_dir, "config"), exist_ok=True)
    for name, cfg in [
        ("model_temt.yaml", model_cfg),
        ("dataset_2d_grid.yaml", dataset_cfg),
        ("train.yaml", train_cfg),
    ]:
        with open(os.path.join(args.exp_dir, "config", name), "w") as f:
            yaml.dump(cfg, f)

    with open(os.path.join(args.exp_dir, "metadata.json"), "w") as f:
        json.dump({
            "experiment_name": "exp02_sample_efficiency_vs_tem",
            "run_name": os.path.basename(args.exp_dir),
            "seed": args.seed,
            "model": "TEM",
        }, f, indent=2)

    # ---- Training ----
    num_updates = train.get("num_updates", 10000)
    log_interval = train.get("logging", {}).get("log_interval", 100)
    eval_interval = train.get("logging", {}).get("eval_interval", 500)
    save_interval = train.get("logging", {}).get("save_interval", 2000)
    grad_clip = train.get("grad_clip_norm", 1.0)
    batch_size = train.get("batch_size", 32)

    pbar = tqdm(total=num_updates, desc="Training TEM", unit="step", dynamic_ncols=True)

    for step in range(num_updates):
        model.train()
        optimizer.zero_grad()

        t_start = time.time()
        batch = train_sampler.sample_batch(batch_size, device)

        # Forward: TEM processes the full truncated trajectory
        output = model(batch)
        logits = output["logits"]           # [B, T, N_x]

        # Loss: CE on sensory prediction + L2 penalties
        targets = batch.x_ids[:, 1:]         # [B, T]
        ce_loss = F.cross_entropy(
            logits.reshape(-1, model.n_sensory),
            targets.reshape(-1),
        )

        l2_weight = torch.tensor(0.0, device=device)
        if lambda_weight > 0:
            for p in model.parameters():
                l2_weight = l2_weight + p.pow(2).sum()

        g_l2 = output["g_seq"].pow(2).mean()
        total_loss = lambda_pi * ce_loss + lambda_weight * l2_weight + lambda_g_l2 * g_l2

        total_loss.backward()

        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        dt = time.time() - t_start

        with torch.no_grad():
            acc = (logits.argmax(dim=-1) == targets).float().mean()

        pbar.set_postfix({"loss": f"{total_loss.item():.3f}", "acc": f"{acc.item():.3f}"})
        pbar.update(1)

        # Logging
        if (step + 1) % log_interval == 0:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "global_step": step + 1,
                    "total": total_loss.item(),
                    "ce": ce_loss.item(),
                    "l2_weight": l2_weight.item(),
                    "g_l2": g_l2.item(),
                    "acc_pi": acc.item(),
                    "dt": dt,
                }) + "\n")

        # Evaluation
        if (step + 1) % eval_interval == 0:
            pbar.write(f"--- Eval @ step {step + 1} ---")
            metrics = evaluate_zero_shot(model, eval_sampler, device, batch_size, n_batches=10)
            os.makedirs(os.path.dirname(eval_path), exist_ok=True)
            with open(eval_path, "a") as f:
                f.write(json.dumps({"global_step": step + 1, **metrics}) + "\n")
            pbar.write(
                "  " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            )

        # Checkpoint
        if (step + 1) % save_interval == 0:
            save_checkpoint(
                model, optimizer,
                os.path.join(args.exp_dir, "checkpoints", f"step_{(step+1):06d}.pt"),
                step + 1,
            )
            save_checkpoint(
                model, optimizer,
                os.path.join(args.exp_dir, "checkpoints", "latest.pt"),
                step + 1,
            )

    pbar.close()

    save_checkpoint(
        model, optimizer,
        os.path.join(args.exp_dir, "checkpoints", "latest.pt"),
        num_updates,
    )
    print(f"Training complete. Final step: {num_updates}")


if __name__ == "__main__":
    main()
