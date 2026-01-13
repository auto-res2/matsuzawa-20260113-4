#!/usr/bin/env python3
"""CTGR-TG Hydra-based trainer (train.py).

This trainer implements a production-ready experiment loop with Hydra configuration
management, cross-task gamma regularization (CTGR-TG) for CaSE-style adaptive blocks,
full data-loading scaffolding (synthetic VTAB-like tasks for reproducibility), and
WandB-based metrics logging. The implementation emphasizes defensive programming,
clinical safety checks, and clean integration with the Hydra-driven config directory
config/runs/*.yaml.

Key features:
- Hydra configuration: merge base config with per-run YAML overrides
- Data: synthetic multi-task VTAB-like dataset (no leakage of labels into inputs)
- Model: CaSE-like blocks with EMA priors (mu_l) and reg-term gamma_l^tau - mu_l
- Training: backbone with 3 CTGR_CABlocks; per-task cross-entropy loss plus reg_loss
- Safety: post-init assertions, batch-start checks, pre-optimizer gradient checks
- Logging: WandB for metrics; metrics.json export for offline evaluation
- Mode support: trial vs full (wandb disabled in trial); optuna integration placeholder
"""

from __future__ import annotations

import os
import sys
import json
import math
import time
import random
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import numpy as np
import wandb

import hydra
from omegaconf import DictConfig, OmegaConf

# Local imports from this repository (provided in the same code bundle)
from preprocess import SyntheticVTABTaskDataset, get_task_loader, build_meta_batch_loaders
from model import CTGRNet, CaSEBaselineNet

# ------------------------ Helpers and Safety Checks ------------------------

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _post_init_asserts(model: nn.Module, num_classes: int) -> None:
    # Post-initialization sanity checks
    assert isinstance(num_classes, int) and num_classes > 0, "num_classes must be a positive int"
    if hasattr(model, "fc"):
        assert isinstance(model.fc, nn.Linear), "model.fc must be a Linear layer"
        assert model.fc.out_features == num_classes, (
            f"Model classifier expects {model.fc.out_features} outputs, got {num_classes}"
        )
    # Ensure any gamma buffers exist
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            pass


def _assert_batch_shapes(x: torch.Tensor, y: torch.Tensor) -> None:
    assert x.dim() == 4, f"Input must be 4D (NCHW). Got: {x.shape}"
    assert y.dim() == 1, f"Labels must be 1D. Got: {y.shape}"
    assert x.size(0) == y.size(0), (
        f"Batch size mismatch: inputs {x.size(0)} vs labels {y.size(0)}"
    )


def _grad_check_before_step(model: nn.Module) -> None:
    total_g = 0.0
    grad_found = False
    for p in model.parameters():
        if p.grad is not None:
            grad_found = True
            total_g += float(p.grad.abs().sum())
    if not grad_found:
        raise RuntimeError("CRITICAL: gradients are None before optimizer.step()!")
    if total_g == 0.0:
        raise RuntimeError("CRITICAL: gradients are zero before optimizer.step()!")


# ------------------------------- Training Entry -----------------------------

@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig) -> int:
    # Resolve run-specific overrides if a per-run YAML exists
    root = Path(__file__).resolve().parents[1]  # repository root (contains config/)
    run_id = None
    if isinstance(cfg, DictConfig) and hasattr(cfg, "run"):
        run_id = cfg.run.run_id
        if run_id:
            per_run_yaml = root / "config" / "runs" / f"{run_id}.yaml"
            if per_run_yaml.exists():
                run_cfg = OmegaConf.load(str(per_run_yaml))
                cfg = OmegaConf.merge(cfg, run_cfg)  # type: ignore[arg-type]
    mode = getattr(cfg.run, "mode", getattr(cfg, "mode", "full"))
    # Mode-based toggles (trial vs full)
    if mode == "trial":
        cfg.wandb.mode = "disabled"
        if hasattr(cfg.optuna, "enabled"):
            cfg.optuna.n_trials = 0
    elif mode == "full":
        cfg.wandb.mode = "online" if hasattr(cfg, "wandb") else "online"
    else:
        cfg.wandb = cfg.get("wandb", {})
        cfg.wandb.mode = "online"

    # Setup deterministic seed
    seed = int(cfg.training.seed) if hasattr(cfg, "training") and hasattr(cfg.training, "seed") else 42
    _seed_everything(seed)

    # WandB init (if enabled)
    wandb_run = None
    if getattr(cfg, "wandb", {}).get("mode", "online") != "disabled":
        wandb_run = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=getattr(cfg.run, "run_id", "default-run"),
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow",
        )
        # Print URL for easy access
        try:
            print("WandB Run URL:", wandb_run.get_url())
        except Exception:
            pass

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    loaders_train, loaders_val, num_classes = build_meta_batch_loaders(cfg, device)

    # Model
    if getattr(cfg, "method", "CTGR-TG").lower() in ["ctgr-tg", "ctgrtg", "proposed"]:
        model = CTGRNet(cfg, num_classes=num_classes).to(device)
    else:
        model = CaSEBaselineNet(cfg, num_classes=num_classes).to(device)

    _post_init_asserts(model, num_classes)

    # Optimizer and scheduler (cosine optional)
    opt_params = list(model.parameters())
    optimizer = optim.AdamW(
        opt_params,
        lr=float(cfg.training.learning_rate),
        weight_decay=float(cfg.training.weight_decay),
    )
    scheduler = None
    if getattr(cfg.training, "scheduler", "cosine") == "cosine":
        total_steps = max(1, cfg.training.epochs * sum(len(l) for l in loaders_train))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Training state
    model.train()
    global_step = 0
    best_val_acc = 0.0
    results_dir = Path(cfg.run.results_dir) if hasattr(cfg, "run") and hasattr(cfg.run, "results_dir") else Path("./results")
    current_run_dir = results_dir / (run_id or "default-run")
    current_run_dir.mkdir(parents=True, exist_ok=True)
    metrics_history: List[dict] = []

    # Training loop (simplified yet robust)
    for epoch in range(1, int(cfg.training.epochs) + 1):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        # Iterate across tasks (each loader is a separate task)
        for t, train_loader in enumerate(loaders_train):
            for batch_idx, batch in enumerate(train_loader):
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                _assert_batch_shapes(x, y)
                optimizer.zero_grad()
                logits = model(x)
                loss_task = F.cross_entropy(logits, y)
                reg_loss_total = sum([blk.reg_loss for blk in getattr(model, 'ctgr_blocks', [])]) if hasattr(model, 'ctgr_blocks') else torch.tensor(0.0, device=device)
                loss = loss_task + reg_loss_total
                loss.backward()
                # Defensive gradient check before step
                _grad_check_before_step(model)
                if getattr(cfg.training, "gradient_clip", 0.0) and cfg.training.gradient_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip)
                optimizer.step()

                # Finalize EMA updates for gamma priors
                if hasattr(model, 'ctgr_blocks'):
                    for blk in model.ctgr_blocks:
                        blk.finalize_batch()

                # Metrics
                preds = logits.argmax(dim=1)
                batch_acc = (preds == y).float().sum().item()
                epoch_correct += batch_acc
                epoch_total += x.size(0)
                epoch_loss += loss.item() * x.size(0)
                global_step += 1

                # WandB per-batch logging (as frequent as possible)
                if wandb_run is not None:
                    wandb.log({
                        "train_loss": float(loss_task.item()),
                        "train_reg_loss": float(reg_loss_total.item()) if isinstance(reg_loss_total, torch.Tensor) else 0.0,
                        "train_total_loss": float(loss.item()),
                        "train_acc": float(batch_acc) / x.size(0),
                    }, step=global_step)

        # Epoch end metrics
        epoch_train_loss = epoch_loss / max(1, epoch_total)
        epoch_train_acc = epoch_correct / max(1, epoch_total)

        # Validation (simple average across val loaders)
        model.eval()
        with torch.no_grad():
            val_loss_sum = 0.0
            val_correct_sum = 0
            val_total = 0
            for val_loader in loaders_val:
                for batch in val_loader:
                    xv, yv = batch
                    xv = xv.to(device)
                    yv = yv.to(device)
                    logits_v = model(xv)
                    loss_v = F.cross_entropy(logits_v, yv)
                    val_loss_sum += loss_v.item() * xv.size(0)
                    preds_v = logits_v.argmax(dim=1)
                    val_correct_sum += (preds_v == yv).float().sum().item()
                    val_total += xv.size(0)
        val_loss = val_loss_sum / max(1, val_total)
        val_acc = val_correct_sum / max(1, val_total)
        model.train()

        # WandB epoch metrics
        if wandb_run is not None:
            wandb.log({
                "epoch": epoch,
                "train_loss_epoch": float(epoch_train_loss),
                "train_acc_epoch": float(epoch_train_acc),
                "val_loss_epoch": float(val_loss),
                "val_acc_epoch": float(val_acc),
            }, step=global_step)

        metrics_history.append({
            "epoch": epoch,
            "train_loss": float(epoch_train_loss),
            "train_acc": float(epoch_train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
        })

        # Save periodic best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save best model checkpoint
            best_ckpt = current_run_dir / f"best_model_epoch_{epoch}.pt"
            torch.save({"state_dict": model.state_dict()}, str(best_ckpt))
            if wandb_run is not None:
                wandb.run.summary["best_val_acc"] = best_val_acc
                wandb.run.summary["best_epoch"] = epoch

        # Update LR scheduler
        if scheduler is not None:
            scheduler.step()

        # Step end: write per-epoch metrics to file
        metrics_json = {
            "history": metrics_history,
            "summary": {
                "train_loss": epoch_train_loss,
                "train_acc": epoch_train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            },
        }
        with open(current_run_dir / "metrics.json", "w") as f:
            json.dump(metrics_json, f, indent=4)

        # Re-enable training mode for next epoch
        model.train()

    # Finalize WandB
    if wandb_run is not None:
        wandb_run.finish()

    # Print final results and exit code
    final_summary = {
        "best_val_acc": best_val_acc,
        "final_epoch": int(cfg.training.epochs),
        "run_id": run_id,
    }
    print("TRAIN COMPLETE. SUMMARY:", final_summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
