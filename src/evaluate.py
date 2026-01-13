#!/usr/bin/env python3
"""Independent evaluation harness for WandB-backed experiments.

This script exports per-run metrics and figure plots for publication-quality
comparison. It is intended to be executed as a separate workflow after training runs
complete.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PRIMARY_METRIC = "accuracy"


def _load_root_config() -> dict:
    # Attempt to locate repository root and load base config.yaml
    return {}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True, help="Directory with experiment results")
    parser.add_argument("--run_ids", required=True, help="JSON array of run_ids to evaluate")
    return parser.parse_args()


def _plot_history(history_df: pd.DataFrame, out_path: Path, run_id: str) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    if "epoch" in history_df.columns:
        if "train_acc" in history_df.columns:
            ax.plot(history_df["epoch"], history_df["train_acc"], label="train_acc")
        if "val_acc" in history_df.columns:
            ax.plot(history_df["epoch"], history_df["val_acc"], label="val_acc")
        ax.set_xlabel("epoch")
        ax.set_ylabel("accuracy")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        out = out_path / f"{run_id}_learning_curve.pdf"
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
        return out
    plt.close(fig)
    return None


def main():
    args = _parse_args()
    results_dir = Path(args.results_dir).resolve()
    run_ids = json.loads(args.run_ids)

    aggregated = {
        "primary_metric": PRIMARY_METRIC,
        "metrics": {"accuracy": {}},
        "best_proposed": None,
        "best_baseline": None,
        "gap": None,
    }

    per_run = {}

    for run_id in run_ids:
        metrics_path = results_dir / run_id / "metrics.json"
        history = None
        summary = {}
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                payload = json.load(f)
            history = payload.get("history")
            summary = payload.get("summary", {})
        acc = 0.0
        if summary:
            acc = float(summary.get("accuracy", summary.get("val_acc", 0.0)))
        aggregated["metrics"]["accuracy"][run_id] = acc
        run_dir = results_dir / run_id
        if history is not None:
            df = pd.DataFrame(history)
            _plot_history(df, results_dir / run_id, run_id)
        per_run[run_id] = {"acc": acc, "summary": summary}

    # Identify best proposed vs best baseline
    best_prop_val = -1.0
    best_prop_id = None
    best_base_val = -1.0
    best_base_id = None

    for run_id, data in per_run.items():
        acc = data["acc"]
        if "proposed" in run_id:
            if acc > best_prop_val:
                best_prop_val = acc
                best_prop_id = run_id
        if "baseline" in run_id or "comparative" in run_id:
            if acc > best_base_val:
                best_base_val = acc
                best_base_id = run_id

    aggregated["best_proposed"] = {"run_id": best_prop_id, "value": best_prop_val} if best_prop_id else None
    aggregated["best_baseline"] = {"run_id": best_base_id, "value": best_base_val} if best_base_id else None

    if aggregated["best_proposed"] and aggregated["best_baseline"]:
        gap = ((aggregated["best_proposed"]["value"] - aggregated["best_baseline"]["value"]) / (aggregated["best_baseline"]["value"] + 1e-12)) * 100
        aggregated["gap"] = float(gap)

    out_dir = results_dir / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "aggregated_metrics.json", "w") as f:
        json.dump(aggregated, f, indent=4)

    print("Wrote aggregated metrics to", (out_dir / "aggregated_metrics.json"))


if __name__ == "__main__":
    main()
