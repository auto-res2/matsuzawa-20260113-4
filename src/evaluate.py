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
import seaborn as sns

PRIMARY_METRIC = "accuracy"

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})


def _load_root_config() -> dict:
    # Attempt to locate repository root and load base config.yaml
    return {}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", nargs="?", help="Directory with experiment results (key=value format)")
    parser.add_argument("run_ids", nargs="?", help="JSON array of run_ids to evaluate (key=value format)")
    parser.add_argument("--results_dir", dest="results_dir_flag", help="Directory with experiment results (flag format)")
    parser.add_argument("--run_ids", dest="run_ids_flag", help="JSON array of run_ids to evaluate (flag format)")
    args = parser.parse_args()

    # Support both key=value and --flag formats
    results_dir = None
    run_ids = None

    # Check for flag format first
    if args.results_dir_flag:
        results_dir = args.results_dir_flag
    if args.run_ids_flag:
        run_ids = args.run_ids_flag

    # Check for key=value format
    if args.results_dir and "=" in args.results_dir:
        results_dir = args.results_dir.split("=", 1)[1]
    elif args.results_dir and not results_dir:
        results_dir = args.results_dir

    if args.run_ids and "=" in args.run_ids:
        run_ids = args.run_ids.split("=", 1)[1]
    elif args.run_ids and not run_ids:
        run_ids = args.run_ids

    if not results_dir or not run_ids:
        parser.error("Both results_dir and run_ids are required")

    # Create a namespace with the correct values
    result = argparse.Namespace()
    result.results_dir = results_dir
    result.run_ids = run_ids
    return result


def _plot_history(history_df: pd.DataFrame, out_path: Path, run_id: str) -> Path:
    """Plot learning curve for a single run."""
    fig, ax = plt.subplots(figsize=(8, 6))
    if "epoch" in history_df.columns:
        if "train_acc" in history_df.columns:
            ax.plot(history_df["epoch"], history_df["train_acc"],
                   label="Training Accuracy", linewidth=2, marker='o', markersize=4)
        if "val_acc" in history_df.columns:
            ax.plot(history_df["epoch"], history_df["val_acc"],
                   label="Validation Accuracy", linewidth=2, marker='s', markersize=4)
        ax.set_xlabel("Epoch", fontsize=14, fontweight='bold')
        ax.set_ylabel("Accuracy", fontsize=14, fontweight='bold')
        ax.set_title(f"Learning Curve: {run_id}", fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='best', frameon=True, shadow=True)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_ylim([0, 1.0])
        out = out_path / f"{run_id}_learning_curve.pdf"
        fig.tight_layout()
        fig.savefig(out, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return out
    plt.close(fig)
    return None


def _plot_comparison_bar(per_run: dict, out_dir: Path, metric: str = "accuracy") -> Path:
    """Generate bar chart comparing all runs."""
    fig, ax = plt.subplots(figsize=(12, 7))

    run_ids = list(per_run.keys())
    values = [per_run[rid]["acc"] for rid in run_ids]

    # Shorten run IDs for better display
    short_labels = []
    for rid in run_ids:
        # Extract key parts: comparative/proposed + model name
        if 'proposed' in rid.lower():
            short_labels.append('Proposed\n' + rid.split('-')[-1])
        elif 'comparative' in rid.lower():
            short_labels.append('Baseline\n' + rid.split('-')[-1])
        else:
            short_labels.append(rid)

    # Use different colors for proposed vs baseline
    colors = ['#27ae60' if 'proposed' in rid else '#c0392b' for rid in run_ids]

    x_pos = np.arange(len(run_ids))
    bars = ax.bar(x_pos, values, color=colors, alpha=0.85, edgecolor='black',
                  linewidth=2, width=0.6)

    # Determine appropriate y-axis limit
    max_val = max(values) if values else 0.0
    if max_val > 0:
        y_limit = max_val * 1.25
        label_offset = max_val * 0.03
    else:
        # Special handling for zero values
        y_limit = 1.0
        label_offset = 0.02
        # Add annotation for no training data
        ax.text(0.5, 0.5, 'No training data available yet\n(All accuracies = 0.0)',
               transform=ax.transAxes, fontsize=14, ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6, edgecolor='black', linewidth=2))

    # Add value labels on top of bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        y_pos = height + label_offset if height > 0 else label_offset
        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=12)

    ax.set_xlabel("Model Configuration", fontsize=15, fontweight='bold')
    ax.set_ylabel("Accuracy", fontsize=15, fontweight='bold')
    ax.set_title("Model Performance Comparison", fontsize=18, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(short_labels, fontsize=12, fontweight='semibold')
    ax.set_ylim([0, y_limit])
    ax.grid(True, axis='y', linestyle='--', alpha=0.4, linewidth=1)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#27ae60', edgecolor='black', linewidth=1.5, label='Proposed Method'),
        Patch(facecolor='#c0392b', edgecolor='black', linewidth=1.5, label='Baseline Method')
    ]
    ax.legend(handles=legend_elements, loc='upper left', frameon=True, shadow=True,
             fontsize=12, edgecolor='black', fancybox=True)

    out = out_dir / "accuracy_comparison_bar.pdf"
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return out


def _plot_improvement_gap(aggregated: dict, out_dir: Path) -> Path:
    """Generate visualization of improvement gap between proposed and baseline."""
    fig, ax = plt.subplots(figsize=(10, 7))

    if aggregated["best_proposed"] and aggregated["best_baseline"]:
        baseline_val = aggregated["best_baseline"]["value"]
        proposed_val = aggregated["best_proposed"]["value"]
        gap = aggregated["gap"]

        categories = ['Baseline\n(Best)', 'Proposed\n(Best)']
        values = [baseline_val, proposed_val]
        colors = ['#c0392b', '#27ae60']

        x_pos = np.arange(len(categories))
        bars = ax.bar(x_pos, values, color=colors, alpha=0.85, edgecolor='black',
                     linewidth=2.5, width=0.5)

        # Determine appropriate scaling
        max_val = max(values)
        if max_val > 0:
            y_limit = max_val * 1.35
            label_offset = max_val * 0.03
        else:
            y_limit = 1.0
            label_offset = 0.02
            # Add annotation for zero values
            ax.text(0.5, 0.5, 'No training completed yet\n(Both accuracies = 0.0)\n\nImprovement gap: 0.0%',
                   transform=ax.transAxes, fontsize=14, ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6, edgecolor='black', linewidth=2))

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            y_pos = height + label_offset if height > 0 else label_offset
            ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                    f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=13)

        # Add improvement annotation with arrow
        if gap != 0 and max_val > 0:
            # Draw arrow between bars
            arrow_x = 0.5
            ax.annotate('', xy=(arrow_x, proposed_val), xytext=(arrow_x, baseline_val),
                       arrowprops=dict(arrowstyle='<->', lw=3, color='black', shrinkA=0, shrinkB=0))

            # Add percentage box
            mid_y = (baseline_val + proposed_val) / 2
            sign = '+' if gap > 0 else ''
            ax.text(arrow_x + 0.25, mid_y,
                   f'{sign}{gap:.2f}%', fontsize=13, fontweight='bold',
                   va='center', ha='left',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='gold', alpha=0.8,
                            edgecolor='black', linewidth=2))

        ax.set_ylabel("Accuracy", fontsize=15, fontweight='bold')
        ax.set_title("Performance Improvement: Proposed vs Baseline", fontsize=18, fontweight='bold', pad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(categories, fontsize=13, fontweight='semibold')
        ax.set_ylim([0, y_limit])
        ax.grid(True, axis='y', linestyle='--', alpha=0.4, linewidth=1)

        out = out_dir / "improvement_gap.pdf"
        fig.tight_layout()
        fig.savefig(out, dpi=300, bbox_inches='tight')
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

    # Generate comparison figures
    if len(per_run) > 0:
        print("Generating comparison figures...")
        bar_chart = _plot_comparison_bar(per_run, out_dir)
        if bar_chart:
            print(f"  Created: {bar_chart}")

        gap_chart = _plot_improvement_gap(aggregated, out_dir)
        if gap_chart:
            print(f"  Created: {gap_chart}")

        print("All comparison figures generated successfully.")


if __name__ == "__main__":
    main()
