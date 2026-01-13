#!/usr/bin/env python3
"""Hydra-based orchestrator for single-run experiments.

This module wraps the training script as a subprocess and monitors logs/files.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig) -> int:
    results_dir = Path(cfg.results_dir) if hasattr(cfg, 'results_dir') else Path('./results')
    results_dir.mkdir(parents=True, exist_ok=True)

    # Handle both string override and config object cases
    if isinstance(cfg.run, str):
        # When run is overridden as a string, use run_id from top-level config
        run_id = getattr(cfg, 'run_id', cfg.run)
    else:
        # When run is a config object, get run_id from it
        run_id = getattr(cfg.run, 'run_id', 'default-run')
    mode = getattr(cfg, 'mode', 'full')

    trainer_path = Path(__file__).resolve().parents[0] / 'train.py'
    cmd = [sys.executable, str(trainer_path), f"run_id={run_id}", f"results_dir={str(results_dir)}", f"mode={mode}"]
    log_file = results_dir / f"{run_id}_train.log"
    with open(log_file, 'w') as f:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end='')
            f.write(line)
        proc.wait()
    return proc.returncode


if __name__ == '__main__':
    raise SystemExit(main())
