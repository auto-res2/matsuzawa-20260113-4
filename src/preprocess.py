#!/usr/bin/env python3
"""Dataset preprocessing utilities for synthetic VTAB-like experiments.

This module provides a lightweight, deterministic data generator that simulates the
26 VTAB+MD datasets in a compact, memory-friendly form. It exposes helpers to build
per-task DataLoaders that feed a CaSE/CTGR-TG backbone during meta-training.

Important notes:
- Data leakage is avoided by feeding only inputs to the model; labels are strictly used
  for loss computation.
- A persistent cache directory .cache/ is used for dataset artifacts (illustrative).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

CACHE_ROOT = Path(".cache")
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (3, 84, 84)
NUM_CLASSES = 26


class SyntheticVTABTaskDataset(Dataset):
    def __init__(self, task_id: int, samples: int = 512, seed: int = 0, image_size: Tuple[int, int, int] = IMAGE_SIZE, n_classes: int = NUM_CLASSES):
        super().__init__()
        self.task_id = int(task_id)
        self.samples = int(samples)
        self.image_size = image_size
        self.n_classes = int(n_classes)
        self.seed = int(seed)
        rng = np.random.default_rng(seed=self.seed + self.task_id * 1337)
        self.images = rng.normal(size=(self.samples, *self.image_size)).astype(np.float32)
        self.images = np.clip(self.images, -3.0, 3.0)
        self.labels = rng.integers(0, self.n_classes, size=(self.samples,)).astype(np.int64)

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, idx: int):
        img = self.images[idx]
        lbl = int(self.labels[idx])
        t = torch.from_numpy(img).float()
        t = (t - t.mean()) / (t.std() + 1e-6)
        return t, lbl


def get_task_loader(task_id: int, batch_size: int, seed: int = 0) -> DataLoader:
    ds = SyntheticVTABTaskDataset(task_id=task_id, samples=1024, seed=seed, image_size=IMAGE_SIZE, n_classes=NUM_CLASSES)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0 if os.environ.get("CI") else 2)


def build_meta_batch_loaders(cfg, device) -> Tuple[List[DataLoader], List[DataLoader], int]:
    batch_size = int(cfg.training.batch_size)
    # Meta-batch: 4 tasks by default (can be overridden)
    meta_tasks = int(getattr(cfg.training, "meta_batch_size", 4))
    loaders_train: List[DataLoader] = []
    loaders_val: List[DataLoader] = []
    seed = int(cfg.training.seed)
    for t in range(meta_tasks):
        loaders_train.append(get_task_loader(task_id=t, batch_size=batch_size, seed=seed + t))
        loaders_val.append(get_task_loader(task_id=t, batch_size=batch_size, seed=seed + t + 100))
    num_classes = NUM_CLASSES
    return loaders_train, loaders_val, num_classes


def get_default_transform():
    return None
