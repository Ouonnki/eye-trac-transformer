# -*- coding: utf-8 -*-
"""Full-data task-level training loader utilities."""

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


INVERSE_MODE = "inverse"
INVERSE_SQRT_MODE = "inverse_sqrt"
EFFECTIVE_NUM_MODE = "effective_num"
MANUAL_WEIGHT_MODE = "manual"
AUTO_WEIGHT_MODE = "auto"
NONE_WEIGHT_MODE = "none"
DEFAULT_ORDINAL_CLIP = [0.5, 3.0]


@dataclass(frozen=True)
class FullDataLoaderConfig:
    batch_size: int
    num_workers: int
    pin_memory: bool
    use_balanced_sampler: bool
    balanced_sampler_mode: str = EFFECTIVE_NUM_MODE
    balanced_sampler_beta: float = 0.999


@dataclass(frozen=True)
class FullDataTrainLoader:
    train_loader: DataLoader
    train_indices: Tuple[int, ...]


@dataclass(frozen=True)
class IndexSubset(Dataset):
    dataset: Dataset
    indices: Tuple[int, ...]

    def __getitem__(self, index: int):
        return self.dataset[self.indices[index]]

    def __len__(self) -> int:
        return len(self.indices)


def build_full_data_train_loader(
    dataset: Dataset,
    options: FullDataLoaderConfig,
    collate_fn: Callable,
) -> FullDataTrainLoader:
    if len(dataset) == 0:
        raise ValueError("full-data training requires at least one sample")

    train_indices = tuple(range(len(dataset)))
    train_subset = IndexSubset(dataset=dataset, indices=train_indices)
    sampler = _build_sampler(dataset, train_indices, options)

    train_loader = DataLoader(
        train_subset,
        batch_size=options.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=options.num_workers,
        pin_memory=options.pin_memory,
    )
    return FullDataTrainLoader(train_loader=train_loader, train_indices=train_indices)


def _build_sampler(
    dataset: Dataset,
    indices: Sequence[int],
    options: FullDataLoaderConfig,
) -> Optional[WeightedRandomSampler]:
    if not options.use_balanced_sampler:
        return None

    labels = _extract_labels(dataset, indices)
    sample_weights = compute_sample_weights(
        labels=labels,
        mode=options.balanced_sampler_mode,
        beta=options.balanced_sampler_beta,
    )
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float32),
        num_samples=len(indices),
        replacement=True,
    )


def _extract_labels(dataset: Dataset, indices: Sequence[int]) -> np.ndarray:
    samples = getattr(dataset, "samples", None)
    if samples is None:
        raise ValueError("balanced sampling requires dataset.samples with label fields")

    labels = []
    for index in indices:
        sample = samples[index]
        if "label" not in sample:
            raise ValueError("balanced sampling requires every sample to include label")
        labels.append(sample["label"])
    return np.asarray(labels, dtype=np.int64)


def compute_sample_weights(labels: np.ndarray, mode: str, beta: float) -> np.ndarray:
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) == 0:
        raise ValueError("cannot compute sample weights for empty labels")

    if mode == INVERSE_MODE:
        class_weights = 1.0 / counts
    elif mode == INVERSE_SQRT_MODE:
        class_weights = 1.0 / np.sqrt(counts)
    elif mode == EFFECTIVE_NUM_MODE:
        effective_num = (1.0 - np.power(beta, counts)) / (1.0 - beta)
        class_weights = 1.0 / effective_num
    else:
        raise ValueError(f"unknown balanced sampler mode: {mode}")

    normalized_weights = class_weights / class_weights.sum() * len(classes)
    weight_by_class = dict(zip(classes.tolist(), normalized_weights.tolist()))
    return np.asarray([weight_by_class[int(label)] for label in labels])


def build_class_weights(training: dict, labels: np.ndarray) -> Optional[torch.Tensor]:
    if not training.get("use_class_weights", False):
        return None
    if training.get("class_weights_mode") == MANUAL_WEIGHT_MODE:
        return torch.tensor(training["class_weights"], dtype=torch.float32)

    classes, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    weights = total / (len(classes) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def build_ordinal_pos_weight(
    training: dict,
    labels: np.ndarray,
    num_classes: int,
) -> Optional[torch.Tensor]:
    mode = training.get("ordinal_pos_weight_mode", AUTO_WEIGHT_MODE)
    if mode == NONE_WEIGHT_MODE:
        return None
    if mode == MANUAL_WEIGHT_MODE:
        weights = training.get("ordinal_pos_weight")
        if weights is None:
            raise ValueError("ordinal_pos_weight_mode=manual requires ordinal_pos_weight")
        return torch.tensor(weights, dtype=torch.float32)
    if mode != AUTO_WEIGHT_MODE:
        raise ValueError(f"unsupported ordinal_pos_weight_mode: {mode}")

    clip = training.get("ordinal_pos_weight_clip", DEFAULT_ORDINAL_CLIP)
    return _compute_ordinal_pos_weight(labels, num_classes, clip)


def _compute_ordinal_pos_weight(labels: np.ndarray, num_classes: int, clip) -> torch.Tensor:
    weights = []
    for threshold_index in range(num_classes - 1):
        positive = float((labels > threshold_index).sum())
        negative = float((labels <= threshold_index).sum())
        raw_weight = 1.0 if positive <= 0 else negative / positive
        weights.append(float(np.clip(raw_weight, float(clip[0]), float(clip[1]))))
    return torch.tensor(weights, dtype=torch.float32)
