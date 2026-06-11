# -*- coding: utf-8 -*-
"""Unlabeled PyTorch dataset for task-level inference."""

from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from src.inference.new_data_types import InferenceSample


class NewDataInferenceDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[InferenceSample],
        max_seq_len: int,
        input_dim: int,
    ):
        self.samples = tuple(samples)
        self.max_seq_len = max_seq_len
        self.input_dim = input_dim

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        sample = self.samples[index]
        segment_count = len(sample.segments)
        segments = np.zeros(
            (segment_count, self.max_seq_len, self.input_dim),
            dtype=np.float32,
        )
        segment_mask = np.zeros(segment_count, dtype=np.bool_)
        sequence_mask = np.zeros(
            (segment_count, self.max_seq_len),
            dtype=np.bool_,
        )
        for segment_index, features in enumerate(sample.segments):
            sequence_length = min(len(features), self.max_seq_len)
            if sequence_length == 0:
                continue
            segments[segment_index, :sequence_length] = features[:sequence_length]
            segment_mask[segment_index] = True
            sequence_mask[segment_index, :sequence_length] = True
        return {
            "segments": torch.from_numpy(segments),
            "segment_mask": torch.from_numpy(segment_mask),
            "segment_seq_mask": torch.from_numpy(sequence_mask),
            "task_conditions": torch.tensor(sample.task_conditions, dtype=torch.long),
            "subject_id": sample.subject_id,
            "task_id": sample.task_id,
            "task_name": sample.task_name,
            "source_path": sample.source_path,
            "warnings": sample.warnings,
            "missing_tasks": sample.missing_tasks,
        }


def inference_collate_fn(batch: List[Dict]) -> Dict:
    if not batch:
        raise ValueError("推理 batch 不能为空")
    batch_size = len(batch)
    max_segments = max(item["segments"].shape[0] for item in batch)
    max_seq_len = batch[0]["segments"].shape[1]
    input_dim = batch[0]["segments"].shape[2]
    segments = torch.zeros((batch_size, max_segments, max_seq_len, input_dim))
    segment_mask = torch.zeros((batch_size, max_segments), dtype=torch.bool)
    sequence_mask = torch.zeros(
        (batch_size, max_segments, max_seq_len),
        dtype=torch.bool,
    )
    for index, item in enumerate(batch):
        count = item["segments"].shape[0]
        segments[index, :count] = item["segments"]
        segment_mask[index, :count] = item["segment_mask"]
        sequence_mask[index, :count] = item["segment_seq_mask"]
    return {
        "segments": segments,
        "segment_mask": segment_mask,
        "segment_seq_mask": sequence_mask,
        "task_conditions": torch.stack([item["task_conditions"] for item in batch]),
        "subject_ids": [item["subject_id"] for item in batch],
        "task_ids": [item["task_id"] for item in batch],
        "task_names": [item["task_name"] for item in batch],
        "source_paths": [item["source_path"] for item in batch],
        "warnings": [item["warnings"] for item in batch],
        "missing_tasks": [item["missing_tasks"] for item in batch],
    }
