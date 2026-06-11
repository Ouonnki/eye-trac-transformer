# -*- coding: utf-8 -*-
"""Immutable data types for new-data inference."""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Tuple

import numpy as np


TaskConditions = Tuple[int, int, int, int, int]


@dataclass(frozen=True)
class TaskSpec:
    key: str
    directory_name: str
    task_id: int
    task_conditions: TaskConditions


@dataclass(frozen=True)
class GazeSample:
    timestamp_us: int
    x: float
    y: float


@dataclass(frozen=True)
class EventSample:
    timestamp_us: int
    name: str
    value: str


@dataclass(frozen=True)
class TrialWindow:
    number: int
    start_us: int
    events: Tuple[EventSample, ...]
    gaze_points: Tuple[GazeSample, ...]
    invalid_gaze_rows: int = 0
    out_of_bounds_rows: int = 0


@dataclass(frozen=True)
class InferenceSample:
    subject_id: str
    task_id: int
    task_name: str
    task_conditions: TaskConditions
    segments: Tuple[np.ndarray, ...]
    source_path: Path
    warnings: Tuple[str, ...] = ()
    missing_tasks: Tuple[str, ...] = ()
    segment_start_times_us: Tuple[int, ...] = ()
    segment_end_times_us: Tuple[int, ...] = ()

    def with_segments(self, segments: Tuple[np.ndarray, ...]) -> "InferenceSample":
        return replace(self, segments=segments)

    def with_missing_tasks(self, missing_tasks: Tuple[str, ...]) -> "InferenceSample":
        return replace(self, missing_tasks=missing_tasks)


@dataclass(frozen=True)
class PredictionResult:
    sample: InferenceSample
    predicted_index: int
    predicted_label: str
    probabilities: Tuple[float, float, float]

    @property
    def output_value(self) -> int:
        return self.predicted_index + 1


TASK_SPECS: Mapping[str, TaskSpec] = {
    "complex": TaskSpec(
        key="complex",
        directory_name="复杂问题解决任务（4，0，0，0，0）",
        task_id=101,
        task_conditions=(4, 0, 0, 0, 0),
    ),
    "situation_awareness": TaskSpec(
        key="situation_awareness",
        directory_name="情景意识任务（4，1，1，0，1）",
        task_id=102,
        task_conditions=(4, 1, 1, 0, 1),
    ),
    "spot_difference": TaskSpec(
        key="spot_difference",
        directory_name="找不同任务（3，1，0，1，0）",
        task_id=103,
        task_conditions=(3, 1, 0, 1, 0),
    ),
}
