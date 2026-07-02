# -*- coding: utf-8 -*-
"""Shared immutable types for Schulte raw validation and inference."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Tuple

from src.inference.new_data_types import InferenceSample, TaskConditions


@dataclass(frozen=True)
class SchulteQuestionInfo:
    task_id: int
    grid_size: int
    number_range_text: str
    number_range: Tuple[int, int]
    click_disappear: bool
    has_distractor: bool
    distractor_count: int
    grid_distractor_count: int
    number_distractor_count: int
    task_conditions: TaskConditions


@dataclass(frozen=True)
class SchulteIssue:
    subject_id: str
    task_id: int
    source_path: Path
    reason: str
    warnings: Tuple[str, ...] = ()


@dataclass(frozen=True)
class SchulteRecord:
    subject_id: str
    task_id: int
    source_path: Path
    status: str
    question: SchulteQuestionInfo
    sample: Optional[InferenceSample] = None
    raw_duration: Optional[float] = None
    reported_error_count: Optional[int] = None
    actual_error_count: int = 0
    click_count: int = 0
    gaze_count: int = 0
    segment_count: int = 0
    timestamp_fix_count: int = 0
    clipped_coordinate_count: int = 0
    warnings: Tuple[str, ...] = ()

    def with_sample(self, sample: InferenceSample) -> "SchulteRecord":
        return replace(self, sample=sample, segment_count=len(sample.segments))


@dataclass(frozen=True)
class SchulteBuild:
    records: Tuple[SchulteRecord, ...]
    issues: Tuple[SchulteIssue, ...]

