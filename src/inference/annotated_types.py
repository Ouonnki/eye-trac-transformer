# -*- coding: utf-8 -*-
"""Shared types for annotated new-data inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Tuple

from src.inference.annotated_csv import RawGroups, ScreenSize
from src.inference.new_data_types import InferenceSample, TaskConditions


@dataclass(frozen=True)
class AnnotatedRecord:
    sample: InferenceSample
    segment_id: str
    segment_kind: str
    source_paths: Tuple[Path, ...]
    status: str = "ready"
    warnings: Tuple[str, ...] = ()


@dataclass(frozen=True)
class IssueRecord:
    task_key: str
    subject_id: str
    segment_id: str
    source_paths: Tuple[Path, ...]
    reason: str
    warnings: Tuple[str, ...] = ()


@dataclass(frozen=True)
class AnnotatedBuild:
    records: Tuple[AnnotatedRecord, ...]
    issues: Tuple[IssueRecord, ...]
    conditions_by_task: Mapping[str, Tuple[TaskConditions, ...]]


@dataclass(frozen=True)
class RecordInput:
    path: Path
    sample: InferenceSample
    segment_id: str
    raw_groups: RawGroups
    screen_size: ScreenSize
