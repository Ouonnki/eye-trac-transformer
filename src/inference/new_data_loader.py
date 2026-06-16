# -*- coding: utf-8 -*-
"""Public loading API for the three new-data tasks."""

from pathlib import Path
from typing import List

import numpy as np

from src.inference.new_data_csv import (
    ENTER_GROUP_GAP_US,
    build_segment,
    read_trial_windows,
)
from src.inference.new_data_features import (
    SAMPLE_INTERVAL_US,
    downsample_gaze_points,
    extract_gaze_features,
    normalize_subject_samples,
    sample_interval_warnings,
)
from src.inference.new_data_types import InferenceSample, TaskSpec


MIN_SEGMENT_POINTS = 2


def load_task_file(
    path: Path,
    task_spec: TaskSpec,
    screen_width: int,
    screen_height: int,
) -> InferenceSample:
    windows = read_trial_windows(path, screen_width, screen_height)
    segments: List[np.ndarray] = []
    starts: List[int] = []
    ends: List[int] = []
    warnings: List[str] = []
    for window in windows:
        gaze_points, end_us, segment_warnings = build_segment(window, task_spec)
        warnings.extend(segment_warnings)
        if gaze_points is None:
            continue
        sampled = downsample_gaze_points(gaze_points)
        if len(sampled) < MIN_SEGMENT_POINTS:
            warnings.append(f"trial {window.number}: 降采样后有效眼动点少于2个")
            continue
        warnings.extend(sample_interval_warnings(sampled, window.number))
        segments.append(extract_gaze_features(sampled, screen_width, screen_height))
        starts.append(window.start_us)
        ends.append(end_us)
    if not segments:
        raise ValueError(f"{path}: 没有生成任何有效片段")
    return InferenceSample(
        subject_id=_subject_id_from_path(path),
        task_id=task_spec.task_id,
        task_name=task_spec.task_name,
        task_conditions=task_spec.task_conditions,
        segments=tuple(segments),
        source_path=path,
        warnings=tuple(warnings),
        segment_start_times_us=tuple(starts),
        segment_end_times_us=tuple(ends),
        task_key=task_spec.key,
    )


def _subject_id_from_path(path: Path) -> str:
    subject_id = path.name.split("_", maxsplit=1)[0]
    if not subject_id:
        raise ValueError(f"{path}: 无法从文件名提取被试编号")
    return subject_id


__all__ = [
    "ENTER_GROUP_GAP_US",
    "SAMPLE_INTERVAL_US",
    "downsample_gaze_points",
    "extract_gaze_features",
    "load_task_file",
    "normalize_subject_samples",
]
