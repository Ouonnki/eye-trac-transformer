# -*- coding: utf-8 -*-
"""Feature building for Schulte raw task samples."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence, Tuple

import numpy as np

from src.data.schemas import ClickPoint, GazePoint, TaskConfig
from src.inference.new_data_types import InferenceSample
from src.inference.schulte_raw_types import SchulteQuestionInfo
from src.segmentation.event_segmenter import AdaptiveSegmenter


FEATURE_DIM = 7
X_FEATURE = 0
Y_FEATURE = 1
DT_FEATURE = 2
VELOCITY_FEATURE = 3
ACCELERATION_FEATURE = 4
DIRECTION_FEATURE = 5
DIRECTION_CHANGE_FEATURE = 6
MILLISECONDS_PER_SECOND = 1000.0
MIN_DT_MS = 1.0
MIN_FEATURE_POINTS = 2


@dataclass(frozen=True)
class SampleBuildInput:
    subject_id: str
    source_path: Path
    question: SchulteQuestionInfo
    clicks: Sequence[ClickPoint]
    gaze_points: Sequence[GazePoint]
    grid_layout: Mapping[int, Tuple[float, float]]
    warnings: Tuple[str, ...]
    screen_size: Tuple[int, int]


def build_inference_sample(build_input: SampleBuildInput) -> InferenceSample:
    segments = _segments(build_input)
    features = tuple(
        extract_segment_features(segment.gaze_points, build_input.screen_size, index == 0)
        for index, segment in enumerate(segments)
    )
    valid_features = tuple(feature for feature in features if len(feature) >= MIN_FEATURE_POINTS)
    if not valid_features:
        raise ValueError(f"{build_input.source_path}: 没有生成任何有效片段")
    return InferenceSample(
        subject_id=build_input.subject_id,
        task_id=build_input.question.task_id,
        task_name=f"题目{build_input.question.task_id}",
        task_conditions=build_input.question.task_conditions,
        segments=valid_features,
        source_path=build_input.source_path,
        warnings=build_input.warnings,
        task_key="schulte_raw",
        condition_label=f"q{build_input.question.task_id:02d}",
    )


def extract_segment_features(
    points: Sequence[GazePoint],
    screen_size: Tuple[int, int],
    is_first_segment: bool,
) -> np.ndarray:
    if len(points) < MIN_FEATURE_POINTS:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32)
    features = np.zeros((len(points), FEATURE_DIM), dtype=np.float32)
    state = (0.0, 0.0)
    for index, point in enumerate(points):
        _set_coordinates(features, index, point, screen_size=screen_size)
        state = _set_dynamic_features(
            features,
            index,
            points,
            state=state,
            is_first_segment=is_first_segment,
        )
    return features


def _segments(build_input: SampleBuildInput):
    task_config = TaskConfig(
        task_id=build_input.question.task_id,
        grid_size=build_input.question.grid_size,
        number_range=build_input.question.number_range,
        click_disappear=build_input.question.click_disappear,
        has_distractor=build_input.question.has_distractor,
        distractor_count=build_input.question.distractor_count,
        grid_distractor_count=build_input.question.grid_distractor_count,
        number_distractor_count=build_input.question.number_distractor_count,
    )
    segmenter = AdaptiveSegmenter(
        task_config=task_config,
        grid_layout=dict(build_input.grid_layout),
        screen_width=build_input.screen_size[0],
        screen_height=build_input.screen_size[1],
    )
    segments = segmenter.segment(list(build_input.clicks), list(build_input.gaze_points))
    if not segments:
        raise ValueError(f"{build_input.source_path}: 没有生成任何搜索片段")
    return segments


def _set_coordinates(
    features: np.ndarray,
    index: int,
    point: GazePoint,
    *,
    screen_size: Tuple[int, int],
) -> None:
    features[index, X_FEATURE] = point.x / screen_size[0]
    features[index, Y_FEATURE] = point.y / screen_size[1]


def _set_dynamic_features(
    features: np.ndarray,
    index: int,
    points: Sequence[GazePoint],
    *,
    state: Tuple[float, float],
    is_first_segment: bool,
) -> Tuple[float, float]:
    if index == 0 and is_first_segment:
        return state
    if index == 0:
        return _set_initial_dynamic(features, points)
    return _set_following_dynamic(features, index, points, state=state)


def _set_initial_dynamic(
    features: np.ndarray,
    points: Sequence[GazePoint],
) -> Tuple[float, float]:
    current, next_point = points[0], points[1]
    dt_ms, distance, direction = _movement(current, next_point)
    features[0, DT_FEATURE] = dt_ms
    features[0, VELOCITY_FEATURE] = distance / dt_ms
    features[0, DIRECTION_FEATURE] = direction / math.pi
    return features[0, VELOCITY_FEATURE], direction


def _set_following_dynamic(
    features: np.ndarray,
    index: int,
    points: Sequence[GazePoint],
    *,
    state: Tuple[float, float],
) -> Tuple[float, float]:
    previous_velocity, previous_direction = state
    dt_ms, distance, direction = _movement(points[index - 1], points[index])
    velocity = distance / dt_ms
    features[index, DT_FEATURE] = dt_ms
    features[index, VELOCITY_FEATURE] = velocity
    features[index, ACCELERATION_FEATURE] = (velocity - previous_velocity) / dt_ms
    features[index, DIRECTION_FEATURE] = direction / math.pi
    features[index, DIRECTION_CHANGE_FEATURE] = _direction_change(direction, previous_direction)
    return velocity, direction


def _movement(first: GazePoint, second: GazePoint) -> Tuple[float, float, float]:
    dt_ms = _time_delta_ms(first.timestamp, second.timestamp)
    dx = second.x - first.x
    dy = second.y - first.y
    return dt_ms, math.hypot(dx, dy), math.atan2(dy, dx)


def _time_delta_ms(first: datetime, second: datetime) -> float:
    raw_ms = (second - first).total_seconds() * MILLISECONDS_PER_SECOND
    return max(raw_ms, MIN_DT_MS)


def _direction_change(direction: float, previous_direction: float) -> float:
    change = abs(direction - previous_direction)
    return min(change, 2 * math.pi - change) / math.pi
