# -*- coding: utf-8 -*-
"""Downsampling, feature extraction, and subject normalization."""

import bisect
import math
from statistics import median
from typing import Sequence, Tuple

import numpy as np

from src.inference.new_data_types import GazeSample, InferenceSample


SAMPLE_INTERVAL_US = 16_667
MIN_EXPECTED_SAMPLE_INTERVAL_US = 14_000
MAX_EXPECTED_SAMPLE_INTERVAL_US = 20_000
NORMALIZATION_EPSILON = 1e-8
NORMALIZATION_CLIP_SIGMA = 10.0
MIN_SEGMENT_POINTS = 2


def downsample_gaze_points(points: Sequence[GazeSample]) -> Tuple[GazeSample, ...]:
    if len(points) < MIN_SEGMENT_POINTS:
        return tuple(points)
    ordered = tuple(sorted(points, key=lambda point: point.timestamp_us))
    timestamps = [point.timestamp_us for point in ordered]
    final_target_limit = timestamps[-1] + SAMPLE_INTERVAL_US // 2
    targets = range(timestamps[0], final_target_limit + 1, SAMPLE_INTERVAL_US)
    selected_indices = [_nearest_index(timestamps, target) for target in targets]
    unique_indices = tuple(dict.fromkeys(selected_indices))
    return tuple(ordered[index] for index in unique_indices)


def extract_gaze_features(
    points: Sequence[GazeSample],
    screen_width: int,
    screen_height: int,
) -> np.ndarray:
    features = np.zeros((len(points), 7), dtype=np.float32)
    previous_velocity = 0.0
    previous_direction = 0.0
    for index, point in enumerate(points):
        features[index, 0] = point.x / screen_width
        features[index, 1] = point.y / screen_height
        if index == 0:
            continue
        previous = points[index - 1]
        dt_ms = max((point.timestamp_us - previous.timestamp_us) / 1_000.0, 1.0)
        dx = point.x - previous.x
        dy = point.y - previous.y
        velocity = math.hypot(dx, dy) / dt_ms
        direction = math.atan2(dy, dx)
        features[index, 2] = dt_ms
        features[index, 3] = velocity
        features[index, 4] = (velocity - previous_velocity) / dt_ms
        features[index, 5] = direction / math.pi
        if index > 1:
            change = abs(direction - previous_direction)
            features[index, 6] = min(change, 2 * math.pi - change) / math.pi
        previous_velocity = velocity
        previous_direction = direction
    return features


def normalize_subject_samples(
    samples: Sequence[InferenceSample],
) -> Tuple[InferenceSample, ...]:
    if not samples:
        return ()
    dynamic = _collect_dynamic_features(samples)
    means = dynamic.mean(axis=0)
    standard_deviations = dynamic.std(axis=0) + NORMALIZATION_EPSILON
    normalized_samples = []
    for sample in samples:
        normalized_segments = tuple(
            _normalize_segment(segment, means, standard_deviations, index == 0)
            for index, segment in enumerate(sample.segments)
        )
        normalized_samples.append(sample.with_segments(normalized_segments))
    return tuple(normalized_samples)


def sample_interval_warnings(
    points: Sequence[GazeSample],
    trial_number: int,
) -> Tuple[str, ...]:
    intervals = [
        current.timestamp_us - previous.timestamp_us
        for previous, current in zip(points, points[1:])
    ]
    median_interval = median(intervals)
    if MIN_EXPECTED_SAMPLE_INTERVAL_US <= median_interval <= MAX_EXPECTED_SAMPLE_INTERVAL_US:
        return ()
    return (f"trial {trial_number}: 降采样中位间隔异常 {median_interval:.0f}us",)


def _nearest_index(timestamps: Sequence[int], target: int) -> int:
    position = bisect.bisect_left(timestamps, target)
    if position == 0:
        return 0
    if position == len(timestamps):
        return len(timestamps) - 1
    before = timestamps[position - 1]
    after = timestamps[position]
    return position - 1 if target - before <= after - target else position


def _collect_dynamic_features(samples: Sequence[InferenceSample]) -> np.ndarray:
    arrays = [
        segment[:, 2:5]
        for sample in samples
        for segment in sample.segments
        if len(segment) > 0
    ]
    if not arrays:
        raise ValueError("被试没有可用于标准化的眼动特征")
    return np.concatenate(arrays, axis=0).astype(np.float64)


def _normalize_segment(
    segment: np.ndarray,
    means: np.ndarray,
    standard_deviations: np.ndarray,
    reset_first_point: bool,
) -> np.ndarray:
    normalized = segment.copy()
    normalized[:, 2:5] = (segment[:, 2:5] - means) / standard_deviations
    normalized[:, 2:5] = np.clip(
        normalized[:, 2:5],
        -NORMALIZATION_CLIP_SIGMA,
        NORMALIZATION_CLIP_SIGMA,
    )
    if reset_first_point and len(normalized):
        normalized[0, 2:7] = 0.0
    return normalized.astype(np.float32)
