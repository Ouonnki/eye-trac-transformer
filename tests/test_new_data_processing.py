# -*- coding: utf-8 -*-
"""Tests for new-data parsing and preprocessing."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.inference.new_data_loader import (
    ENTER_GROUP_GAP_US,
    SAMPLE_INTERVAL_US,
    downsample_gaze_points,
    extract_gaze_features,
    load_task_file,
    normalize_subject_samples,
)
from src.inference.new_data_types import TASK_SPECS, GazeSample, InferenceSample


CSV_FIELDS = [
    "Recording timestamp",
    "Event",
    "Event value",
    "Gaze point X",
    "Gaze point Y",
]


def event_row(timestamp, event, value=""):
    return [timestamp, event, value, "", ""]


def gaze_row(timestamp, x=100, y=200):
    return [timestamp, "", "", x, y]


def write_csv(path: Path, rows, fields=CSV_FIELDS):
    with path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(fields)
        writer.writerows(rows)


class NewDataProcessingTest(unittest.TestCase):
    def test_complex_uses_first_mouse_event(self):
        rows = [event_row(0, "TaskStart1")]
        rows += [gaze_row(value) for value in (1_000, 9_333, 17_666, 26_000)]
        rows += [event_row(30_000, "MouseEvent", "Down, Left")]
        rows += [gaze_row(35_000), event_row(40_000, "MouseEvent", "Up, Left")]

        sample = self._load("complex", rows)

        self.assertEqual(len(sample.segments), 1)
        self.assertLessEqual(sample.segments[0][-1, 2], 20.0)

    def test_situation_awareness_uses_last_enter_in_first_group(self):
        rows = [event_row(0, "TaskStart1")]
        rows += [gaze_row(value) for value in range(1_000, 250_000, 8_333)]
        rows += [
            event_row(80_000, "KeyboardEvent", "Enter"),
            event_row(120_000, "KeyboardEvent", "Enter"),
            event_row(120_000 + ENTER_GROUP_GAP_US + 1, "KeyboardEvent", "Enter"),
        ]

        sample = self._load("situation_awareness", rows)

        self.assertEqual(len(sample.segments), 1)
        raw_duration = sample.segment_end_times_us[0] - sample.segment_start_times_us[0]
        self.assertEqual(raw_duration, 120_000)

    def test_spot_difference_uses_last_mouse_before_matching_end(self):
        rows = [event_row(0, "TaskStart2")]
        rows += [gaze_row(value) for value in range(1_000, 120_000, 8_333)]
        rows += [
            event_row(40_000, "MouseEvent", "Down, Left"),
            event_row(80_000, "MouseEvent", "Up, Left"),
            event_row(90_000, "TaskEnd2"),
            event_row(100_000, "MouseEvent", "Down, Left"),
        ]

        sample = self._load("spot_difference", rows)

        self.assertEqual(sample.segment_end_times_us, (80_000,))

    def test_downsampling_selects_nearest_bucket_start(self):
        points = [
            GazeSample(timestamp_us=value, x=float(index), y=0.0)
            for index, value in enumerate((0, 8_333, 16_666, 24_999, 33_332))
        ]

        sampled = downsample_gaze_points(points)

        self.assertEqual(SAMPLE_INTERVAL_US, 16_667)
        self.assertEqual([point.timestamp_us for point in sampled], [0, 16_666, 33_332])

    def test_feature_extraction_and_subject_normalization(self):
        points = [
            GazeSample(0, 0.0, 0.0),
            GazeSample(20_000, 20.0, 0.0),
            GazeSample(40_000, 60.0, 0.0),
        ]
        features = extract_gaze_features(points, 100, 100)
        sample = InferenceSample(
            subject_id="1",
            task_id=101,
            task_name="复杂问题解决任务",
            task_conditions=(4, 0, 0, 0, 0),
            segments=(features,),
            source_path=Path("sample.csv"),
        )

        normalized = normalize_subject_samples((sample,))

        self.assertTrue(np.allclose(features[0, 2:7], 0.0))
        self.assertFalse(np.shares_memory(features, normalized[0].segments[0]))
        self.assertTrue(np.allclose(normalized[0].segments[0][0, 2:7], 0.0))
        self.assertLessEqual(np.abs(normalized[0].segments[0][:, 2:5]).max(), 10.0)

    def test_invalid_header_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "缺少必需列"):
            self._load("complex", [], fields=["Recording timestamp"])

    def test_file_without_valid_segment_fails(self):
        rows = [event_row(0, "TaskStart1"), gaze_row(1_000)]
        with self.assertRaisesRegex(ValueError, "没有生成任何有效片段"):
            self._load("complex", rows)

    def _load(self, spec_key, rows, fields=CSV_FIELDS):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "0429324_task.csv"
            write_csv(path, rows, fields)
            return load_task_file(path, TASK_SPECS[spec_key], 1920, 1080)


if __name__ == "__main__":
    unittest.main()
