# -*- coding: utf-8 -*-
"""Streaming CSV parsing and task-specific event segmentation."""

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from src.inference.new_data_types import (
    EventSample,
    GazeSample,
    TaskSpec,
    TrialWindow,
)


ENTER_GROUP_GAP_US = 100_000
MIN_SEGMENT_POINTS = 2
TASK_START_PATTERN = re.compile(r"TaskStart(\d+)")
REQUIRED_COLUMNS = {
    "Recording timestamp",
    "Event",
    "Event value",
    "Gaze point X",
    "Gaze point Y",
}


@dataclass
class _TrialBuilder:
    number: int
    start_us: int
    events: List[EventSample] = field(default_factory=list)
    gaze_points: List[GazeSample] = field(default_factory=list)
    invalid_gaze_rows: int = 0
    out_of_bounds_rows: int = 0

    def freeze(self) -> TrialWindow:
        return TrialWindow(
            number=self.number,
            start_us=self.start_us,
            events=tuple(self.events),
            gaze_points=tuple(self.gaze_points),
            invalid_gaze_rows=self.invalid_gaze_rows,
            out_of_bounds_rows=self.out_of_bounds_rows,
        )


def read_trial_windows(
    path: Path,
    screen_width: int,
    screen_height: int,
) -> Tuple[TrialWindow, ...]:
    windows: List[TrialWindow] = []
    current = None
    with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        _validate_columns(path, reader.fieldnames)
        for row_number, row in enumerate(reader, start=2):
            timestamp_us = _parse_timestamp(path, row_number, row)
            event_name = (row["Event"] or "").strip()
            match = TASK_START_PATTERN.fullmatch(event_name)
            if match:
                if current is not None:
                    windows.append(current.freeze())
                current = _TrialBuilder(int(match.group(1)), timestamp_us)
            if current is None:
                continue
            _append_event(current, timestamp_us, event_name, row["Event value"])
            _append_gaze(current, timestamp_us, row, screen_width, screen_height)
    if current is not None:
        windows.append(current.freeze())
    if not windows:
        raise ValueError(f"{path}: 未找到 TaskStartN 事件")
    return tuple(windows)


def build_segment(
    window: TrialWindow,
    task_spec: TaskSpec,
) -> Tuple[Tuple[GazeSample, ...] | None, int, Tuple[str, ...]]:
    end_us, warnings = _resolve_end_time(window, task_spec)
    warnings.extend(_window_data_warnings(window))
    if end_us is None:
        return None, window.start_us, tuple(warnings)
    gaze = tuple(
        point
        for point in window.gaze_points
        if window.start_us <= point.timestamp_us <= end_us
    )
    if len(gaze) < MIN_SEGMENT_POINTS:
        warnings.append(f"trial {window.number}: 窗口内有效眼动点少于2个")
        return None, end_us, tuple(warnings)
    return gaze, end_us, tuple(warnings)


def _validate_columns(path: Path, fieldnames) -> None:
    columns = set(fieldnames or ())
    missing = sorted(REQUIRED_COLUMNS - columns)
    if missing:
        raise ValueError(f"{path}: 缺少必需列: {', '.join(missing)}")


def _parse_timestamp(path: Path, row_number: int, row: Dict[str, str]) -> int:
    value = (row["Recording timestamp"] or "").strip()
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{path}:{row_number}: Recording timestamp 无效: {value}") from error


def _append_event(
    builder: _TrialBuilder,
    timestamp_us: int,
    event_name: str,
    event_value: str,
) -> None:
    if event_name:
        builder.events.append(
            EventSample(timestamp_us, event_name, (event_value or "").strip())
        )


def _append_gaze(
    builder: _TrialBuilder,
    timestamp_us: int,
    row: Dict[str, str],
    screen_width: int,
    screen_height: int,
) -> None:
    x_value = (row["Gaze point X"] or "").strip()
    y_value = (row["Gaze point Y"] or "").strip()
    if not x_value and not y_value:
        return
    try:
        x_coord = float(x_value)
        y_coord = float(y_value)
    except ValueError:
        builder.invalid_gaze_rows += 1
        return
    if not (0 <= x_coord <= screen_width and 0 <= y_coord <= screen_height):
        builder.out_of_bounds_rows += 1
    builder.gaze_points.append(GazeSample(timestamp_us, x_coord, y_coord))


def _resolve_end_time(
    window: TrialWindow,
    task_spec: TaskSpec,
) -> Tuple[int | None, List[str]]:
    if task_spec.key == "complex":
        end_us = _first_event_time(window, "MouseEvent")
        return end_us, _missing_event_warning(window, "MouseEvent", end_us)
    if task_spec.key == "situation_awareness":
        end_us = _last_enter_in_first_group(window)
        return end_us, _missing_event_warning(window, "KeyboardEvent=Enter", end_us)
    return _spot_difference_end(window)


def _first_event_time(window: TrialWindow, event_name: str) -> int | None:
    return next(
        (event.timestamp_us for event in window.events if event.name == event_name),
        None,
    )


def _last_enter_in_first_group(window: TrialWindow) -> int | None:
    enter_times = [
        event.timestamp_us
        for event in window.events
        if event.name == "KeyboardEvent" and event.value == "Enter"
    ]
    if not enter_times:
        return None
    group_end = enter_times[0]
    for timestamp_us in enter_times[1:]:
        if timestamp_us - group_end > ENTER_GROUP_GAP_US:
            break
        group_end = timestamp_us
    return group_end


def _spot_difference_end(window: TrialWindow) -> Tuple[int | None, List[str]]:
    expected_end = f"TaskEnd{window.number}"
    boundary = next(
        (event.timestamp_us for event in window.events if event.name == expected_end),
        None,
    )
    warnings = []
    if boundary is None:
        warnings.append(f"trial {window.number}: 缺少 {expected_end}，使用下一个TaskStart或文件末尾")
    mouse_times = [
        event.timestamp_us
        for event in window.events
        if event.name == "MouseEvent" and (boundary is None or event.timestamp_us <= boundary)
    ]
    if not mouse_times:
        warnings.append(f"trial {window.number}: 缺少 MouseEvent")
        return None, warnings
    return mouse_times[-1], warnings


def _missing_event_warning(
    window: TrialWindow,
    event_description: str,
    end_us: int | None,
) -> List[str]:
    if end_us is not None:
        return []
    return [f"trial {window.number}: 缺少 {event_description}"]


def _window_data_warnings(window: TrialWindow) -> List[str]:
    warnings = []
    if window.invalid_gaze_rows:
        warnings.append(f"trial {window.number}: {window.invalid_gaze_rows}行眼动坐标无效")
    if window.out_of_bounds_rows:
        warnings.append(f"trial {window.number}: {window.out_of_bounds_rows}行坐标超出屏幕")
    return warnings
