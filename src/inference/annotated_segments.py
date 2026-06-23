# -*- coding: utf-8 -*-
"""Segment builders for annotated new-data validation and inference."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from src.inference.annotated_csv import (
    LEVEL_IDS,
    TOPIC_IDS,
    ScreenSize,
    find_child_dir,
    parse_condition_combinations,
    read_situation_awareness_groups,
    read_spot_groups,
    subject_id_from_path,
)
from src.inference.annotated_types import (
    AnnotatedBuild,
    AnnotatedRecord,
    IssueRecord,
    RecordInput,
)
from src.inference.new_data_features import (
    downsample_gaze_points,
    extract_gaze_features,
    normalize_subject_samples,
    sample_interval_warnings,
)
from src.inference.new_data_types import (
    GazeSample,
    InferenceSample,
    TASK_SPECS,
    TaskConditions,
)


DEFAULT_SCREEN_WIDTH = 1920
DEFAULT_SCREEN_HEIGHT = 1080
MIN_SEGMENT_POINTS = 2


def build_situation_awareness_records(
    path: Path,
    screen_width: int,
    screen_height: int,
) -> Tuple[AnnotatedRecord, ...]:
    screen_size = (screen_width, screen_height)
    groups_by_level = read_situation_awareness_groups(path, screen_size)
    records = tuple(
        _build_record(RecordInput(path, _sa_sample(path), level, groups_by_level[level], screen_size))
        for level in LEVEL_IDS
    )
    return _with_combined_record(records)


def build_spot_subject_records(
    subject_id: str,
    task_paths: Mapping[str, Path],
    *screen_args: int,
) -> Tuple[AnnotatedRecord, ...]:
    _require_topics(subject_id, task_paths)
    screen_size = _screen_size_from_args(screen_args)
    records = tuple(
        _build_record(
            RecordInput(path, _spot_sample(subject_id, path), topic, read_spot_groups(path, screen_size), screen_size)
        )
        for topic, path in _ordered_topic_paths(task_paths)
    )
    return _with_combined_record(records)


def build_annotated_records(
    data_dir: Path,
    screen_width: int = DEFAULT_SCREEN_WIDTH,
    screen_height: int = DEFAULT_SCREEN_HEIGHT,
) -> AnnotatedBuild:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"新数据+标注目录不存在: {data_dir}")
    sa_dir = find_child_dir(data_dir, ("情景意识",))
    spot_dir = find_child_dir(data_dir, ("找不同数据",))
    spot_records, spot_issues = _build_spot_records(spot_dir, screen_width, screen_height)
    records = _build_sa_records(sa_dir, screen_width, screen_height) + spot_records
    issues = _read_sa_summary_issues(sa_dir) + spot_issues
    return AnnotatedBuild(records, issues, _conditions_by_task(sa_dir, spot_dir))


def expand_records_by_conditions(
    records: Sequence[AnnotatedRecord],
    conditions_by_task: Mapping[str, Sequence[TaskConditions]],
) -> Tuple[AnnotatedRecord, ...]:
    expanded = []
    for record in records:
        for conditions in _record_conditions(record, conditions_by_task):
            sample = record.sample.with_task_conditions(conditions, _condition_label(conditions))
            expanded.append(replace(record, sample=sample))
    return tuple(expanded)


def _build_sa_records(
    sa_dir: Path,
    screen_width: int,
    screen_height: int,
) -> Tuple[AnnotatedRecord, ...]:
    paths = tuple(sorted(path for path in sa_dir.glob("*.csv") if "summary" not in path.name))
    if not paths:
        raise FileNotFoundError(f"{sa_dir}: 未找到情景意识眼动 CSV")
    return tuple(
        record
        for path in paths
        for record in build_situation_awareness_records(path, screen_width, screen_height)
    )


def _build_spot_records(
    spot_dir: Path,
    screen_width: int,
    screen_height: int,
) -> Tuple[Tuple[AnnotatedRecord, ...], Tuple[IssueRecord, ...]]:
    eye_dir = find_child_dir(spot_dir, ("找不同", "眼动"))
    behavior_dir = find_child_dir(spot_dir, ("行为",))
    subject_paths = _spot_subject_paths(eye_dir)
    records: List[AnnotatedRecord] = []
    issues: List[IssueRecord] = []
    for subject_id in sorted(subject_paths):
        issues.extend(_spot_pairing_issues(subject_id, subject_paths[subject_id], behavior_dir))
        records.extend(build_spot_subject_records(subject_id, subject_paths[subject_id], screen_width, screen_height))
    return tuple(records), tuple(issues)


def _build_record(record_input: RecordInput) -> AnnotatedRecord:
    segments, starts, ends, generated = _feature_segments(record_input)
    sample = replace(
        record_input.sample,
        segments=segments,
        warnings=record_input.raw_groups.warnings + generated,
        segment_start_times_us=starts,
        segment_end_times_us=ends,
    )
    return AnnotatedRecord(
        sample=sample,
        segment_id=record_input.segment_id,
        segment_kind="single",
        source_paths=(record_input.path,),
        warnings=sample.warnings,
    )


def _with_combined_record(records: Tuple[AnnotatedRecord, ...]) -> Tuple[AnnotatedRecord, ...]:
    normalized = normalize_subject_samples(tuple(record.sample for record in records))
    normalized_records = tuple(
        replace(record, sample=sample, warnings=sample.warnings)
        for record, sample in zip(records, normalized)
    )
    combined = _combined_record(normalized_records)
    return normalized_records + (combined,)


def _combined_record(records: Tuple[AnnotatedRecord, ...]) -> AnnotatedRecord:
    first = records[0]
    segments = tuple(segment for record in records for segment in record.sample.segments)
    starts = tuple(time for record in records for time in record.sample.segment_start_times_us)
    ends = tuple(time for record in records for time in record.sample.segment_end_times_us)
    warnings = _unique(warning for record in records for warning in record.warnings)
    sample = replace(
        first.sample,
        segments=segments,
        warnings=warnings,
        segment_start_times_us=starts,
        segment_end_times_us=ends,
    )
    return AnnotatedRecord(sample, "combined", "combined", _source_paths(records), warnings=warnings)


def _feature_segments(
    record_input: RecordInput,
) -> Tuple[Tuple[np.ndarray, ...], Tuple[int, ...], Tuple[int, ...], Tuple[str, ...]]:
    if not record_input.raw_groups.groups:
        raise ValueError(f"{record_input.path}: 未生成任何业务段")
    segments, starts, ends, warnings = [], [], [], []
    for index, (name, points) in enumerate(record_input.raw_groups.groups, start=1):
        sampled = downsample_gaze_points(points)
        if len(sampled) < MIN_SEGMENT_POINTS:
            warnings.append(f"{name}: 降采样后有效眼动点少于 2 个，跳过该内部 segment")
            continue
        warnings.extend(sample_interval_warnings(sampled, index))
        segments.append(extract_gaze_features(sampled, *record_input.screen_size))
        starts.append(sampled[0].timestamp_us)
        ends.append(sampled[-1].timestamp_us)
    if not segments:
        raise ValueError(f"{record_input.path}: {record_input.segment_id} 没有有效内部 segment")
    return tuple(segments), tuple(starts), tuple(ends), tuple(warnings)


def _read_sa_summary_issues(sa_dir: Path) -> Tuple[IssueRecord, ...]:
    paths = tuple(sa_dir.glob("*summary*.csv"))
    if not paths:
        return ()
    with paths[0].open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        return tuple(_summary_issue(paths[0], row) for row in reader if _summary_is_issue(row))


def _summary_is_issue(row: Mapping[str, str]) -> bool:
    return "完成" not in (row.get("status") or "")


def _summary_issue(path: Path, row: Mapping[str, str]) -> IssueRecord:
    subject_id = (row.get("participant_id") or "").strip()
    reason = (row.get("warnings") or row.get("status") or "summary 标记不可用").strip()
    return IssueRecord("situation_awareness", subject_id, "all", (path,), reason)


def _spot_subject_paths(eye_dir: Path) -> Dict[str, Dict[str, Path]]:
    grouped: Dict[str, Dict[str, Path]] = {}
    for topic in TOPIC_IDS:
        topic_dir = eye_dir / topic
        if not topic_dir.is_dir():
            raise FileNotFoundError(f"{topic_dir}: 找不同题目目录不存在")
        for path in sorted(topic_dir.glob("*.csv")):
            grouped.setdefault(subject_id_from_path(path), {})[topic] = path
    return grouped


def _spot_pairing_issues(
    subject_id: str,
    paths: Mapping[str, Path],
    behavior_dir: Path,
) -> Tuple[IssueRecord, ...]:
    issues = []
    for topic in TOPIC_IDS:
        if topic not in paths:
            issues.append(IssueRecord("spot_difference", subject_id, topic, (), "缺少眼动 CSV"))
            continue
        behavior_path = behavior_dir / topic / f"{subject_id}_{topic[1:]}_behavior.xlsx"
        if not behavior_path.is_file():
            issues.append(IssueRecord("spot_difference", subject_id, topic, (behavior_path,), "缺少行为 XLSX"))
    return tuple(issues)


def _require_topics(subject_id: str, task_paths: Mapping[str, Path]) -> None:
    missing = tuple(topic for topic in TOPIC_IDS if topic not in task_paths)
    if missing:
        raise ValueError(f"{subject_id}: 缺少找不同题目文件 {', '.join(missing)}")


def _screen_size_from_args(screen_args: Tuple[int, ...]) -> ScreenSize:
    if len(screen_args) != 2:
        raise TypeError("build_spot_subject_records 需要 screen_width 和 screen_height")
    return screen_args[0], screen_args[1]


def _ordered_topic_paths(paths: Mapping[str, Path]) -> Tuple[Tuple[str, Path], ...]:
    return tuple((topic, paths[topic]) for topic in TOPIC_IDS)


def _record_conditions(
    record: AnnotatedRecord,
    conditions_by_task: Mapping[str, Sequence[TaskConditions]],
) -> Sequence[TaskConditions]:
    task_key = record.sample.task_key
    if task_key not in conditions_by_task:
        raise KeyError(f"{record.sample.source_path}: 未找到任务条件 {task_key}")
    return conditions_by_task[task_key]


def _conditions_by_task(sa_dir: Path, spot_dir: Path) -> Dict[str, Tuple[TaskConditions, ...]]:
    return {
        "situation_awareness": parse_condition_combinations(sa_dir.name),
        "spot_difference": parse_condition_combinations(spot_dir.name),
    }


def _sa_sample(path: Path) -> InferenceSample:
    spec = TASK_SPECS["situation_awareness"]
    return InferenceSample(subject_id_from_path(path), spec.task_id, "情景意识", spec.task_conditions, (), path, task_key=spec.key)


def _spot_sample(subject_id: str, path: Path) -> InferenceSample:
    spec = TASK_SPECS["spot_difference"]
    return InferenceSample(subject_id, spec.task_id, "找不同", spec.task_conditions, (), path, task_key=spec.key)


def _source_paths(records: Iterable[AnnotatedRecord]) -> Tuple[Path, ...]:
    return tuple(path for record in records for path in record.source_paths)


def _condition_label(task_conditions: TaskConditions) -> str:
    return "-".join(str(value) for value in task_conditions)


def _unique(values: Iterable[str]) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))
