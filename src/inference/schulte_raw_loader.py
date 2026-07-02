# -*- coding: utf-8 -*-
"""Validation and sample building for Schulte raw data folders."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Sequence, Tuple

from src.inference.new_data_features import normalize_subject_samples
from src.inference.new_data_types import InferenceSample
from src.inference.schulte_raw_features import SampleBuildInput, build_inference_sample
from src.inference.schulte_raw_question import TASK_IDS, load_schulte_question_info
from src.inference.schulte_raw_types import (
    SchulteBuild,
    SchulteIssue,
    SchulteQuestionInfo,
    SchulteRecord,
)
from src.inference.schulte_raw_xlsx import (
    actual_error_count,
    grid_layout,
    parse_click_points,
    parse_gaze_points,
    read_level_stats,
    read_task_workbook,
)


DEFAULT_SCREEN_WIDTH = 1920
DEFAULT_SCREEN_HEIGHT = 1080
READY_STATUS = "ready"
INVALID_STATUS = "invalid"
MISSING_STATUS = "missing"


@dataclass(frozen=True)
class _RecordInput:
    subject_id: str
    source_path: Path
    question: SchulteQuestionInfo
    screen_size: Tuple[int, int]


def build_schulte_records(
    data_dir: Path,
    question_info_path: Path,
    *,
    screen_width: int = DEFAULT_SCREEN_WIDTH,
    screen_height: int = DEFAULT_SCREEN_HEIGHT,
    task_ids: Sequence[int] = TASK_IDS,
) -> SchulteBuild:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"舒尔特原始数据目录不存在: {data_dir}")
    questions = load_schulte_question_info(question_info_path)
    screen_size = (screen_width, screen_height)
    records, issues = _build_all_records(
        data_dir,
        questions,
        tuple(task_ids),
        screen_size=screen_size,
    )
    return SchulteBuild(_normalize_ready_records(tuple(records)), tuple(issues))


def _build_all_records(
    data_dir: Path,
    questions: Mapping[int, SchulteQuestionInfo],
    task_ids: Tuple[int, ...],
    *,
    screen_size: Tuple[int, int],
) -> Tuple[List[SchulteRecord], List[SchulteIssue]]:
    subjects = _subject_dirs(data_dir)
    records: List[SchulteRecord] = []
    issues: List[SchulteIssue] = []
    for subject_dir in subjects:
        for task_id in task_ids:
            record, issue = _build_record_or_issue(subject_dir, questions[task_id], screen_size)
            records.append(record)
            if issue is not None:
                issues.append(issue)
    return records, issues


def _build_record_or_issue(
    subject_dir: Path,
    question: SchulteQuestionInfo,
    screen_size: Tuple[int, int],
) -> Tuple[SchulteRecord, SchulteIssue]:
    source_path = subject_dir / f"{question.task_id}.xlsx"
    record_input = _RecordInput(subject_dir.name, source_path, question, screen_size)
    if not source_path.is_file():
        reason = f"缺少题目文件: {source_path}"
        return _empty_record(record_input, MISSING_STATUS, (reason,)), _issue(record_input, reason)
    try:
        return _ready_record(record_input), None
    except Exception as error:
        reason = str(error)
        return _empty_record(record_input, INVALID_STATUS, (reason,)), _issue(record_input, reason)


def _ready_record(record_input: _RecordInput) -> SchulteRecord:
    frames = read_task_workbook(record_input.source_path)
    level = read_level_stats(frames.level, record_input.source_path)
    clicks = parse_click_points(frames.operation, record_input.source_path, record_input.screen_size)
    gaze = parse_gaze_points(frames.gaze, record_input.source_path, record_input.screen_size)
    actual_errors = actual_error_count(frames.operation)
    warnings = _warnings(level.reported_error_count, actual_errors, clicks.warnings + gaze.warnings)
    sample = build_inference_sample(
        SampleBuildInput(
            subject_id=record_input.subject_id,
            source_path=record_input.source_path,
            question=record_input.question,
            clicks=clicks.points,
            gaze_points=gaze.points,
            grid_layout=grid_layout(frames.grid, record_input.source_path),
            warnings=warnings,
            screen_size=record_input.screen_size,
        )
    )
    return _record(record_input, level, actual_errors, clicks=clicks, gaze=gaze, sample=sample)


def _record(record_input, level, actual_errors, *, clicks, gaze, sample) -> SchulteRecord:
    return SchulteRecord(
        subject_id=record_input.subject_id,
        task_id=record_input.question.task_id,
        source_path=record_input.source_path,
        status=READY_STATUS,
        question=record_input.question,
        sample=sample,
        raw_duration=level.duration,
        reported_error_count=level.reported_error_count,
        actual_error_count=actual_errors,
        click_count=len(clicks.points),
        gaze_count=len(gaze.points),
        segment_count=len(sample.segments),
        timestamp_fix_count=clicks.timestamp_fix_count + gaze.timestamp_fix_count,
        clipped_coordinate_count=clicks.clipped_coordinate_count + gaze.clipped_coordinate_count,
        warnings=sample.warnings,
    )


def _empty_record(
    record_input: _RecordInput,
    status: str,
    warnings: Tuple[str, ...],
) -> SchulteRecord:
    return SchulteRecord(
        subject_id=record_input.subject_id,
        task_id=record_input.question.task_id,
        source_path=record_input.source_path,
        status=status,
        question=record_input.question,
        warnings=warnings,
    )


def _normalize_ready_records(records: Tuple[SchulteRecord, ...]) -> Tuple[SchulteRecord, ...]:
    normalized = {}
    for subject_id, samples in _ready_samples_by_subject(records).items():
        for sample in normalize_subject_samples(samples):
            normalized[(subject_id, sample.source_path)] = sample
    return tuple(_with_normalized_sample(record, normalized) for record in records)


def _ready_samples_by_subject(records: Sequence[SchulteRecord]):
    grouped = defaultdict(list)
    for record in records:
        if record.status == READY_STATUS and record.sample is not None:
            grouped[record.subject_id].append(record.sample)
    return grouped


def _with_normalized_sample(
    record: SchulteRecord,
    normalized: Mapping[Tuple[str, Path], InferenceSample],
) -> SchulteRecord:
    key = (record.subject_id, record.source_path)
    if key not in normalized:
        return record
    return record.with_sample(normalized[key])


def _warnings(
    reported_error_count: int,
    actual_errors: int,
    generated: Tuple[str, ...],
) -> Tuple[str, ...]:
    values = list(generated)
    if reported_error_count != actual_errors:
        values.append(f"错误次数不一致: 关卡信息={reported_error_count}, 操作表={actual_errors}")
    return tuple(values)


def _issue(record_input: _RecordInput, reason: str) -> SchulteIssue:
    return SchulteIssue(
        subject_id=record_input.subject_id,
        task_id=record_input.question.task_id,
        source_path=record_input.source_path,
        reason=reason,
    )


def _subject_dirs(data_dir: Path) -> Tuple[Path, ...]:
    subjects = tuple(sorted(path for path in data_dir.iterdir() if path.is_dir() and path.name.isdigit()))
    if not subjects:
        raise FileNotFoundError(f"{data_dir}: 未找到被试目录")
    return subjects
