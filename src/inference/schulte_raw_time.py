# -*- coding: utf-8 -*-
"""Timestamp parsing and explicit repair for known Schulte raw anomalies."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Sequence, Tuple

import pandas as pd


DATA_ROW_OFFSET = 2
MICROSECONDS_PER_MILLISECOND = 1000
ONE_SECOND = timedelta(seconds=1)
TIME_PATTERNS = (
    re.compile(r"^(\d{4})/(\d{1,2})/(\d{1,2})\s+(\d{1,2}):(\d{2}):(\d{2}):(\d{1,3})$"),
    re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})\s+(\d{1,2}):(\d{2}):(\d{2})\.(\d{1,3})$"),
)


def parse_timestamp(
    value: Any,
    path: Path,
    sheet_name: str,
    *,
    row_number: int,
) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if pd.isna(value):
        raise ValueError(f"{path} {sheet_name} 第{row_number}行: 时间为空")
    text = str(value).strip()
    parsed = _parse_with_known_patterns(text)
    if parsed is not None:
        return parsed
    return _parse_with_pandas(text, path, sheet_name, row_number=row_number)


def repair_known_rollbacks(
    timestamps: Sequence[datetime],
    path: Path,
    sheet_name: str,
) -> Tuple[Tuple[datetime, ...], Tuple[str, ...], int]:
    repaired = []
    warnings = []
    for index, timestamp in enumerate(timestamps):
        if not repaired or timestamp >= repaired[-1]:
            repaired.append(timestamp)
            continue
        corrected = _same_second_zero_fix(repaired[-1], timestamp)
        _ensure_corrected(
            timestamp,
            corrected,
            timestamps,
            index=index,
            path=path,
            sheet_name=sheet_name,
        )
        warnings.append(
            _repair_warning(
                sheet_name,
                index,
                repaired[-1],
                original=timestamp,
                corrected=corrected,
            )
        )
        repaired.append(corrected)
    return tuple(repaired), tuple(warnings), len(warnings)


def _parse_with_known_patterns(text: str) -> datetime:
    for pattern in TIME_PATTERNS:
        match = pattern.match(text)
        if match is not None:
            return _datetime_from_match(match)
    return None


def _datetime_from_match(match: re.Match) -> datetime:
    groups = match.groups()
    milliseconds = int(groups[6].ljust(3, "0")[:3])
    return datetime(
        int(groups[0]),
        int(groups[1]),
        int(groups[2]),
        int(groups[3]),
        int(groups[4]),
        int(groups[5]),
        milliseconds * MICROSECONDS_PER_MILLISECOND,
    )


def _parse_with_pandas(
    text: str,
    path: Path,
    sheet_name: str,
    *,
    row_number: int,
) -> datetime:
    try:
        return pd.to_datetime(text).to_pydatetime().replace(tzinfo=None)
    except Exception as error:
        raise ValueError(f"{path} {sheet_name} 第{row_number}行: 时间无法解析: {text}") from error


def _same_second_zero_fix(previous: datetime, current: datetime) -> datetime:
    if current.microsecond != 0:
        return None
    same_second = previous.replace(microsecond=0) == current
    if not same_second or previous.microsecond == 0:
        return None
    return current + ONE_SECOND


def _ensure_corrected(
    original: datetime,
    corrected: datetime,
    timestamps: Sequence[datetime],
    *,
    index: int,
    path: Path,
    sheet_name: str,
) -> None:
    row_number = index + DATA_ROW_OFFSET
    if corrected is None:
        raise ValueError(f"{path} {sheet_name} 第{row_number}行: 时间回退无法自动修正: {original}")
    if index + 1 < len(timestamps) and corrected > timestamps[index + 1]:
        raise ValueError(f"{path} {sheet_name} 第{row_number}行: 修正后时间超过下一行")


def _repair_warning(
    sheet_name: str,
    index: int,
    previous: datetime,
    *,
    original: datetime,
    corrected: datetime,
) -> str:
    row_number = index + DATA_ROW_OFFSET
    return (
        f"{sheet_name} 第{row_number}行时间回退已修正: "
        f"{previous} -> {original} => {corrected}"
    )
