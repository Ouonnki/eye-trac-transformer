# -*- coding: utf-8 -*-
"""End-to-end runner for unlabeled new-data inference."""

from collections import defaultdict
from itertools import product
import logging
from pathlib import Path
import re
from typing import Dict, List, Mapping, Sequence, Tuple

from src.inference.new_data_loader import (
    load_task_file,
    normalize_subject_samples,
)
from src.inference.new_data_model import (
    load_inference_model,
    predict_samples,
    resolve_device,
)
from src.inference.new_data_output import write_prediction_outputs
from src.inference.new_data_types import (
    InferenceSample,
    TASK_SPECS,
    TaskConditions,
    TaskFileSet,
    TaskSpec,
)


DEFAULT_CHECKPOINT_PATH = Path(
    "outputs/task_level_full_data/"
    "task_level_ordinal_manual11_bilstm_full_data_20260607_233158/"
    "best_model.pt"
)
DEFAULT_DATA_DIR = Path("新数据（40人）")
DEFAULT_OUTPUT_ROOT = Path("outputs/new_data_inference")
CONDITION_PATTERN = re.compile(r"[（(]([^（）()]*)[）)]")
CONDITION_DIMENSIONS = 5


logger = logging.getLogger(__name__)


def run_new_data_inference(
    data_dir: Path = DEFAULT_DATA_DIR,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    device_name: str = "auto",
) -> Path:
    device = resolve_device(device_name)
    model, config = load_inference_model(checkpoint_path, device)
    task_files = discover_task_files(data_dir)
    samples = load_new_data_samples(task_files, config)
    normalized = normalize_and_annotate_samples(samples)
    expanded = expand_samples_by_conditions(
        normalized,
        _conditions_by_task_key(task_files),
    )
    predictions = predict_samples(model, config, expanded, device)
    return write_prediction_outputs(predictions, output_root)


def discover_task_files(data_dir: Path) -> Tuple[TaskFileSet, ...]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"新数据目录不存在: {data_dir}")
    discovered = []
    for task_spec in TASK_SPECS.values():
        task_dir = _find_task_directory(data_dir, task_spec)
        eye_directory = _find_eye_directory(task_dir)
        task_paths = tuple(sorted(eye_directory.rglob("*.csv")))
        if not task_paths:
            raise FileNotFoundError(f"{eye_directory}: 未找到眼动 CSV")
        _validate_unique_subjects(task_paths, task_spec)
        discovered.append(
            TaskFileSet(
                spec=task_spec,
                task_dir=task_dir,
                eye_dir=eye_directory,
                conditions=parse_task_conditions(task_dir.name),
                paths=task_paths,
            )
        )
    _validate_complete_subject_sets(discovered)
    return tuple(discovered)


def load_new_data_samples(
    task_files: Sequence[TaskFileSet],
    config: Dict,
) -> Tuple[InferenceSample, ...]:
    sequence = config["sequence"]
    samples = []
    total = sum(len(task_set.paths) for task_set in task_files)
    for task_set in task_files:
        for path in task_set.paths:
            logger.info("读取眼动文件 %s/%s: %s", len(samples) + 1, total, path)
            samples.append(
                load_task_file(
                    path,
                    task_set.spec,
                    sequence["screen_width"],
                    sequence["screen_height"],
                )
            )
    return tuple(samples)


def parse_task_conditions(directory_name: str) -> Tuple[TaskConditions, ...]:
    match = CONDITION_PATTERN.search(directory_name)
    if match is None:
        raise ValueError(f"{directory_name}: 任务目录名缺少条件括号")
    parts = _condition_parts(directory_name, match.group(1))
    choices = tuple(_parse_condition_part(directory_name, part) for part in parts)
    return tuple(tuple(values) for values in product(*choices))


def expand_samples_by_conditions(
    samples: Sequence[InferenceSample],
    conditions_by_task_key: Mapping[str, Sequence[TaskConditions]],
) -> Tuple[InferenceSample, ...]:
    expanded = []
    for sample in samples:
        if sample.task_key not in conditions_by_task_key:
            raise KeyError(f"{sample.source_path}: 未找到任务条件: {sample.task_key}")
        for conditions in conditions_by_task_key[sample.task_key]:
            expanded.append(
                sample.with_task_conditions(conditions, _condition_label(conditions))
            )
    return tuple(expanded)


def normalize_and_annotate_samples(
    samples: Sequence[InferenceSample],
) -> Tuple[InferenceSample, ...]:
    by_subject: Dict[str, List[InferenceSample]] = defaultdict(list)
    for sample in samples:
        by_subject[sample.subject_id].append(sample)
    normalized_by_path = {}
    expected_tasks = {spec.task_name for spec in TASK_SPECS.values()}
    for subject_samples in by_subject.values():
        actual_tasks = {sample.task_name for sample in subject_samples}
        missing_tasks = tuple(sorted(expected_tasks - actual_tasks))
        for sample in normalize_subject_samples(subject_samples):
            normalized_by_path[sample.source_path] = sample.with_missing_tasks(
                missing_tasks
            )
    return tuple(normalized_by_path[sample.source_path] for sample in samples)


def _condition_parts(directory_name: str, expression: str) -> Tuple[str, ...]:
    normalized = expression.replace(",", "，")
    parts = tuple(part.strip() for part in normalized.split("，"))
    if len(parts) != CONDITION_DIMENSIONS:
        raise ValueError(f"{directory_name}: 任务条件必须为5维，实际{len(parts)}维")
    if any(not part for part in parts):
        raise ValueError(f"{directory_name}: 任务条件包含空值")
    return parts


def _parse_condition_part(directory_name: str, part: str) -> Tuple[int, ...]:
    values = []
    for raw_value in part.split(":"):
        value = raw_value.strip()
        if not value:
            raise ValueError(f"{directory_name}: 条件维度包含空值")
        try:
            values.append(int(value))
        except ValueError as error:
            raise ValueError(f"{directory_name}: 条件值不是整数: {value}") from error
    return tuple(values)


def _condition_label(task_conditions: TaskConditions) -> str:
    return "-".join(str(value) for value in task_conditions)


def _conditions_by_task_key(
    task_files: Sequence[TaskFileSet],
) -> Dict[str, Tuple[TaskConditions, ...]]:
    return {task_set.spec.key: task_set.conditions for task_set in task_files}


def _find_task_directory(data_dir: Path, task_spec: TaskSpec) -> Path:
    candidates = tuple(
        path
        for path in data_dir.iterdir()
        if path.is_dir() and path.name.startswith(task_spec.directory_prefix)
    )
    if len(candidates) != 1:
        raise ValueError(
            f"{data_dir}: {task_spec.task_name} 目录数量应为1，实际{len(candidates)}"
        )
    return candidates[0]


def _find_eye_directory(task_dir: Path) -> Path:
    if not task_dir.is_dir():
        raise FileNotFoundError(f"任务目录不存在: {task_dir}")
    candidates = tuple(
        path for path in task_dir.iterdir() if path.is_dir() and "眼动" in path.name
    )
    if len(candidates) != 1:
        raise ValueError(f"{task_dir}: 预期1个眼动目录，实际找到{len(candidates)}个")
    return candidates[0]


def _validate_unique_subjects(paths: Sequence[Path], task_spec: TaskSpec) -> None:
    subject_ids = [path.name.split("_", maxsplit=1)[0] for path in paths]
    if len(subject_ids) != len(set(subject_ids)):
        raise ValueError(f"{task_spec.directory_name}: 存在重复被试 CSV")


def _validate_complete_subject_sets(task_files: Sequence[TaskFileSet]) -> None:
    expected = set(_subject_ids(task_files[0].paths))
    for task_set in task_files[1:]:
        actual = set(_subject_ids(task_set.paths))
        if actual == expected:
            continue
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"{task_set.spec.task_name}: 被试集合不完整，缺失={missing}, 额外={extra}"
        )


def _subject_ids(paths: Sequence[Path]) -> Tuple[str, ...]:
    return tuple(path.name.split("_", maxsplit=1)[0] for path in paths)
