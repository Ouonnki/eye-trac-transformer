# -*- coding: utf-8 -*-
"""End-to-end runner for unlabeled new-data inference."""

from collections import defaultdict
import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
from src.inference.new_data_types import InferenceSample, TASK_SPECS, TaskSpec


DEFAULT_CHECKPOINT_PATH = Path(
    "outputs/task_level_full_data/"
    "task_level_ordinal_manual11_bilstm_full_data_20260607_233158/"
    "best_model.pt"
)
DEFAULT_DATA_DIR = Path("新数据")
DEFAULT_OUTPUT_ROOT = Path("outputs/new_data_inference")


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
    predictions = predict_samples(model, config, normalized, device)
    return write_prediction_outputs(predictions, data_dir, output_root)


def discover_task_files(data_dir: Path) -> Tuple[Tuple[TaskSpec, Path], ...]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"新数据目录不存在: {data_dir}")
    discovered = []
    for task_spec in TASK_SPECS.values():
        task_dir = data_dir / task_spec.directory_name
        eye_directory = _find_eye_directory(task_dir)
        task_paths = tuple(sorted(eye_directory.rglob("*.csv")))
        if not task_paths:
            raise FileNotFoundError(f"{eye_directory}: 未找到眼动 CSV")
        _validate_unique_subjects(task_paths, task_spec)
        discovered.extend((task_spec, path) for path in task_paths)
    return tuple(discovered)


def load_new_data_samples(
    task_files: Sequence[Tuple[TaskSpec, Path]],
    config: Dict,
) -> Tuple[InferenceSample, ...]:
    sequence = config["sequence"]
    samples = []
    total = len(task_files)
    for index, (task_spec, path) in enumerate(task_files, start=1):
        logger.info("读取眼动文件 %s/%s: %s", index, total, path)
        samples.append(
            load_task_file(
                path,
                task_spec,
                sequence["screen_width"],
                sequence["screen_height"],
            )
        )
    return tuple(samples)


def normalize_and_annotate_samples(
    samples: Sequence[InferenceSample],
) -> Tuple[InferenceSample, ...]:
    by_subject: Dict[str, List[InferenceSample]] = defaultdict(list)
    for sample in samples:
        by_subject[sample.subject_id].append(sample)
    normalized_by_path = {}
    expected_tasks = {spec.directory_name.split("（", 1)[0] for spec in TASK_SPECS.values()}
    for subject_samples in by_subject.values():
        actual_tasks = {sample.task_name for sample in subject_samples}
        missing_tasks = tuple(sorted(expected_tasks - actual_tasks))
        for sample in normalize_subject_samples(subject_samples):
            normalized_by_path[sample.source_path] = sample.with_missing_tasks(missing_tasks)
    return tuple(normalized_by_path[sample.source_path] for sample in samples)


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
