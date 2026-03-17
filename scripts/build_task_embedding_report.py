# -*- coding: utf-8 -*-
"""
构建任务嵌入消融/单模型/多模型对比报表（Excel）

输入:
- 消融模式：baseline（不带任务嵌入）结果 + baseline + task 结果
- 单模型模式：单个 test_results.json 或其所在目录
- 多模型模式：多个 test_results.json 或其所在目录（逗号分隔）

输出:
- 消融模式：每个分布 3~4 行（Baseline/Baseline + task/差值%，可选随机行）
- 单模型模式：每个分布 1 行（编码器 + 指标）
- 多模型模式：长表汇总（编码器 + 分布 + 指标）
"""

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import label_binarize

# 允许从仓库根目录导入 src/*
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.task_level_dataset import TaskLevelGazeDataset, TaskLevelSequenceConfig, task_level_collate_fn
from src.models.task_level_model import TaskLevelEncoder
from src.models.losses import decode_ordinal_logits, ordinal_probs_to_class_probs


METRIC_COLUMNS = [
    "ACC",
    "Precision Macro",
    "Precision Weighted",
    "Recall Macro",
    "Recall Weighted",
    "F1 Macro",
    "Spearman",
    "F1 Weight",
    "G-Mean",
    "MCC",
    "QWK",
    "AUC-PR",
]

SPLIT_ORDER = ["InDist", "Test1", "Test2", "Test3"]
SPLIT_LABELS = {
    "InDist": "同分布 (Val)",
    "Test1": "不同分布1 (Test1)",
    "Test2": "不同分布2 (Test2)",
    "Test3": "不同分布3 (Test3)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="构建任务嵌入消融/单模型/多模型报表",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--baseline-results",
        type=str,
        default=None,
        help="不带任务嵌入结果路径（test_results.json 或其所在目录）",
    )
    parser.add_argument(
        "--task-results",
        type=str,
        default=None,
        help="带任务嵌入结果路径（test_results.json 或其所在目录）",
    )
    parser.add_argument(
        "--single-results",
        type=str,
        default=None,
        help="单模型结果路径（test_results.json 或其所在目录）",
    )
    parser.add_argument(
        "--multi-results",
        type=str,
        default=None,
        help="多模型结果路径列表（逗号分隔，每个为 test_results.json 或其所在目录）",
    )
    parser.add_argument(
        "--encoder-name",
        type=str,
        default=None,
        help="单模型模式下覆盖编码器名称（默认从 config.json 推断）",
    )
    parser.add_argument(
        "--random-results",
        type=str,
        default=None,
        help="随机对照结果路径（random_baseline_results.json 或其所在目录）；不传则不写入随机对照行",
    )
    parser.add_argument(
        "--random-kind",
        type=str,
        default="uniform_random",
        help="随机对照类型（例如 uniform_random / stratified_random）",
    )
    parser.add_argument(
        "--output-xlsx",
        type=str,
        default=None,
        help="输出 Excel 路径；默认根据模式输出到 outputs/task_level/ 下",
    )
    return parser.parse_args()


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_results_path(path_like: str, expected_file: str) -> Path:
    path = Path(path_like)
    if path.is_dir():
        path = path / expected_file
    if not path.exists():
        raise FileNotFoundError(f"结果文件不存在: {path}")
    if not path.is_file():
        raise ValueError(f"结果路径不是文件: {path}")
    return path


def canonical_split_name(name: str) -> str:
    low = name.lower()
    if "val" in low or "同分布" in low:
        return "InDist"
    if "test1" in low:
        return "Test1"
    if "test2" in low:
        return "Test2"
    if "test3" in low:
        return "Test3"
    return name


def extract_split_results(payload: Dict[str, Any], random_kind: str = None) -> Dict[str, Dict[str, Any]]:
    source = payload
    if random_kind is not None and random_kind in payload and isinstance(payload[random_kind], dict):
        source = payload[random_kind]

    extracted: Dict[str, Dict[str, Any]] = {}
    for raw_name, metrics in source.items():
        if not isinstance(metrics, dict):
            continue
        if "labels" not in metrics or "predictions" not in metrics:
            continue
        split = canonical_split_name(raw_name)
        extracted[split] = metrics
    return extracted


def get_ordered_splits(
    baseline_data: Dict[str, Dict[str, Any]],
    task_data: Dict[str, Dict[str, Any]],
) -> List[str]:
    shared = set(baseline_data.keys()) & set(task_data.keys())
    if not shared:
        raise ValueError("Baseline 与 Baseline + task 没有可对齐的分布")

    preferred = [split for split in SPLIT_ORDER if split in shared]
    extras = sorted(split for split in shared if split not in SPLIT_ORDER)
    return preferred + extras


def get_ordered_splits_single(data: Dict[str, Dict[str, Any]]) -> List[str]:
    splits = list(data.keys())
    preferred = [split for split in SPLIT_ORDER if split in splits]
    extras = sorted(split for split in splits if split not in SPLIT_ORDER)
    return preferred + extras


def load_run_config(run_dir: Path) -> Optional[Dict[str, Any]]:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_encoder_name(run_dir: Path, config: Optional[Dict[str, Any]] = None) -> str:
    if config and "model" in config and "segment_encoder_type" in config["model"]:
        return str(config["model"]["segment_encoder_type"])
    return run_dir.name


class _Subset:
    """轻量 Subset，避免依赖训练脚本。"""

    def __init__(self, dataset, indices: List[int]):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx: int):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def split_2x2_indices(
    dataset: TaskLevelGazeDataset,
    train_subjects: int = 100,
    train_tasks: int = 20,
    train_val_split: float = 0.9,
    seed: int = 42,
) -> List[List[int]]:
    """与训练脚本一致的 2x2 划分，返回 train/val/test1/test2/test3 索引。"""
    subject_ids = sorted(list(set([s["subject_id"] for s in dataset.samples])))
    train_subject_ids = subject_ids[:train_subjects]
    test_subject_ids = subject_ids[train_subjects:]

    train_subject_set = set(train_subject_ids)
    test_subject_set = set(test_subject_ids)
    train_task_ids = set(range(1, train_tasks + 1))
    test_task_ids = set(range(train_tasks + 1, 31))

    train_pool: List[int] = []
    test1: List[int] = []
    test2: List[int] = []
    test3: List[int] = []

    for i, sample in enumerate(dataset.samples):
        sid = sample["subject_id"]
        tid = sample["task_id"]
        if sid in train_subject_set and tid in train_task_ids:
            train_pool.append(i)
        elif sid in test_subject_set and tid in train_task_ids:
            test1.append(i)
        elif sid in train_subject_set and tid in test_task_ids:
            test2.append(i)
        elif sid in test_subject_set and tid in test_task_ids:
            test3.append(i)

    rng = np.random.RandomState(seed)
    train_pool_subjects = sorted(list(set([dataset.samples[i]["subject_id"] for i in train_pool])))
    rng.shuffle(train_pool_subjects)

    n_train = int(len(train_pool_subjects) * train_val_split)
    train_subject_set = set(train_pool_subjects[:n_train])
    val_subject_set = set(train_pool_subjects[n_train:])

    train_indices = [i for i in train_pool if dataset.samples[i]["subject_id"] in train_subject_set]
    val_indices = [i for i in train_pool if dataset.samples[i]["subject_id"] in val_subject_set]
    return [train_indices, val_indices, test1, test2, test3]


def infer_run_dir(path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_dir() else path.parent


def recover_indist_metrics_from_run_dir(run_dir: Path) -> Dict[str, Any]:
    """
    从 run 目录（包含 best_model.pt/config.json）回算 Val(同分布) 的 labels/predictions/probabilities。
    """
    import torch
    from scipy.stats import spearmanr
    from torch.utils.data import DataLoader

    config_path = run_dir / "config.json"
    checkpoint_path = run_dir / "best_model.pt"
    if not config_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError(f"{run_dir} 缺少 config.json 或 best_model.pt，无法回算 Val")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    data_path = Path(config["data"]["processed_data_path"])
    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    with open(data_path, "rb") as f:
        processed_data = pickle.load(f)

    seq_config = TaskLevelSequenceConfig(
        max_seq_len=config["sequence"]["max_seq_len"],
        max_segments=config["sequence"]["max_segments"],
    )
    dataset = TaskLevelGazeDataset(
        processed_data=processed_data,
        config=seq_config,
        fit_normalizer=True,
    )

    _, val_indices, _, _, _ = split_2x2_indices(
        dataset=dataset,
        train_subjects=config["experiment"]["train_subjects"],
        train_tasks=config["experiment"]["train_tasks"],
        train_val_split=config["training"]["train_val_split"],
        seed=config["experiment"]["random_seed"],
    )
    val_dataset = _Subset(dataset, val_indices)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        collate_fn=task_level_collate_fn,
        num_workers=0,
        pin_memory=False,
    )

    if "segment_encoder_type" not in config["model"]:
        raise ValueError("config.json 缺少 model.segment_encoder_type，无法回算 Val")
    use_task_embedding = config["model"].get("use_task_embedding", True)
    head_type = config["model"].get("head_type", "classification")
    segment_encoder_type = config["model"]["segment_encoder_type"]
    segment_rnn_hidden_size = config["model"].get("segment_rnn_hidden_size", config["model"]["segment_d_model"])
    segment_rnn_layers = config["model"].get("segment_rnn_layers", 1)
    segment_rnn_dropout = config["model"].get("segment_rnn_dropout", config["model"].get("dropout", 0.0))
    segment_cnn_channels = config["model"].get("segment_cnn_channels", None)
    segment_cnn_kernel_sizes = config["model"].get("segment_cnn_kernel_sizes", None)

    model = TaskLevelEncoder(
        input_dim=config["sequence"]["input_dim"],
        max_seq_len=config["sequence"]["max_seq_len"],
        max_segments=config["sequence"]["max_segments"],
        segment_d_model=config["model"]["segment_d_model"],
        segment_nhead=config["model"]["segment_nhead"],
        segment_num_layers=config["model"]["segment_num_layers"],
        segment_encoder_type=segment_encoder_type,
        segment_rnn_hidden_size=segment_rnn_hidden_size,
        segment_rnn_layers=segment_rnn_layers,
        segment_rnn_dropout=segment_rnn_dropout,
        segment_cnn_channels=segment_cnn_channels,
        segment_cnn_kernel_sizes=segment_cnn_kernel_sizes,
        attention_dim=config["model"]["attention_dim"],
        task_embedding_dim=config["model"]["task_embedding_dim"],
        use_task_embedding=use_task_embedding,
        task_embedding_type=config["model"].get("task_embedding_type", "independent"),
        use_conditional_pooling=config["model"].get("use_conditional_pooling", False),
        dropout=config["model"]["dropout"],
        num_classes=config["model"]["num_classes"],
        head_type=head_type,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    all_preds: List[int] = []
    all_labels: List[int] = []
    all_probs: List[List[float]] = []

    with torch.no_grad():
        for batch in val_loader:
            segments = batch["segments"].to(device)
            segment_mask = batch["segment_mask"].to(device)
            task_conditions = batch["task_conditions"].to(device)
            labels = batch["labels"].to(device)

            segment_seq_mask = batch.get("segment_seq_mask")
            if segment_seq_mask is not None:
                segment_seq_mask = segment_seq_mask.to(device)
            logits = model(segments, segment_mask, task_conditions, segment_seq_mask)
            if head_type == "ordinal":
                thresholds = config["training"].get("ordinal_thresholds", None)
                if thresholds is None:
                    thresholds = [0.5] * (config["model"]["num_classes"] - 1)
                thresholds_tensor = torch.tensor(thresholds, dtype=logits.dtype, device=logits.device)
                preds = decode_ordinal_logits(logits, thresholds_tensor)
                cum_probs = torch.sigmoid(logits)
                probs = ordinal_probs_to_class_probs(cum_probs)
            else:
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

            all_labels.extend(labels.cpu().numpy().astype(int).tolist())
            all_preds.extend(preds.cpu().numpy().astype(int).tolist())
            all_probs.extend(probs.cpu().numpy().astype(float).tolist())

    labels_np = np.asarray(all_labels, dtype=int)
    preds_np = np.asarray(all_preds, dtype=int)
    spearman_corr, spearman_p = spearmanr(labels_np, preds_np)
    if spearman_corr is None or np.isnan(spearman_corr):
        spearman_corr = 0.0
    if spearman_p is None or np.isnan(spearman_p):
        spearman_p = 1.0

    return {
        "loss": 0.0,
        "accuracy": float(accuracy_score(labels_np, preds_np)),
        "f1_weighted": float(f1_score(labels_np, preds_np, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(labels_np, preds_np, average="macro", zero_division=0)),
        "spearman": float(spearman_corr),
        "spearman_p": float(spearman_p),
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs,
    }


def make_probabilities(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: Any,
    split_name: str,
) -> np.ndarray:
    num_classes = int(max(np.max(labels), np.max(predictions))) + 1
    if num_classes <= 1:
        num_classes = 2

    probs = None
    if probabilities is not None:
        arr = np.asarray(probabilities, dtype=float)
        if arr.ndim == 2 and arr.shape[0] == len(labels):
            probs = arr

    if probs is None:
        print(f"[WARN] {split_name}: 缺少有效 probabilities，回退为 one-hot 预测分数")
        probs = np.zeros((len(predictions), num_classes), dtype=float)
        probs[np.arange(len(predictions)), predictions.astype(int)] = 1.0
        return probs

    if probs.shape[1] < num_classes:
        pad = np.zeros((probs.shape[0], num_classes - probs.shape[1]), dtype=float)
        probs = np.concatenate([probs, pad], axis=1)
    elif probs.shape[1] > num_classes:
        probs = probs[:, :num_classes]

    row_sum = probs.sum(axis=1, keepdims=True)
    bad = row_sum.squeeze(-1) <= 0
    if np.any(bad):
        probs[bad] = 1.0 / probs.shape[1]
        row_sum = probs.sum(axis=1, keepdims=True)
    probs = probs / row_sum
    return probs


def compute_gmean(labels: np.ndarray, preds: np.ndarray) -> float:
    classes = np.unique(labels)
    recalls = recall_score(labels, preds, labels=classes, average=None, zero_division=0)
    if recalls.size == 0:
        return 0.0
    if np.any(recalls <= 0):
        return 0.0
    return float(np.exp(np.mean(np.log(recalls))))


def safe_spearman(labels: np.ndarray, preds: np.ndarray) -> float:
    corr, _ = spearmanr(labels, preds)
    if corr is None or np.isnan(corr):
        return 0.0
    return float(corr)


def safe_auc_pr(labels: np.ndarray, probs: np.ndarray) -> float:
    num_classes = probs.shape[1]
    classes = list(range(num_classes))
    y_true_bin = label_binarize(labels, classes=classes)
    if y_true_bin.ndim == 1:
        y_true_bin = y_true_bin.reshape(-1, 1)
    try:
        value = average_precision_score(y_true_bin, probs, average="macro")
    except ValueError:
        value = 0.0
    if value is None or np.isnan(value):
        return 0.0
    return float(value)


def compute_metrics(metrics: Dict[str, Any], split_name: str) -> Dict[str, float]:
    labels = np.asarray(metrics["labels"], dtype=int)
    preds = np.asarray(metrics["predictions"], dtype=int)
    probs = make_probabilities(labels, preds, metrics.get("probabilities"), split_name)

    return {
        "ACC": float(accuracy_score(labels, preds)),
        "Precision Macro": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "Precision Weighted": float(precision_score(labels, preds, average="weighted", zero_division=0)),
        "Recall Macro": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "Recall Weighted": float(recall_score(labels, preds, average="weighted", zero_division=0)),
        "F1 Macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "Spearman": safe_spearman(labels, preds),
        "F1 Weight": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "G-Mean": compute_gmean(labels, preds),
        "MCC": float(matthews_corrcoef(labels, preds)),
        "QWK": float(cohen_kappa_score(labels, preds, weights="quadratic")),
        "AUC-PR": safe_auc_pr(labels, probs),
    }


def relative_change_percent(new: float, base: float) -> str:
    if abs(base) < 1e-12:
        if abs(new) < 1e-12:
            return "0.00%"
        return "N/A"
    value = (new - base) / base * 100.0
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}%"


def build_rows(
    baseline_data: Dict[str, Dict[str, Any]],
    task_data: Dict[str, Dict[str, Any]],
    random_data: Optional[Dict[str, Dict[str, Any]]] = None,
    include_random: bool = True,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for split in get_ordered_splits(baseline_data, task_data):
        baseline_metrics = compute_metrics(baseline_data[split], f"{split}/baseline")
        task_metrics = compute_metrics(task_data[split], f"{split}/task")

        random_metrics = None
        if include_random and random_data is not None:
            if split in random_data:
                random_metrics = compute_metrics(random_data[split], f"{split}/random")
            else:
                print(f"[WARN] 随机结果缺少分布 {split}，该分布跳过随机行")

        split_label = SPLIT_LABELS.get(split, split)

        row_baseline = {"分布": split_label, "方法": "Baseline"}
        row_task = {"分布": split_label, "方法": "Baseline + task"}
        row_delta = {"分布": split_label, "方法": "差值/比例"}

        for col in METRIC_COLUMNS:
            row_baseline[col] = round(baseline_metrics[col], 6)
            row_task[col] = round(task_metrics[col], 6)
            row_delta[col] = relative_change_percent(task_metrics[col], baseline_metrics[col])

        if random_metrics is not None:
            row_random = {"分布": split_label, "方法": "随机"}
            for col in METRIC_COLUMNS:
                row_random[col] = round(random_metrics[col], 6)
            rows.extend([row_random, row_baseline, row_task, row_delta])
        else:
            rows.extend([row_baseline, row_task, row_delta])

    return rows


def build_single_rows(
    data: Dict[str, Dict[str, Any]],
    encoder_name: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for split in get_ordered_splits_single(data):
        metrics = compute_metrics(data[split], f"{split}/{encoder_name}")
        row = {"分布": SPLIT_LABELS.get(split, split), "编码器": encoder_name}
        for col in METRIC_COLUMNS:
            row[col] = round(metrics[col], 6)
        rows.append(row)
    return rows


def build_multi_rows(
    all_data: List[Dict[str, Dict[str, Any]]],
    encoder_names: List[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for data, name in zip(all_data, encoder_names):
        rows.extend(build_single_rows(data, name))
    return rows


def main() -> None:
    args = parse_args()

    if args.single_results or args.multi_results:
        if args.baseline_results or args.task_results:
            raise ValueError("单模型/多模型模式下不允许传 --baseline-results 或 --task-results")
        if args.single_results and args.multi_results:
            raise ValueError("单模型与多模型模式不能同时使用")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if args.single_results:
            results_path = resolve_results_path(args.single_results, "test_results.json")
            run_dir = infer_run_dir(args.single_results)
            payload = load_json(str(results_path))
            data = extract_split_results(payload)
            if "InDist" not in data:
                try:
                    data["InDist"] = recover_indist_metrics_from_run_dir(run_dir)
                    print(f"[INFO] 单模型 Val 回算完成: n={len(data['InDist']['labels'])}")
                except Exception as exc:
                    print(f"[WARN] 单模型 Val 回算失败: {exc}")

            config = load_run_config(run_dir)
            encoder_name = args.encoder_name or infer_encoder_name(run_dir, config)
            rows = build_single_rows(data, encoder_name)
            df = pd.DataFrame(rows)

            if args.output_xlsx is None:
                output_path = Path("outputs/task_level") / f"single_model_report_{encoder_name}_{timestamp}.xlsx"
            else:
                output_path = Path(args.output_xlsx)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="single_model_report")
            print(f"[INFO] 单模型报表已生成: {output_path}")
            return

        results_list = [p.strip() for p in args.multi_results.split(",") if p.strip()]
        if not results_list:
            raise ValueError("--multi-results 为空")
        all_data: List[Dict[str, Dict[str, Any]]] = []
        encoder_names: List[str] = []
        for path_like in results_list:
            results_path = resolve_results_path(path_like, "test_results.json")
            run_dir = infer_run_dir(path_like)
            payload = load_json(str(results_path))
            data = extract_split_results(payload)
            if "InDist" not in data:
                try:
                    data["InDist"] = recover_indist_metrics_from_run_dir(run_dir)
                    print(f"[INFO] 多模型 Val 回算完成: {run_dir}")
                except Exception as exc:
                    print(f"[WARN] 多模型 Val 回算失败: {run_dir} ({exc})")
            config = load_run_config(run_dir)
            encoder_names.append(infer_encoder_name(run_dir, config))
            all_data.append(data)

        rows = build_multi_rows(all_data, encoder_names)
        df = pd.DataFrame(rows)
        if args.output_xlsx is None:
            output_path = Path("outputs/task_level") / f"encoder_comparison_{timestamp}.xlsx"
        else:
            output_path = Path(args.output_xlsx)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="encoder_comparison")
        print(f"[INFO] 多模型对比报表已生成: {output_path}")
        return

    if not args.baseline_results or not args.task_results:
        raise ValueError("消融模式需要同时提供 --baseline-results 与 --task-results")

    baseline_path = resolve_results_path(args.baseline_results, "test_results.json")
    task_path = resolve_results_path(args.task_results, "test_results.json")
    baseline_run_dir = infer_run_dir(args.baseline_results)
    task_run_dir = infer_run_dir(args.task_results)

    random_path: Optional[Path] = None
    if args.random_results:
        random_path = resolve_results_path(args.random_results, "random_baseline_results.json")
    else:
        print("[INFO] 未传 --random-results，将生成不含随机对照的报表")

    baseline_payload = load_json(str(baseline_path))
    task_payload = load_json(str(task_path))

    baseline_data = extract_split_results(baseline_payload)
    task_data = extract_split_results(task_payload)

    random_data = None
    if random_path is not None:
        random_payload = load_json(str(random_path))
        random_data = extract_split_results(random_payload, random_kind=args.random_kind)

    should_have_indist = False
    if "InDist" in baseline_data or "InDist" in task_data:
        should_have_indist = True
    if random_data is not None and "InDist" in random_data:
        should_have_indist = True

    if should_have_indist and "InDist" not in baseline_data:
        print(f"[INFO] Baseline 缺少 Val，尝试从 {baseline_run_dir} 回算")
        baseline_data["InDist"] = recover_indist_metrics_from_run_dir(baseline_run_dir)
        print(f"[INFO] Baseline Val 回算完成: n={len(baseline_data['InDist']['labels'])}")

    if should_have_indist and "InDist" not in task_data:
        print(f"[INFO] Baseline + task 缺少 Val，尝试从 {task_run_dir} 回算")
        task_data["InDist"] = recover_indist_metrics_from_run_dir(task_run_dir)
        print(f"[INFO] Baseline + task Val 回算完成: n={len(task_data['InDist']['labels'])}")

    split_names = get_ordered_splits(baseline_data, task_data)
    print(f"[INFO] 将生成分布: {', '.join(split_names)}")

    rows = build_rows(
        baseline_data=baseline_data,
        task_data=task_data,
        random_data=random_data,
        include_random=(random_data is not None),
    )
    columns = ["分布", "方法"] + METRIC_COLUMNS
    df = pd.DataFrame(rows, columns=columns)

    if args.output_xlsx is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path("outputs/task_level") / f"task_embedding_comparison_{timestamp}.xlsx"
    else:
        output_path = Path(args.output_xlsx)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="task_embedding_ablation")

    print(f"[OK] 报表已生成: {output_path}")


if __name__ == "__main__":
    main()
