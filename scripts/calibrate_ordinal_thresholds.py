# -*- coding: utf-8 -*-
"""
基于验证集搜索序数阈值，并输出校准后的各分布指标。

输入:
- run 目录或 test_results.json

输出:
- 控制台打印最佳阈值与各分布前后对比
- 可选写入校准结果 JSON
- 可选回写配置文件中的 training.ordinal_thresholds
"""

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="序数阈值校准工具（基于 Val 集最大化 macro F1）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="run目录或 test_results.json 路径",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=0.2,
        help="阈值网格起点",
    )
    parser.add_argument(
        "--end",
        type=float,
        default=0.8,
        help="阈值网格终点",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.01,
        help="阈值网格步长",
    )
    parser.add_argument(
        "--update-config",
        type=str,
        default=None,
        help="可选：将最优阈值回写到该配置文件的 training.ordinal_thresholds",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="可选：输出校准报告 JSON 路径",
    )
    return parser.parse_args()


def resolve_results_path(path_like: str) -> Path:
    p = Path(path_like)
    if p.is_dir():
        p = p / "test_results.json"
    if not p.exists():
        raise FileNotFoundError(f"未找到结果文件: {p}")
    return p


def canonical_split_name(name: str) -> str:
    low = name.lower()
    if "val" in low or "同分布" in low:
        return "Val"
    if "test1" in low:
        return "Test1"
    if "test2" in low:
        return "Test2"
    if "test3" in low:
        return "Test3"
    return name


def class_probs_to_cum_probs(class_probs: np.ndarray) -> np.ndarray:
    """
    P(y=c) -> P(y>k), shape: (N, C) -> (N, C-1)
    """
    num_classes = class_probs.shape[1]
    if num_classes < 2:
        raise ValueError("类别数必须 >= 2")
    cum = []
    for k in range(num_classes - 1):
        cum.append(class_probs[:, k + 1 :].sum(axis=1))
    return np.stack(cum, axis=1)


def decode_from_cum(cum_probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return (cum_probs > thresholds.reshape(1, -1)).sum(axis=1).astype(np.int64)


def evaluate_split(labels: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
    }


def grid_search_thresholds(
    val_labels: np.ndarray,
    val_probs: np.ndarray,
    start: float,
    end: float,
    step: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    cum_probs = class_probs_to_cum_probs(val_probs)
    num_thresholds = cum_probs.shape[1]
    grid = np.arange(start, end + 1e-9, step)

    best_metrics = None
    best_thresholds = None
    for combo in product(grid, repeat=num_thresholds):
        thresholds = np.asarray(combo, dtype=np.float64)
        preds = decode_from_cum(cum_probs, thresholds)
        metrics = evaluate_split(val_labels, preds)
        if best_metrics is None:
            best_metrics = metrics
            best_thresholds = thresholds
            continue
        if metrics["f1_macro"] > best_metrics["f1_macro"]:
            best_metrics = metrics
            best_thresholds = thresholds
        elif metrics["f1_macro"] == best_metrics["f1_macro"] and metrics["accuracy"] > best_metrics["accuracy"]:
            best_metrics = metrics
            best_thresholds = thresholds

    return best_thresholds, best_metrics


def main() -> None:
    args = parse_args()
    results_path = resolve_results_path(args.results)
    payload = json.load(open(results_path, "r", encoding="utf-8"))

    # 规范化 split 名称
    split_map: Dict[str, Dict] = {}
    for raw_name, data in payload.items():
        split_map[canonical_split_name(raw_name)] = data

    if "Val" not in split_map:
        raise ValueError("结果中缺少 Val(同分布) 分布，无法校准阈值")

    val = split_map["Val"]
    val_labels = np.asarray(val["labels"], dtype=np.int64)
    val_probs = np.asarray(val["probabilities"], dtype=np.float64)

    thresholds, best_val_metrics = grid_search_thresholds(
        val_labels=val_labels,
        val_probs=val_probs,
        start=args.start,
        end=args.end,
        step=args.step,
    )

    report = {
        "results_path": str(results_path),
        "search_grid": {"start": args.start, "end": args.end, "step": args.step},
        "best_thresholds": [float(x) for x in thresholds.tolist()],
        "best_val_metrics": best_val_metrics,
        "splits": {},
    }

    print("最佳阈值:", report["best_thresholds"])
    print("Val 最佳指标:", best_val_metrics)

    for split_name in ["Val", "Test1", "Test2", "Test3"]:
        if split_name not in split_map:
            continue
        data = split_map[split_name]
        labels = np.asarray(data["labels"], dtype=np.int64)
        probs = np.asarray(data["probabilities"], dtype=np.float64)
        preds = decode_from_cum(class_probs_to_cum_probs(probs), thresholds)
        calibrated_metrics = evaluate_split(labels, preds)

        baseline_metrics = {
            "accuracy": float(data["accuracy"]),
            "f1_macro": float(data["f1_macro"]),
            "f1_weighted": float(data["f1_weighted"]),
        }
        delta_metrics = {
            k: float(calibrated_metrics[k] - baseline_metrics[k])
            for k in baseline_metrics
        }

        report["splits"][split_name] = {
            "baseline": baseline_metrics,
            "calibrated": calibrated_metrics,
            "delta": delta_metrics,
        }

        print(
            f"{split_name}: "
            f"F1_macro {baseline_metrics['f1_macro']:.4f} -> {calibrated_metrics['f1_macro']:.4f} "
            f"(Δ {delta_metrics['f1_macro']:+.4f}), "
            f"Acc {baseline_metrics['accuracy']:.4f} -> {calibrated_metrics['accuracy']:.4f} "
            f"(Δ {delta_metrics['accuracy']:+.4f})"
        )

    if args.update_config:
        cfg_path = Path(args.update_config)
        cfg = json.load(open(cfg_path, "r", encoding="utf-8"))
        cfg.setdefault("training", {})
        cfg["training"]["ordinal_thresholds"] = [float(x) for x in thresholds.tolist()]
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        print(f"已回写配置阈值: {cfg_path}")

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"已输出报告: {out}")


if __name__ == "__main__":
    main()
