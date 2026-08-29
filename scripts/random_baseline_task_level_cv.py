# -*- coding: utf-8 -*-
"""基于任务级十折产出的均匀随机对照。

读取每折 test_results.json 的真实标签；对每个 split 均匀随机预测类别
(0/1/2，各 1/3)，使用固定 seed 保证可复现。指标口径与
build_task_embedding_report.py 一致：ACC、Macro-F1、G-Mean、AUC-PR、QWK。
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    f1_score,
    recall_score,
)
from sklearn.preprocessing import label_binarize


SPLIT_NAMES = [
    'Val (同分布)',
    'Test1 (新被试+旧题)',
    'Test2 (旧被试+新题)',
    'Test3 (新被试+新题)',
]
METRIC_NAMES = ('accuracy', 'f1_macro', 'g_mean', 'auc_pr', 'qwk')


def parse_args():
    parser = argparse.ArgumentParser(description='任务级十折均匀随机对照')
    parser.add_argument('--cv-dir', type=Path, required=True, help='已有十折输出目录')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-classes', type=int, default=3)
    return parser.parse_args()


def compute_gmean(labels, predictions):
    classes = np.unique(labels)
    recalls = recall_score(
        labels, predictions, labels=classes, average=None, zero_division=0
    )
    if recalls.size == 0 or np.any(recalls <= 0):
        return 0.0
    return float(np.exp(np.mean(np.log(recalls))))


def safe_auc_pr(labels, probabilities):
    num_classes = probabilities.shape[1]
    y_true_bin = label_binarize(labels, classes=list(range(num_classes)))
    if y_true_bin.ndim == 1:
        y_true_bin = y_true_bin.reshape(-1, 1)
    try:
        value = average_precision_score(y_true_bin, probabilities, average='macro')
    except ValueError:
        value = 0.0
    if value is None or np.isnan(value):
        return 0.0
    return float(value)


def compute_metrics(labels, predictions, probabilities):
    qwk = cohen_kappa_score(labels, predictions, weights='quadratic')
    return {
        'accuracy': float(accuracy_score(labels, predictions)),
        'f1_macro': float(f1_score(labels, predictions, average='macro', zero_division=0)),
        'g_mean': compute_gmean(labels, predictions),
        'auc_pr': safe_auc_pr(labels, probabilities),
        'qwk': 0.0 if qwk is None or np.isnan(qwk) else float(qwk),
    }


def aggregate(folds):
    summary = {}
    for split in SPLIT_NAMES:
        summary[split] = {}
        for metric in METRIC_NAMES:
            values = [float(f['splits'][split][metric]) for f in folds]
            summary[split][metric] = {
                'values': values,
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
            }
    return summary


def write_table(path, summary):
    lines = [
        '# Uniform random baseline (10-fold)',
        '',
        '| Split | ACC | Macro-F1 | G-Mean | AUC-PR | QWK |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for split in SPLIT_NAMES:
        s = summary[split]
        cells = [
            f"{s[m]['mean']:.4f} ± {s[m]['std']:.4f}"
            for m in METRIC_NAMES
        ]
        lines.append(f"| {split} | " + ' | '.join(cells) + ' |')
    lines.extend([
        '',
        'Random strategy: uniform class sampling, P(class)=1/3, seed=42.',
        'AUC-PR uses one-hot random class scores and macro one-vs-rest average precision.',
    ])
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return '\n'.join(lines)


def write_comparison_table(path, model_summary, random_summary):
    lines = [
        '# Distribution-shift performance with random control',
        '',
        '| Method | Split | ACC | Macro-F1 | G-Mean | AUC-PR | QWK |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for method, summary in (
        ('Ordinal BiLSTM', model_summary),
        ('Random', random_summary),
    ):
        for split in SPLIT_NAMES:
            s = summary[split]
            cells = [
                f"{s[m]['mean']:.4f} ± {s[m]['std']:.4f}"
                for m in METRIC_NAMES
            ]
            lines.append(f"| {method} | {split} | " + ' | '.join(cells) + ' |')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    folds = []

    for fold_dir in sorted(args.cv_dir.glob('fold_*')):
        results_path = fold_dir / 'test_results.json'
        if not results_path.exists():
            continue
        with results_path.open() as f:
            test_results = json.load(f)
        fold_result = {
            'fold': int(fold_dir.name.split('_')[1]),
            'splits': {},
        }
        for split in SPLIT_NAMES:
            labels = np.asarray(test_results[split]['labels'], dtype=int)
            predictions = rng.integers(0, args.num_classes, size=len(labels))
            probabilities = np.eye(args.num_classes, dtype=float)[predictions]
            fold_result['splits'][split] = compute_metrics(
                labels, predictions, probabilities
            )
        folds.append(fold_result)

    if not folds:
        raise ValueError(f'未找到 fold_*/test_results.json: {args.cv_dir}')

    summary = aggregate(folds)
    output = {
        'strategy': 'uniform_random',
        'seed': args.seed,
        'num_classes': args.num_classes,
        'n_folds': len(folds),
        'folds': folds,
        'summary': summary,
    }
    json_path = args.cv_dir / 'random_baseline_cv.json'
    with json_path.open('w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    table = write_table(args.cv_dir / 'random_baseline_metrics_table.md', summary)
    cv_summary_path = args.cv_dir / 'cv_summary.json'
    if cv_summary_path.exists():
        with cv_summary_path.open() as f:
            model_summary = json.load(f)['summary']
        write_comparison_table(
            args.cv_dir / 'cv_vs_random_metrics_table.md',
            model_summary,
            summary,
        )
    print(table)
    print(f'\nJSON: {json_path}')


if __name__ == '__main__':
    main()
