# -*- coding: utf-8 -*-
"""
十折交叉验证训练脚本（任务级）

划分方案：
1. 2x2 划分（与 train_task_level 一致）：
   - train_pool = 前100被试 × 前20题 (2000 样本)
   - test1 = 后57被试 × 前20题（新被试+旧题）
   - test2 = 前100被试 × 后10题（旧被试+新题）
   - test3 = 后57被试 × 后10题（新被试+新题）
2. 样本级十折（StratifiedKFold, shuffle=True, 固定 seed）：
   - 在 train_pool (2000 样本) 上分 10 折，每折 val = 200 样本（10%），train = 1800 样本
   - 10 折轮转后每个训练样本恰好当过一次 val（覆盖全部 2000 样本）

训练/评估与 train_task_level 完全一致（TaskLevelEncoder + TaskLevelTrainer，
早停 val_f1_macro，评估 loss/ACC/Macro-F1/G-Mean/AUC-PR/QWK，并保留
f1_weighted/spearman）。

用法：
    python scripts/train_task_level_cv.py --config configs/task_level_ordinal_manual11_bilstm.json
    python scripts/train_task_level_cv.py --folds 10 --fold-by sample --output-dir outputs/task_level_cv

输出：
    <output_dir>/<exp_name>_cv<folds>fold_<timestamp>/
        fold_01/...fold_10/   best_model.pt, final_model.pt, history.json,
                             test_results.json, training_curves.png
        cv_summary.json      每折×4集 全部指标 + mean±std
        config.json
"""

import os
import sys
import json
import pickle
import logging
import argparse
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from scipy.stats import spearmanr
from sklearn.metrics import recall_score, average_precision_score, cohen_kappa_score
from sklearn.preprocessing import label_binarize

# 添加项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.task_level_model import TaskLevelEncoder
from src.models.task_level_dataset import (
    TaskLevelGazeDataset,
    TaskLevelSequenceConfig,
    task_level_collate_fn,
)
from src.models.task_level_trainer import TaskLevelTrainer
from src.models.augmentation import GazeAugmentation

# 复用 train_task_level 的公共函数
from scripts.train_task_level import (
    set_seed,
    compute_class_weights,
    compute_sample_weights,
    compute_ordinal_pos_weight,
    plot_training_curves,
    split_2x2,
    Subset,
)

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="任务级十折交叉验证训练",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str,
                        default="configs/task_level_ordinal_manual11_bilstm.json")
    parser.add_argument("--data", type=str, default=None,
                        help="覆盖 config 的 processed 数据路径")
    parser.add_argument("--folds", type=int, default=10, help="交叉验证折数")
    parser.add_argument("--fold-by", choices=["sample"], default="sample",
                        help="分折单位（当前仅样本级 StratifiedKFold）")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="覆盖 config 的 experiment.output_dir")
    parser.add_argument("--resume-dir", type=str, default=None,
                        help="指定已有输出目录直接续跑（跳过已完成的 fold）")
    parser.add_argument("--seed", type=int, default=None,
                        help="覆盖随机种子（默认用 config 的 random_seed，全折固定）")
    return parser.parse_args()


def build_2x2_pool(dataset, train_subjects=100, train_tasks=20, seed=42):
    """2x2 划分，返回 (train_pool, test1, test2, test3) indices。
    train_val_split=1.0 使 split_2x2 不划分 train/val，train 即完整 2000 样本池。"""
    train_indices, val_indices, test1, test2, test3 = split_2x2(
        dataset,
        train_subjects=train_subjects,
        train_tasks=train_tasks,
        train_val_split=1.0,
        seed=seed,
    )
    assert len(val_indices) == 0, "train_val_split=1.0 时应无 val"
    return train_indices, test1, test2, test3


def make_sample_folds(pool_indices, labels, n_folds, seed):
    """样本级 StratifiedKFold。返回 [(train_idx, val_idx), ...]（均为 pool 内位置）。"""
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(skf.split(pool_indices, labels))


def build_loader(dataset, indices, batch_size, num_workers, shuffle=False, sampler=None):
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=task_level_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )


def get_class_weights(train_labels, config):
    if not config['training']['use_class_weights']:
        return None
    if config['training'].get('class_weights_mode', 'auto') == 'manual':
        return torch.tensor(config['training']['class_weights'], dtype=torch.float32)
    return compute_class_weights(np.array(train_labels))


def get_ordinal_pos_weight(config):
    if config['model'].get('head_type', 'classification') != 'ordinal':
        return None
    mode = config['training'].get('ordinal_pos_weight_mode', 'auto')
    if mode == 'none':
        return None
    if mode == 'manual':
        w = config['training'].get('ordinal_pos_weight')
        if w is None:
            raise ValueError("ordinal_pos_weight_mode=manual 时必须提供 training.ordinal_pos_weight")
        return torch.tensor(w, dtype=torch.float32)
    if mode == 'auto':
        return compute_ordinal_pos_weight(
            config['training'].get('ordinal_pos_weight_clip', [0.5, 3.0])
        )
    raise ValueError(f"不支持的 ordinal_pos_weight_mode: {mode}")


def compute_gmean(labels, predictions):
    """多分类 G-Mean：各类别 recall 的几何均值。"""
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    classes = np.unique(labels)
    recalls = recall_score(
        labels, predictions, labels=classes, average=None, zero_division=0
    )
    if recalls.size == 0 or np.any(recalls <= 0):
        return 0.0
    return float(np.exp(np.mean(np.log(recalls))))


def safe_auc_pr(labels, probabilities):
    """多分类 AUC-PR：One-vs-Rest macro average precision。"""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
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


def evaluate_sets(trainer, eval_sets):
    """逐集评估，补充 Spearman、G-Mean、AUC-PR、QWK。"""
    results = {}
    for name, loader in eval_sets:
        metrics = trainer.evaluate(loader, desc=name)
        labels = np.asarray(metrics['labels'], dtype=int)
        predictions = np.asarray(metrics['predictions'], dtype=int)
        probabilities = np.asarray(metrics['probabilities'], dtype=float)
        spearman_corr, spearman_p = spearmanr(labels, predictions)
        metrics['spearman'] = float(spearman_corr)
        metrics['spearman_p'] = float(spearman_p)
        metrics['g_mean'] = compute_gmean(labels, predictions)
        metrics['auc_pr'] = safe_auc_pr(labels, probabilities)
        metrics['qwk'] = float(
            cohen_kappa_score(labels, predictions, weights='quadratic')
        )
        results[name] = metrics
    return results


def train_one_fold(
    fold_k,
    dataset,
    train_idx,
    val_idx,
    test1_idx,
    test2_idx,
    test3_idx,
    config,
    device,
    fold_dir,
    seed,
):
    """训练单折（逻辑与 train_task_level.main 一致），返回 fold 结果 dict。"""
    set_seed(seed)

    # ---- 数据增强（仅训练集样本） ----
    aug_config = config.get('augmentation', {})
    if aug_config:
        dataset.augmentation = GazeAugmentation(aug_config)
        dataset.training_indices = set(train_idx)
    else:
        dataset.augmentation = None
        dataset.training_indices = set()

    train_labels = [dataset.samples[i]['label'] for i in train_idx]

    # ---- 类别权重 ----
    class_weights = get_class_weights(train_labels, config)

    # ---- DataLoader ----
    num_workers = config['training'].get('num_workers', 4)
    batch_size = config['training']['batch_size']
    sampler = None
    if config['training'].get('use_balanced_sampler', False):
        mode = config['training'].get('balanced_sampler_mode', 'effective_num')
        beta = config['training'].get('balanced_sampler_beta', 0.999)
        sample_weights = compute_sample_weights(
            np.array(train_labels), mode=mode, beta=beta
        )
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.float32),
            num_samples=len(train_idx),
            replacement=True,
        )
    train_loader = build_loader(
        dataset, train_idx, batch_size, num_workers,
        shuffle=(sampler is None), sampler=sampler,
    )
    val_loader = build_loader(dataset, val_idx, batch_size, num_workers)
    test1_loader = build_loader(dataset, test1_idx, batch_size, num_workers)
    test2_loader = build_loader(dataset, test2_idx, batch_size, num_workers)
    test3_loader = build_loader(dataset, test3_idx, batch_size, num_workers)

    # ---- 模型 ----
    mcfg = config['model']
    segment_encoder_type = mcfg['segment_encoder_type']
    model = TaskLevelEncoder(
        input_dim=config['sequence']['input_dim'],
        max_seq_len=config['sequence']['max_seq_len'],
        max_segments=config['sequence']['max_segments'],
        segment_d_model=mcfg['segment_d_model'],
        segment_nhead=mcfg['segment_nhead'],
        segment_num_layers=mcfg['segment_num_layers'],
        segment_encoder_type=segment_encoder_type,
        segment_rnn_hidden_size=mcfg.get('segment_rnn_hidden_size', mcfg['segment_d_model']),
        segment_rnn_layers=mcfg.get('segment_rnn_layers', 1),
        segment_rnn_dropout=mcfg.get('segment_rnn_dropout', mcfg.get('dropout', 0.0)),
        segment_cnn_channels=mcfg.get('segment_cnn_channels', None),
        segment_cnn_kernel_sizes=mcfg.get('segment_cnn_kernel_sizes', None),
        attention_dim=mcfg['attention_dim'],
        task_embedding_dim=mcfg['task_embedding_dim'],
        use_task_embedding=mcfg.get('use_task_embedding', True),
        task_embedding_type=mcfg.get('task_embedding_type', 'independent'),
        use_conditional_pooling=mcfg.get('use_conditional_pooling', False),
        dropout=mcfg['dropout'],
        num_classes=mcfg['num_classes'],
        head_type=mcfg.get('head_type', 'classification'),
    ).to(device)

    # ---- Trainer ----
    tcfg = config['training']
    trainer = TaskLevelTrainer(
        model=model,
        device=device,
        num_classes=mcfg['num_classes'],
        head_type=mcfg.get('head_type', 'classification'),
        class_weights=class_weights if tcfg['use_class_weights'] else None,
        lr=tcfg['lr'],
        weight_decay=tcfg['weight_decay'],
        grad_clip=tcfg['grad_clip'],
        use_focal_loss=tcfg.get('use_focal_loss', False),
        focal_loss_alpha=tcfg.get('focal_loss_alpha', None),
        focal_loss_gamma=tcfg.get('focal_loss_gamma', 2.0),
        label_smoothing=tcfg.get('label_smoothing', 0.0),
        ordinal_thresholds=tcfg.get('ordinal_thresholds', None),
        ordinal_pos_weight=get_ordinal_pos_weight(config),
    )

    # ---- 训练循环（同 train_task_level） ----
    early_stop_metric = tcfg.get('early_stop_metric', 'f1_weighted')
    best_val_metric = -1.0
    patience_counter = 0
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [], 'train_f1_macro': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_f1_macro': [],
    }

    for epoch in range(1, tcfg['epochs'] + 1):
        train_metrics = trainer.train_epoch(train_loader, epoch, tcfg['epochs'])
        val_metrics = trainer.evaluate(val_loader, desc=f"Epoch {epoch} [Val]")

        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['train_f1'].append(train_metrics['f1_weighted'])
        history['train_f1_macro'].append(train_metrics['f1_macro'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1_weighted'])
        history['val_f1_macro'].append(val_metrics['f1_macro'])

        print(f"fold{fold_k} epoch {epoch:<4d} "
              f"train(loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.4f} "
              f"f1w={train_metrics['f1_weighted']:.4f}) "
              f"val(loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.4f} "
              f"f1w={val_metrics['f1_weighted']:.4f} f1m={val_metrics['f1_macro']:.4f})")

        trainer.scheduler.step(val_metrics['loss'])

        current_metric = val_metrics[early_stop_metric]
        if current_metric > best_val_metric:
            best_val_metric = current_metric
            patience_counter = 0
            trainer.save_checkpoint(fold_dir / 'best_model.pt', epoch, best_val_metric)
        else:
            patience_counter += 1
            if patience_counter >= tcfg['patience']:
                print(f"fold{fold_k} 早停! {tcfg['patience']} epoch 无改善 "
                      f"(Best Val {early_stop_metric}: {best_val_metric:.4f})")
                break

    best_epoch = int(np.argmax(history[f'val_{early_stop_metric}']) + 1) if history[f'val_{early_stop_metric}'] else 0
    trainer.save_checkpoint(fold_dir / 'final_model.pt', epoch, best_val_metric)

    # ---- 评估 best 模型 ----
    trainer.load_checkpoint(fold_dir / 'best_model.pt')
    eval_sets = [
        ("Val (同分布)", val_loader),
        ("Test1 (新被试+旧题)", test1_loader),
        ("Test2 (旧被试+新题)", test2_loader),
        ("Test3 (新被试+新题)", test3_loader),
    ]
    eval_results = evaluate_sets(trainer, eval_sets)

    # ---- 保存折内产出 ----
    fold_results_json = {}
    for name, metrics in eval_results.items():
        fold_results_json[name] = {
            'loss': metrics['loss'],
            'accuracy': metrics['accuracy'],
            'f1_weighted': metrics['f1_weighted'],
            'f1_macro': metrics['f1_macro'],
            'g_mean': metrics['g_mean'],
            'auc_pr': metrics['auc_pr'],
            'qwk': metrics['qwk'],
            'spearman': metrics['spearman'],
            'spearman_p': metrics['spearman_p'],
            'predictions': [int(p) for p in metrics['predictions']],
            'labels': [int(l) for l in metrics['labels']],
            'probabilities': [[float(x) for x in prob] for prob in metrics.get('probabilities', [])],
        }
    with open(fold_dir / 'test_results.json', 'w') as f:
        json.dump(fold_results_json, f, indent=2, ensure_ascii=False)

    with open(fold_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    plot_training_curves(history, fold_dir / 'training_curves.png', best_epoch, early_stop_metric)

    # 精简指标（不含 predictions/labels，避免 summary 过大）—— 全部转 Python 原生类型
    fold_summary = {'fold': int(fold_k), 'best_epoch': best_epoch,
                    f'best_val_{early_stop_metric}': float(best_val_metric)}
    for name, metrics in eval_results.items():
        fold_summary[name] = {
            'loss': float(metrics['loss']), 'accuracy': float(metrics['accuracy']),
            'f1_weighted': float(metrics['f1_weighted']), 'f1_macro': float(metrics['f1_macro']),
            'g_mean': float(metrics['g_mean']), 'auc_pr': float(metrics['auc_pr']),
            'qwk': float(metrics['qwk']), 'spearman': float(metrics['spearman']),
            'spearman_p': float(metrics['spearman_p']),
        }
    return fold_summary


def aggregate(fold_summaries, split_names, metric_names):
    """对每个 split×metric 计算 mean±std。"""
    summary = {}
    for split in split_names:
        summary[split] = {}
        for metric in metric_names:
            values = [float(f[split][metric]) for f in fold_summaries]
            summary[split][metric] = {
                'values': values,
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
            }
    return summary


def resume_fold_summary(fold_dir, early_stop_metric):
    """从已有 fold 输出重建 fold_summary（用于断点续跑）。"""
    with open(fold_dir / 'test_results.json') as f:
        tr = json.load(f)
    with open(fold_dir / 'history.json') as f:
        hist = json.load(f)
    key = f'val_{early_stop_metric}'
    if key in hist and hist[key]:
        best_metric = float(max(hist[key]))
        best_epoch = int(np.argmax(hist[key]) + 1)
    else:
        best_metric, best_epoch = 0.0, 0
    fs = {'fold': int(fold_dir.name.split('_')[1]), 'best_epoch': best_epoch,
          f'best_val_{early_stop_metric}': best_metric}
    for name, m in tr.items():
        fs[name] = {k2: float(m[k2]) for k2 in
                    ('loss', 'accuracy', 'f1_weighted', 'f1_macro', 'g_mean',
                     'auc_pr', 'qwk', 'spearman', 'spearman_p')}
    return fs


def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    args = parse_args()

    with open(args.config) as f:
        config = json.load(f)

    seed = args.seed if args.seed is not None else config['experiment']['random_seed']
    n_folds = args.folds
    train_subjects = config['experiment']['train_subjects']
    train_tasks = config['experiment']['train_tasks']
    data_path = args.data or config['data']['processed_data_path']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"设备: {device}")
    logger.info(f"数据: {data_path} | folds={n_folds} | fold_by={args.fold_by} | seed={seed}")

    # ---- 加载数据 + 构建任务级数据集 ----
    with open(data_path, 'rb') as f:
        processed_data = pickle.load(f)
    logger.info(f"加载 {len(processed_data)} 个被试")

    seq_config = TaskLevelSequenceConfig(
        max_seq_len=config['sequence']['max_seq_len'],
        max_segments=config['sequence']['max_segments'],
    )
    dataset = TaskLevelGazeDataset(processed_data=processed_data, config=seq_config,
                                   fit_normalizer=False)
    logger.info(f"总样本数: {len(dataset)}")

    # ---- 2x2 划分 ----
    train_pool, test1, test2, test3 = build_2x2_pool(
        dataset, train_subjects=train_subjects, train_tasks=train_tasks, seed=seed,
    )
    pool_labels = np.array([dataset.samples[i]['label'] for i in train_pool])
    logger.info(f"train_pool: {len(train_pool)} (分布 {np.bincount(pool_labels)})")
    logger.info(f"test1: {len(test1)} | test2: {len(test2)} | test3: {len(test3)}")

    # ---- 十折 ----
    folds = make_sample_folds(train_pool, pool_labels, n_folds, seed)
    sizes = [len(v) for _, v in folds]
    assert all(s == len(train_pool) // n_folds for s in sizes), f"折大小不均: {sizes}"
    logger.info(f"每折 val 大小: {sizes} (覆盖全部训练样本: {sum(sizes) == len(train_pool)})")

    # ---- 输出目录 ----
    if args.resume_dir:
        out_root = Path(args.resume_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        logger.info(f"续跑模式，输出目录: {out_root}")
    else:
        base_out = args.output_dir or config['experiment']['output_dir']
        exp_name = config['experiment'].get('name', 'exp')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_root = Path(base_out) / f"{exp_name}_cv{n_folds}fold_{timestamp}"
        out_root.mkdir(parents=True, exist_ok=True)

    # ---- 逐折训练（已有输出的折自动跳过，用于断点续跑） ----
    fold_summaries = []
    for k, (tr_pos, va_pos) in enumerate(folds, start=1):
        fold_dir = out_root / f"fold_{k:02d}"
        if (fold_dir / 'test_results.json').exists():
            print(f"fold_{k:02d} 已存在，跳过（resume）")
            fs = resume_fold_summary(fold_dir, config['training'].get('early_stop_metric', 'f1_weighted'))
            with open(fold_dir / 'fold_summary.json', 'w') as f:
                json.dump(fs, f, indent=2, ensure_ascii=False)
            fold_summaries.append(fs)
            continue
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_idx = [train_pool[i] for i in tr_pos]
        val_idx = [train_pool[i] for i in va_pos]
        print(f"\n{'='*80}\n开始折 {k}/{n_folds} "
              f"(train={len(train_idx)}, val={len(val_idx)})\n{'='*80}")
        fold_summary = train_one_fold(
            k, dataset, train_idx, val_idx, test1, test2, test3,
            config, device, fold_dir, seed,
        )
        fold_summaries.append(fold_summary)
        with open(out_root / f'fold_{k:02d}' / 'fold_summary.json', 'w') as f:
            json.dump(fold_summary, f, indent=2, ensure_ascii=False)

    # ---- 汇总 ----
    split_names = ["Val (同分布)", "Test1 (新被试+旧题)", "Test2 (旧被试+新题)", "Test3 (新被试+新题)"]
    metric_names = [
        'loss', 'accuracy', 'f1_weighted', 'f1_macro',
        'g_mean', 'auc_pr', 'qwk', 'spearman',
    ]
    cv_summary = {
        'n_folds': n_folds,
        'fold_by': args.fold_by,
        'seed': seed,
        'data_path': data_path,
        'config_path': args.config,
        'folds': fold_summaries,
        'summary': aggregate(fold_summaries, split_names, metric_names),
    }
    with open(out_root / 'cv_summary.json', 'w') as f:
        json.dump(cv_summary, f, indent=2, ensure_ascii=False)
    with open(out_root / 'config.json', 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # ---- 保存论文表格指标 ----
    table_lines = [
        '# Task-level 10-fold CV metrics',
        '',
        '| Split | ACC | Macro-F1 | G-Mean | AUC-PR | QWK |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for split in split_names:
        s = cv_summary['summary'][split]
        cells = [
            f"{s[metric]['mean']:.4f} ± {s[metric]['std']:.4f}"
            for metric in ('accuracy', 'f1_macro', 'g_mean', 'auc_pr', 'qwk')
        ]
        table_lines.append(f"| {split} | " + ' | '.join(cells) + ' |')
    table_lines.extend([
        '',
        'Definitions: G-Mean = geometric mean of per-class recall; '
        'AUC-PR = macro one-vs-rest average precision; '
        'QWK = quadratic weighted Cohen kappa.',
    ])
    (out_root / 'cv_metrics_table.md').write_text(
        '\n'.join(table_lines) + '\n', encoding='utf-8'
    )

    # ---- 打印汇总表 ----
    print("\n" + "=" * 95)
    print("十折交叉验证结果汇总 (mean ± std)")
    print("=" * 95)
    header = f"{'数据集':<25} {'ACC':<14} {'Macro-F1':<14} {'G-Mean':<14} {'AUC-PR':<14} {'QWK':<14}"
    print(header)
    print("-" * 110)
    for split in split_names:
        s = cv_summary['summary'][split]
        print(f"{split:<25} "
              f"{s['accuracy']['mean']:.4f}±{s['accuracy']['std']:.4f}   "
              f"{s['f1_macro']['mean']:.4f}±{s['f1_macro']['std']:.4f}   "
              f"{s['g_mean']['mean']:.4f}±{s['g_mean']['std']:.4f}   "
              f"{s['auc_pr']['mean']:.4f}±{s['auc_pr']['std']:.4f}   "
              f"{s['qwk']['mean']:.4f}±{s['qwk']['std']:.4f}")
    print("=" * 95)
    logger.info(f"完成! 结果保存在: {out_root}")


if __name__ == '__main__':
    main()
