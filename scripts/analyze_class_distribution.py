# -*- coding: utf-8 -*-
"""
类别分布分析脚本

深入分析训练集、验证集和测试集的类别分布，
以及类别权重计算是否正确。
"""

import json
import sys
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.split_strategy import TwoByTwoSplitter
from src.models.dl_dataset import HierarchicalGazeDataset, SequenceConfig


def compute_class_weights(labels: np.ndarray, num_classes: int = 3) -> Tuple[torch.Tensor, Dict]:
    """
    计算类别权重（与训练器中相同的逻辑）

    权重计算公式: w_i = 1 / (count_i + epsilon)
    然后归一化使均值为1
    """
    class_counts = np.bincount(labels, minlength=num_classes)

    # 计算权重
    epsilon = 1e-6
    class_weights = 1.0 / (class_counts + epsilon)
    class_weights = class_weights / class_weights.mean()  # 归一化

    return torch.tensor(class_weights, dtype=torch.float32), {
        'counts': class_counts.tolist(),
        'weights': class_weights.tolist(),
        'proportions': (class_counts / class_counts.sum()).tolist()
    }


def analyze_dataset(dataset: HierarchicalGazeDataset, name: str) -> Dict:
    """分析单个数据集的类别分布"""
    labels = []
    for i in range(len(dataset)):
        label = dataset[i]['label'].item()
        labels.append(label)

    labels = np.array(labels)

    # 计算统计信息
    unique, counts = np.unique(labels, return_counts=True)
    class_dist = dict(zip(unique.tolist(), counts.tolist()))

    # 确保所有类别都有计数
    for c in [1, 2, 3]:
        if c not in class_dist:
            class_dist[c] = 0

    # 计算类别权重
    weights, weight_info = compute_class_weights(labels)

    return {
        'name': name,
        'total_samples': len(labels),
        'class_distribution': class_dist,
        'class_counts': weight_info['counts'],
        'class_proportions': weight_info['proportions'],
        'class_weights': weight_info['weights'],
        'majority_class': max(class_dist, key=class_dist.get),
        'minority_class': min(class_dist, key=class_dist.get),
        'imbalance_ratio': class_dist[class_dist[max(class_dist, key=class_dist.get)]] /
                          max(class_dist[class_dist[min(class_dist, key=class_dist.get)]], 1)
    }


def main():
    """主函数"""
    print("=" * 80)
    print("类别分布深度分析")
    print("=" * 80)

    # 加载预处理数据
    processed_data_path = 'data/processed/processed_data.pkl'

    if not Path(processed_data_path).exists():
        print(f"错误: 预处理数据不存在: {processed_data_path}")
        print("请先运行预处理脚本生成数据")
        return

    print(f"\n加载预处理数据: {processed_data_path}")
    import pickle
    with open(processed_data_path, 'rb') as f:
        all_data = pickle.load(f)
    print(f"加载了 {len(all_data)} 个被试数据")

    # 划分数据集（使用默认配置）
    train_subjects = 100
    train_tasks = 20
    random_seed = 42

    splitter = TwoByTwoSplitter(
        train_subjects=train_subjects,
        train_tasks=train_tasks,
        random_state=random_seed
    )

    splits = splitter.split(all_data)

    # 创建序列配置
    seq_config = SequenceConfig.from_gaze_data(all_data)

    # 分析训练集
    print("\n" + "=" * 80)
    print("数据集类别分布分析")
    print("=" * 80)

    results = {}

    # 创建训练集（使用 fit_normalizer=True）
    train_dataset = HierarchicalGazeDataset(
        data=splits['train'],
        seq_config=seq_config,
        fit_normalizer=True
    )
    train_analysis = analyze_dataset(train_dataset, "训练集 (Train)")
    results['train'] = train_analysis

    # 创建验证集（使用训练集的归一化参数）
    from sklearn.model_selection import KFold
    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    train_indices, val_indices = next(kf.split(train_dataset))

    # 创建验证集
    val_dataset = torch.utils.data.Subset(train_dataset, val_indices)
    val_labels = [train_dataset[i]['label'].item() for i in val_indices]
    val_labels = np.array(val_labels)

    unique, counts = np.unique(val_labels, return_counts=True)
    class_dist = dict(zip(unique.tolist(), counts.tolist()))
    for c in [1, 2, 3]:
        if c not in class_dist:
            class_dist[c] = 0

    results['val'] = {
        'name': '验证集 (Val)',
        'total_samples': len(val_labels),
        'class_distribution': class_dist,
        'class_counts': [class_dist.get(1, 0), class_dist.get(2, 0), class_dist.get(3, 0)],
        'class_proportions': [class_dist.get(1, 0)/len(val_labels), class_dist.get(2, 0)/len(val_labels),
                             class_dist.get(3, 0)/len(val_labels)]
    }

    # 分析测试集
    for test_name in ['test1', 'test2', 'test3']:
        test_dataset = HierarchicalGazeDataset(
            data=splits[test_name],
            seq_config=seq_config,
            fit_normalizer=False  # 使用训练集的归一化参数
        )
        test_analysis = analyze_dataset(test_dataset, f"测试集 ({test_name.upper()})")
        results[test_name] = test_analysis

    # 打印结果
    print(f"\n{'数据集':<20} {'总数':<8} {'类别1':<8} {'类别2':<8} {'类别3':<8} "
          f"{'比例(1:2:3)':<15} {'不平衡比':<10}")
    print("-" * 100)

    for key in ['train', 'val', 'test1', 'test2', 'test3']:
        r = results[key]
        if 'class_counts' in r:
            counts = r['class_counts']
            props = r['class_proportions']
            ratio = f"{props[0]:.2f}:{props[1]:.2f}:{props[2]:.2f}"
            print(f"{r['name']:<20} {r['total_samples']:<8} "
                  f"{counts[0]:<8} {counts[1]:<8} {counts[2]:<8} "
                  f"{ratio:<15} {r.get('imbalance_ratio', 0):.2f}")

    # 打印类别权重
    print("\n" + "=" * 80)
    print("类别权重分析")
    print("=" * 80)

    train_r = results['train']
    print(f"\n训练集类别分布: {train_r['class_counts']}")
    print(f"训练集类别比例: {train_r['class_proportions']}")
    print(f"\n计算的类别权重: {train_r['class_weights']}")

    # 分析类别权重是否合理
    weights = np.array(train_r['class_weights'])
    counts = np.array(train_r['class_counts'])

    print(f"\n权重 vs 样本数:")
    for i, c in enumerate([1, 2, 3]):
        print(f"  类别{c}: 样本数={counts[i]:>3}, 权重={weights[i]:.4f}, "
              f"权重×样本数={weights[i]*counts[i]:.2f}")

    # 检查权重是否过小或过大
    if weights.max() / weights.min() > 10:
        print(f"\n⚠️  警告: 类别权重差异过大 (比值={weights.max()/weights.min():.2f})")
        print("   这可能导致训练不稳定")

    # 检查类别不平衡
    imbalance = train_r['imbalance_ratio']
    if imbalance > 3:
        print(f"\n⚠️  警告: 类别不平衡严重 (比值={imbalance:.2f})")
        print("   建议:")
        print("   1. 使用更强的类别权重")
        print("   2. 使用 Focal Loss")
        print("   3. 对少数类进行过采样")

    # 生成诊断报告
    report_path = Path('outputs/dl_models/analysis/class_distribution_report.md')
    report_path.parent.mkdir(exist_ok=True)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 类别分布分析报告\n\n")

        f.write("## 数据集类别分布\n\n")
        f.write(f"| 数据集 | 总数 | 类别1 | 类别2 | 类别3 | 比例 |\n")
        f.write(f"|--------|------|-------|-------|-------|------|\n")

        for key in ['train', 'val', 'test1', 'test2', 'test3']:
            r = results[key]
            if 'class_counts' in r:
                counts = r['class_counts']
                props = r['class_proportions']
                ratio = f"{props[0]:.2f}:{props[1]:.2f}:{props[2]:.2f}"
                f.write(f"| {r['name']} | {r['total_samples']} | "
                       f"{counts[0]} | {counts[1]} | {counts[2]} | {ratio} |\n")

        f.write("\n## 类别权重分析\n\n")
        f.write(f"训练集类别分布: {train_r['class_counts']}\n\n")
        f.write(f"计算的类别权重: {train_r['class_weights']}\n\n")

        f.write("### 权重 vs 样本数\n\n")
        f.write("| 类别 | 样本数 | 权重 | 权重×样本数 |\n")
        f.write("|------|--------|------|-------------|\n")
        for i, c in enumerate([1, 2, 3]):
            f.write(f"| {c} | {counts[i]} | {weights[i]:.4f} | {weights[i]*counts[i]:.2f} |\n")

        f.write("\n## 诊断结论\n\n")

        if imbalance > 3:
            f.write(f"### ⚠️ 严重类别不平衡\n\n")
            f.write(f"- 不平衡比值: {imbalance:.2f}\n")
            f.write(f"- 多数类: 类别{train_r['majority_class']} ({counts[train_r['majority_class']-1]}样本)\n")
            f.write(f"- 少数类: 类别{train_r['minority_class']} ({counts[train_r['minority_class']-1]}样本)\n\n")
            f.write("### 建议改进措施\n\n")
            f.write("1. **调整类别权重计算方式**\n")
            f.write("   - 当前使用: `w_i = 1 / (count_i + epsilon)`\n")
            f.write("   - 建议使用: `w_i = sqrt(N / (C * count_i))` 或 `w_i = median(count) / count_i`\n\n")
            f.write("2. **使用 Focal Loss**\n")
            f.write("   - 对难分类样本给予更高权重\n")
            f.write("   - 公式: `FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)`\n\n")
            f.write("3. **调整学习率和优化器**\n")
            f.write("   - 降低学习率\n")
            f.write("   - 使用 per-class learning rate\n\n")
        else:
            f.write("### ✅ 类别分布相对均衡\n")
            f.write(f"- 不平衡比值: {imbalance:.2f}\n\n")

    print(f"\n分析报告已保存到: {report_path}")


if __name__ == '__main__':
    main()
