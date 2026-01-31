# -*- coding: utf-8 -*-
"""
调试脚本：打印第一个 batch 的标签值

用于验证训练时标签是否正确（应该是 0, 1, 2）
"""

import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import UnifiedConfig
from src.models.dl_dataset import SequenceConfig
from src.data.split_strategy import TwoByTwoSplitter


def debug_labels():
    """调试标签值"""
    print("="*80)
    print("标签值调试")
    print("="*80)

    # 加载配置
    config = UnifiedConfig.from_json('configs/hitask.json')

    # 加载数据
    data_path = 'data/processed/processed_data.pkl'
    if not Path(data_path).exists():
        print(f"数据文件不存在: {data_path}")
        return

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    # 划分数据
    splitter = TwoByTwoSplitter(100, 20, 42)
    splits = splitter.split(data)

    # 创建数据集
    from experiments.dl_transformer_experiment import LightweightGazeDataset

    train_dataset = LightweightGazeDataset(
        data=splits['train'],
        config=SequenceConfig(),
        fit_normalizer=True,
        task_type='classification',
    )

    print(f"\n数据集大小: {len(train_dataset)}")

    # 检查前 10 个样本的标签
    print(f"\n前 10 个样本的标签:")
    for i in range(min(10, len(train_dataset))):
        sample = train_dataset[i]
        label = sample['label']
        print(f"  样本 {i}: label={label.item()} (dtype={label.dtype})")

    # 统计所有标签
    all_labels = []
    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        all_labels.append(sample['label'].item())

    import numpy as np
    unique, counts = np.unique(all_labels, return_counts=True)

    print(f"\n训练集标签分布:")
    for u, c in zip(unique, counts):
        print(f"  标签 {u}: {c} 样本 ({c/len(all_labels)*100:.1f}%)")

    # 验证类别权重计算
    print(f"\n类别权重计算:")

    # 方法1: 使用 dataset.data 和 category
    labels_from_category = [d['category'] - 1 for d in train_dataset.data]
    class_counts = np.bincount(labels_from_category, minlength=3)
    weights = 1.0 / (class_counts + 1e-6)
    weights = weights / weights.sum() * 3
    print(f"  从 category 计算: class_counts={class_counts}, weights={weights}")

    # 方法2: 使用 dataset 的实际标签
    class_counts2 = np.bincount(all_labels, minlength=3)
    weights2 = 1.0 / (class_counts2 + 1e-6)
    weights2 = weights2 / weights2.sum() * 3
    print(f"  从实际标签计算: class_counts={class_counts2}, weights={weights2}")

    if not np.array_equal(class_counts, class_counts2):
        print(f"\n⚠️  警告: 两种方法得到的类别分布不一致！")
        print(f"   这意味着 dataset.data 中的 category 和 __getitem__ 返回的 label 不一致！")


if __name__ == '__main__':
    debug_labels()
