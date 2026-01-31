# -*- coding: utf-8 -*-
"""
检查 category 到 label 的映射关系

验证原始 category 值 (1,2,3) 是否正确转换为 label (0,1,2)
"""

import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import UnifiedConfig
from src.models.dl_dataset import SequenceConfig
from src.data.split_strategy import TwoByTwoSplitter
from experiments.dl_transformer_experiment import LightweightGazeDataset


def check_category_mapping():
    """检查 category 到 label 的映射"""
    print("="*80)
    print("Category → Label 映射检查")
    print("="*80)

    # 加载数据
    data_path = 'data/processed/processed_data.pkl'
    with open(data_path, 'rb') as f:
        all_data = pickle.load(f)

    # 划分
    splitter = TwoByTwoSplitter(100, 20, 42)
    splits = splitter.split(all_data)

    # 创建数据集
    train_dataset = LightweightGazeDataset(
        data=splits['train'],
        config=SequenceConfig(),
        fit_normalizer=True,
        task_type='classification',
    )

    # 检查每个样本的原始 category 和转换后的 label
    print(f"\n检查前 20 个样本:")
    print(f"{'样本':<6} {'原始category':<12} {'转换后label':<12} {'对应存储类别'}")
    print("-"*60)

    category_to_label = {}
    label_to_category = {}

    for i in range(min(len(train_dataset), len(train_dataset.data))):
        sample_data = train_dataset.data[i]  # 原始数据
        sample = train_dataset[i]            # 经过 __getitem__ 处理

        original_category = sample_data['category']
        final_label = sample['label'].item()

        # 存储映射
        if original_category not in category_to_label:
            category_to_label[original_category] = []
        category_to_label[original_category].append(final_label)

        if i < 20:
            stored_class = final_label + 1  # 转回存储格式
            print(f"{i:<6} {original_category:<12} {final_label:<12} 类别{stored_class}")

    # 验证映射一致性
    print(f"\n映射关系验证:")
    print("-"*60)

    for cat in sorted(category_to_label.keys()):
        labels = category_to_label[cat]
        unique_labels = set(labels)
        print(f"原始 category {cat} → 转换后的 label: {unique_labels}")

        if len(unique_labels) != 1:
            print(f"  ⚠️  警告: 同一个 category 映射到多个 label!")
        else:
            expected_label = cat - 1
            actual_label = list(unique_labels)[0]
            if actual_label == expected_label:
                print(f"  ✅ 正确: category {cat} → label {actual_label}")
            else:
                print(f"  ❌ 错误: category {cat} → label {actual_label}, 期望 {expected_label}")

    # 统计原始 category 分布
    print(f"\n训练集原始 category 分布:")
    categories = [d['category'] for d in train_dataset.data]
    import numpy as np
    unique, counts = np.unique(categories, return_counts=True)
    for cat, count in zip(unique, counts):
        print(f"  category {cat}: {count} 样本 ({count/len(categories)*100:.1f}%)")

    # 统计转换后 label 分布
    print(f"\n训练集转换后 label 分布:")
    labels = [train_dataset[i]['label'].item() for i in range(len(train_dataset))]
    unique, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  label {label}: {count} 样本 ({count/len(labels)*100:.1f}%)")


if __name__ == '__main__':
    check_category_mapping()
