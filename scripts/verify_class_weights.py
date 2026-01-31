# -*- coding: utf-8 -*-
"""
类别权重计算验证脚本

验证当前的权重计算逻辑是否有问题
"""

import numpy as np

# 测试数据：假设类别分布（基于之前的分析）
# 类别 1: 25 样本，类别 2: 111 样本，类别 3: 21 样本
category_1based = [1] * 25 + [2] * 111 + [3] * 21
print("原始类别分布 (1-based):")
print(f"  类别 1: 25, 类别 2: 111, 类别 3: 21")

# 转换为 0-based
labels_0based = [c - 1 for c in category_1based]
print(f"\n转换为 0-based 后的标签:")
print(f"  min={min(labels_0based)}, max={max(labels_0based)}")
print(f"  bincount: {np.bincount(labels_0based)}")

# 当前的权重计算方法（来自 dl_trainer.py:183-190）
num_classes = 3
class_counts = np.bincount(labels_0based, minlength=num_classes)
print(f"\nclass_counts: {class_counts}")

weights = 1.0 / (class_counts + 1e-6)
weights_sum = weights / weights.sum() * num_classes
print(f"\n当前权重计算方法 (sum 归一化):")
print(f"  原始权重 (1/count): {weights}")
print(f"  归一化后 (weights / sum * 3): {weights_sum}")

# 正确的权重计算方法（mean 归一化）
weights_mean = weights / weights.mean()
print(f"\n正确的权重计算方法 (mean 归一化):")
print(f"  归一化后 (weights / mean): {weights_mean}")

# 分析问题
print("\n" + "="*60)
print("问题分析")
print("="*60)

print(f"\n当前方法的结果:")
print(f"  类别 0 (原始类别1): 权重={weights_sum[0]:.6f}")
print(f"  类别 1 (原始类别2): 权重={weights_sum[1]:.6f}")
print(f"  类别 2 (原始类别3): 权重={weights_sum[2]:.6f}")

print(f"\n正确方法的结果:")
print(f"  类别 0 (原始类别1): 权重={weights_mean[0]:.6f}")
print(f"  类别 1 (原始类别2): 权重={weights_mean[1]:.6f}")
print(f"  类别 2 (原始类别3): 权重={weights_mean[2]:.6f}")

print(f"\n权重比 (当前方法):")
print(f"  多数类(2) vs 少数类(1): {weights_sum[1]/weights_sum[0]:.2f}x")
print(f"  多数类(2) vs 少数类(3): {weights_sum[1]/weights_sum[2]:.2f}x")

print(f"\n权重比 (正确方法):")
print(f"  多数类(2) vs 少数类(1): {weights_mean[1]/weights_mean[0]:.2f}x")
print(f"  多数类(2) vs 少数类(3): {weights_mean[1]/weights_mean[2]:.2f}x")

print("\n" + "="*60)
print("结论")
print("="*60)
print("\n当前方法使用 sum() 归一化，导致:")
print("  - 所有权重之和等于 num_classes")
print("  - 多数类获得更大的权重！")
print("  - 这与类别权重的目的完全相反！")
print("\n正确方法应该使用 mean() 归一化，使得:")
print("  - 平均权重为 1")
print("  - 少数类获得更大的权重")
print("  - 这样可以平衡类别不平衡")
