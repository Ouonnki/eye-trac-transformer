# -*- coding: utf-8 -*-
"""
标签值诊断脚本

验证数据集中的标签值是否正确（应该是 0/1/2 而不是 1/2/3）
"""

import json
import pickle
from pathlib import Path

import numpy as np


def check_experiment_labels():
    """检查实验结果中的标签值"""
    print("=" * 80)
    print("检查实验结果中的标签值")
    print("=" * 80)

    # 读取实验结果
    exp_path = Path('outputs/dl_models/classification_20260130_164927/experiment_results.json')
    if not exp_path.exists():
        print("实验结果不存在")
        return

    with open(exp_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    predictions = data.get('predictions', {})

    # 收集所有真实标签
    true_labels = []
    pred_labels = []

    for subj_id, pred_data in predictions.items():
        true_cls = pred_data['true_class']
        pred_cls = pred_data['predicted_class']
        true_labels.append(true_cls)
        pred_labels.append(pred_cls)

    print(f"\n标签值范围检查:")
    print(f"  真实标签: min={min(true_labels)}, max={max(true_labels)}, unique={sorted(set(true_labels))}")
    print(f"  预测标签: min={min(pred_labels)}, max={max(pred_labels)}, unique={sorted(set(pred_labels))}")

    # 检查是否是 0-based 还是 1-based
    if min(true_labels) == 0:
        print(f"\n✅ 标签是 0-based (0, 1, 2)")
    elif min(true_labels) == 1:
        print(f"\n⚠️  标签是 1-based (1, 2, 3)")
        print(f"   这可能是问题所在！CrossEntropyLoss 期望 0-based 标签")

    # 如果标签是 1-based，分析对类别权重的影响
    if min(true_labels) == 1:
        print(f"\n类别权重分析（假设标签是 1-based）:")
        counts = np.bincount(true_labels)
        print(f"  bincount 结果: {counts}")
        print(f"  实际类别分布: 类别1={counts[1]}, 类别2={counts[2]}, 类别3={counts[3] if len(counts) > 3 else 0}")

        # 模拟权重计算（假设 minlenth=3）
        print(f"\n模拟类别权重计算（minlength=3）:")
        truncated = counts[:3]  # bincount with minlength=3
        print(f"  截断后的计数: {truncated}")
        weights = 1.0 / (truncated + 1e-6)
        weights = weights / weights.mean()
        print(f"  计算的权重: {weights}")
        print(f"  ⚠️  注意: 类别3的样本被忽略了！")


def check_model_output_classes():
    """检查模型输出的类别数"""
    print(f"\n" + "=" * 80)
    print("模型输出类别数检查")
    print("=" * 80)

    # 检查配置
    config_path = Path('configs/hitask.json')
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        num_classes = config.get('task', {}).get('num_classes', 3)
        print(f"\n配置的 num_classes: {num_classes}")

        print(f"\n模型输出维度 vs 标签范围:")
        print(f"  模型输出: {num_classes} 个类别的 logits")
        print(f"  如果标签是 1-based (1,2,3):")
        print(f"    - 标签 3 会越界（最大有效索引是 {num_classes-1}）")
        print(f"    - 这会导致索引错误或错误的预测")


def main():
    check_experiment_labels()
    check_model_output_classes()

    print(f"\n" + "=" * 80)
    print("诊断结论")
    print("=" * 80)

    # 读取实验结果
    exp_path = Path('outputs/dl_models/classification_20260130_164927/experiment_results.json')
    if exp_path.exists():
        with open(exp_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        predictions = data.get('predictions', {})
        true_labels = [p['true_class'] for p in predictions.values()]

        if min(true_labels) == 1:
            print(f"\n⚠️  发现问题: 标签是 1-based (1, 2, 3)")
            print(f"\n可能的影响:")
            print(f"  1. 模型输出类别 0, 1, 2 的 logits")
            print(f"  2. 但标签是 1, 2, 3")
            print(f"  3. 类别3的标签会越界（索引3不存在）")
            print(f"  4. 这可能导致模型倾向于预测类别0或类别1")
            print(f"\n建议修复:")
            print(f"  在 collate_fn 中将标签减1: label = label - 1")
            print(f"  或在数据预处理阶段就将 category 转换为 0-based")
        else:
            print(f"\n✅ 标签格式正确 (0-based)")


if __name__ == '__main__':
    main()
