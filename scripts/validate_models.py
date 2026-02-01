# -*- coding: utf-8 -*-
"""
模型验证脚本

分析两个模型的泛化效果，验证是否存在类别不均衡导致的指标虚高问题。

运行方式：
    python scripts/validate_models.py --data data/processed/processed_data.pkl

输出：
    - outputs/validation/class_distribution.json
    - outputs/validation/confusion_matrices.png
    - outputs/validation/task_condition_analysis.json
    - outputs/validation/validation_report.md
"""

import os
import sys
import json
import pickle
import argparse
from collections import defaultdict
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.split_strategy import TwoByTwoSplitter


def load_processed_data(data_path: str):
    """加载预处理数据"""
    print(f'加载预处理数据: {data_path}')
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    print(f'已加载 {len(data)} 个被试')
    return data


def analyze_class_distribution(splits: dict) -> dict:
    """
    分析各划分的类别分布

    Returns:
        分布统计字典
    """
    distribution = {}

    for split_name, split_data in splits.items():
        categories = [d.get('category', 2) for d in split_data]

        # 统计各类别数量
        class_counts = {}
        for cls in [1, 2, 3]:
            class_counts[f'class_{cls}'] = categories.count(cls)

        class_counts['total'] = len(categories)

        # 计算比例
        if len(categories) > 0:
            for cls in [1, 2, 3]:
                class_counts[f'class_{cls}_ratio'] = round(
                    class_counts[f'class_{cls}'] / len(categories), 3
                )

        distribution[split_name] = class_counts

        print(f"\n{split_name}:")
        print(f"  Class 1: {class_counts['class_1']} ({class_counts.get('class_1_ratio', 0):.1%})")
        print(f"  Class 2: {class_counts['class_2']} ({class_counts.get('class_2_ratio', 0):.1%})")
        print(f"  Class 3: {class_counts['class_3']} ({class_counts.get('class_3_ratio', 0):.1%})")
        print(f"  总计: {class_counts['total']}")

    return distribution


def analyze_task_conditions(data: list, splits: dict) -> dict:
    """
    分析训练任务和测试任务的条件差异

    Returns:
        任务条件分析结果
    """
    # 收集所有任务的条件
    task_conditions = {}

    for subject in data:
        for task in subject.get('tasks', []):
            task_id = task.get('task_id', 'unknown')
            if task_id not in task_conditions:
                # 尝试从片段中获取任务条件
                conditions = task.get('task_conditions', {})
                if not conditions and 'segments' in task and len(task['segments']) > 0:
                    seg = task['segments'][0]
                    conditions = seg.get('task_conditions', {})
                task_conditions[task_id] = conditions

    # 划分训练任务和测试任务
    sorted_task_ids = sorted(task_conditions.keys())
    train_task_ids = set(sorted_task_ids[:20])
    test_task_ids = set(sorted_task_ids[20:])

    print(f"\n训练任务 (前20题): {sorted(train_task_ids)}")
    print(f"测试任务 (后{len(test_task_ids)}题): {sorted(test_task_ids)}")

    # 分析条件分布
    def summarize_conditions(task_ids):
        summary = defaultdict(list)
        for tid in task_ids:
            cond = task_conditions.get(tid, {})
            for key, value in cond.items():
                summary[key].append(value)

        result = {}
        for key, values in summary.items():
            if len(values) > 0:
                if isinstance(values[0], (int, float)):
                    result[key] = {
                        'mean': round(np.mean(values), 3),
                        'std': round(np.std(values), 3),
                        'values': values
                    }
                else:
                    result[key] = {'values': values}
        return result

    train_summary = summarize_conditions(train_task_ids)
    test_summary = summarize_conditions(test_task_ids)

    print("\n任务条件分析:")
    print("=" * 50)

    all_keys = set(train_summary.keys()) | set(test_summary.keys())
    for key in sorted(all_keys):
        train_val = train_summary.get(key, {})
        test_val = test_summary.get(key, {})

        if 'mean' in train_val:
            print(f"{key}:")
            print(f"  训练任务: mean={train_val.get('mean', 'N/A')}, std={train_val.get('std', 'N/A')}")
            print(f"  测试任务: mean={test_val.get('mean', 'N/A')}, std={test_val.get('std', 'N/A')}")
        else:
            print(f"{key}:")
            print(f"  训练任务: {train_val.get('values', 'N/A')}")
            print(f"  测试任务: {test_val.get('values', 'N/A')}")

    return {
        'train_tasks': list(train_task_ids),
        'test_tasks': list(test_task_ids),
        'train_conditions': {k: v.get('values', []) for k, v in train_summary.items()},
        'test_conditions': {k: v.get('values', []) for k, v in test_summary.items()},
    }


def analyze_hierarchical_predictions(results_path: str) -> dict:
    """
    分析层级模型的预测结果

    Returns:
        预测分析结果
    """
    if not os.path.exists(results_path):
        print(f"层级模型结果文件不存在: {results_path}")
        return {}

    with open(results_path) as f:
        results = json.load(f)

    predictions = results.get('predictions', {})

    # 按划分分组
    split_preds = defaultdict(lambda: {'true': [], 'predicted': []})

    for subject_id, pred_info in predictions.items():
        split_name = pred_info.get('split', 'unknown')
        split_preds[split_name]['true'].append(pred_info.get('true_class', 0))
        split_preds[split_name]['predicted'].append(pred_info.get('predicted_class', 0))

    # 分析每个划分
    analysis = {}

    for split_name in ['test1', 'test2', 'test3']:
        if split_name not in split_preds:
            continue

        true_labels = split_preds[split_name]['true']
        pred_labels = split_preds[split_name]['predicted']

        # 计算混淆矩阵
        cm = np.zeros((3, 3), dtype=int)
        for t, p in zip(true_labels, pred_labels):
            if 1 <= t <= 3 and 1 <= p <= 3:
                cm[t-1, p-1] += 1

        # 计算每个类别的召回率
        recalls = {}
        for cls in range(3):
            total = cm[cls, :].sum()
            if total > 0:
                recalls[f'class_{cls+1}_recall'] = round(cm[cls, cls] / total, 3)
            else:
                recalls[f'class_{cls+1}_recall'] = 0.0

        # 检测预测偏斜
        pred_counts = {}
        for cls in [1, 2, 3]:
            pred_counts[f'predicted_class_{cls}'] = pred_labels.count(cls)

        analysis[split_name] = {
            'confusion_matrix': cm.tolist(),
            'recalls': recalls,
            'prediction_distribution': pred_counts,
            'total_samples': len(true_labels)
        }

        print(f"\n层级模型 {split_name} 分析:")
        print(f"  混淆矩阵:\n{cm}")
        print(f"  类别召回率: {recalls}")
        print(f"  预测分布: {pred_counts}")

    return analysis


def plot_confusion_matrices(hier_analysis: dict, output_dir: str):
    """绘制混淆矩阵"""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, split_name in enumerate(['test1', 'test2', 'test3']):
        if split_name not in hier_analysis:
            continue

        cm = np.array(hier_analysis[split_name]['confusion_matrix'])

        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Class 1', 'Class 2', 'Class 3'],
                    yticklabels=['Class 1', 'Class 2', 'Class 3'])
        ax.set_title(f'层级模型 - {split_name}')
        ax.set_xlabel('预测类别')
        ax.set_ylabel('真实类别')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), dpi=150)
    plt.close()
    print(f"\n混淆矩阵已保存到: {output_dir}/confusion_matrices.png")


def generate_report(
    class_distribution: dict,
    task_analysis: dict,
    hier_analysis: dict,
    output_dir: str
):
    """生成验证报告"""
    report = []
    report.append("# 模型验证报告")
    report.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 类别分布
    report.append("## 一、类别分布分析\n")
    report.append("| 划分 | Class 1 | Class 2 | Class 3 | 总计 |")
    report.append("|-----|---------|---------|---------|------|")

    for split_name in ['train', 'test1', 'test2', 'test3']:
        dist = class_distribution.get(split_name, {})
        report.append(
            f"| {split_name} | "
            f"{dist.get('class_1', 0)} ({dist.get('class_1_ratio', 0):.1%}) | "
            f"{dist.get('class_2', 0)} ({dist.get('class_2_ratio', 0):.1%}) | "
            f"{dist.get('class_3', 0)} ({dist.get('class_3_ratio', 0):.1%}) | "
            f"{dist.get('total', 0)} |"
        )

    # 层级模型分析
    report.append("\n## 二、层级模型预测分析\n")

    for split_name in ['test1', 'test2', 'test3']:
        if split_name not in hier_analysis:
            continue

        analysis = hier_analysis[split_name]
        report.append(f"### {split_name}\n")

        # 预测分布
        pred_dist = analysis.get('prediction_distribution', {})
        report.append(f"**预测分布**: Class 1={pred_dist.get('predicted_class_1', 0)}, "
                      f"Class 2={pred_dist.get('predicted_class_2', 0)}, "
                      f"Class 3={pred_dist.get('predicted_class_3', 0)}")

        # 类别召回率
        recalls = analysis.get('recalls', {})
        report.append(f"\n**类别召回率**:")
        report.append(f"- Class 1: {recalls.get('class_1_recall', 0):.1%}")
        report.append(f"- Class 2: {recalls.get('class_2_recall', 0):.1%}")
        report.append(f"- Class 3: {recalls.get('class_3_recall', 0):.1%}")
        report.append("")

    # 关键发现
    report.append("\n## 三、关键发现\n")

    # 检查test2是否全部预测为class 1
    if 'test2' in hier_analysis:
        test2 = hier_analysis['test2']
        pred_dist = test2.get('prediction_distribution', {})
        total = test2.get('total_samples', 1)
        class1_ratio = pred_dist.get('predicted_class_1', 0) / total

        if class1_ratio > 0.8:
            report.append(f"⚠️ **层级模型test2塌缩**: {class1_ratio:.1%}的样本被预测为Class 1，"
                          "模型在新任务上完全失效。")

        recalls = test2.get('recalls', {})
        if recalls.get('class_2_recall', 1) < 0.1 and recalls.get('class_3_recall', 1) < 0.1:
            report.append("⚠️ **严重召回问题**: Class 2和Class 3的召回率接近0，模型只能识别Class 1。")

    # 类别不均衡检查
    train_dist = class_distribution.get('train', {})
    if train_dist:
        ratios = [
            train_dist.get('class_1_ratio', 0),
            train_dist.get('class_2_ratio', 0),
            train_dist.get('class_3_ratio', 0)
        ]
        max_ratio = max(ratios)
        min_ratio = min([r for r in ratios if r > 0] or [0])

        if max_ratio / (min_ratio or 1) > 3:
            report.append(f"⚠️ **类别不均衡**: 训练集类别比例差异较大 "
                          f"(最大{max_ratio:.1%}, 最小{min_ratio:.1%})，"
                          "建议使用加权损失函数。")

    # 保存报告
    report_path = os.path.join(output_dir, 'validation_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    print(f"\n验证报告已保存到: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='模型验证脚本')
    parser.add_argument('--data', type=str, default='data/processed/processed_data.pkl',
                        help='预处理数据路径')
    parser.add_argument('--hier-results', type=str,
                        default='outputs/dl_models/experiment_results.json',
                        help='层级模型结果路径')
    parser.add_argument('--output', type=str, default='outputs/validation',
                        help='输出目录')
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("模型验证分析")
    print("=" * 60)

    # 1. 加载数据
    if not os.path.exists(args.data):
        print(f"错误: 数据文件不存在: {args.data}")
        return

    data = load_processed_data(args.data)

    # 2. 执行2x2划分
    splitter = TwoByTwoSplitter(train_subjects=100, train_tasks=20)
    splits = splitter.split(data)

    # 3. 类别分布分析
    print("\n" + "=" * 60)
    print("类别分布分析")
    print("=" * 60)
    class_distribution = analyze_class_distribution(splits)

    # 保存类别分布
    with open(os.path.join(args.output, 'class_distribution.json'), 'w') as f:
        json.dump(class_distribution, f, indent=2)

    # 4. 任务条件分析
    print("\n" + "=" * 60)
    print("任务条件分析")
    print("=" * 60)
    task_analysis = analyze_task_conditions(data, splits)

    # 保存任务条件分析
    with open(os.path.join(args.output, 'task_condition_analysis.json'), 'w') as f:
        json.dump(task_analysis, f, indent=2, default=str)

    # 5. 层级模型预测分析
    print("\n" + "=" * 60)
    print("层级模型预测分析")
    print("=" * 60)
    hier_analysis = analyze_hierarchical_predictions(args.hier_results)

    # 保存层级模型分析
    with open(os.path.join(args.output, 'hierarchical_analysis.json'), 'w') as f:
        json.dump(hier_analysis, f, indent=2)

    # 6. 绘制混淆矩阵
    if hier_analysis:
        try:
            plot_confusion_matrices(hier_analysis, args.output)
        except Exception as e:
            print(f"绘制混淆矩阵失败: {e}")

    # 7. 生成验证报告
    generate_report(class_distribution, task_analysis, hier_analysis, args.output)

    print("\n" + "=" * 60)
    print("验证完成！")
    print(f"输出目录: {args.output}")
    print("=" * 60)


if __name__ == '__main__':
    main()
