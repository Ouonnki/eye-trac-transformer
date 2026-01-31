# -*- coding: utf-8 -*-
"""
训练曲线分析脚本

分析实验结果中的训练历史，绘制训练曲线并诊断验证损失上升问题。
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_experiment_results(exp_dir: str) -> Dict:
    """加载实验结果"""
    results_path = Path(exp_dir) / 'experiment_results.json'
    if not results_path.exists():
        raise FileNotFoundError(f"实验结果文件不存在: {results_path}")

    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_all_experiments(base_dir: str = 'outputs/dl_models') -> List[Dict]:
    """查找所有实验目录"""
    base_path = Path(base_dir)
    experiments = []

    for exp_dir in sorted(base_path.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name.startswith('.'):
            continue

        results_file = exp_dir / 'experiment_results.json'
        if results_file.exists():
            try:
                data = json.load(open(results_file, 'r', encoding='utf-8'))
                data['_dir'] = str(exp_dir)
                experiments.append(data)
            except Exception as e:
                print(f"警告: 无法加载 {exp_dir}: {e}")

    return experiments


def analyze_single_experiment(data: Dict) -> Dict:
    """分析单个实验的结果"""
    repeat_results = data.get('repeat_results', [])

    if not repeat_results:
        print(f"警告: 实验 {data.get('_dir')} 没有 repeat_results")
        return None

    # 收集所有重复实验的指标
    train_losses = [r['train']['loss'] for r in repeat_results if 'train' in r]
    train_accs = [r['train']['accuracy'] for r in repeat_results if 'train' in r]
    epochs = [r['train']['epoch'] for r in repeat_results if 'train' in r]

    # 测试集指标
    test1_accs = [r['test1']['accuracy'] for r in repeat_results if 'test1' in r]
    test2_accs = [r['test2']['accuracy'] for r in repeat_results if 'test2' in r]
    test3_accs = [r['test3']['accuracy'] for r in repeat_results if 'test3' in r]

    return {
        'dir': data.get('_dir'),
        'timestamp': data.get('experiment_info', {}).get('timestamp', 'N/A'),
        'train_loss_mean': np.mean(train_losses) if train_losses else None,
        'train_loss_std': np.std(train_losses) if train_losses else None,
        'train_acc_mean': np.mean(train_accs) if train_accs else None,
        'epoch_mean': np.mean(epochs) if epochs else None,
        'test1_acc_mean': np.mean(test1_accs) if test1_accs else None,
        'test2_acc_mean': np.mean(test2_accs) if test2_accs else None,
        'test3_acc_mean': np.mean(test3_accs) if test3_accs else None,
    }


def compare_experiments(experiments: List[Dict]) -> None:
    """比较多个实验的结果"""
    print("\n" + "=" * 80)
    print("实验结果对比分析")
    print("=" * 80)

    analyzed = [analyze_single_experiment(exp) for exp in experiments]
    analyzed = [a for a in analyzed if a is not None]

    if not analyzed:
        print("没有可分析的实验数据")
        return

    # 打印对比表格
    print(f"\n{'实验目录':<40} {'时间':<20} {'Train Loss':<12} {'Train Acc':<10} {'Epoch':<8}")
    print("-" * 100)

    for a in analyzed:
        dir_name = Path(a['dir']).name
        timestamp = a['timestamp'][:19] if a['timestamp'] != 'N/A' else 'N/A'
        print(f"{dir_name:<40} {timestamp:<20} "
              f"{a['train_loss_mean']:.4f}±{a['train_loss_std']:.4f}  "
              f"{a['train_acc_mean']:.3f}      "
              f"{a['epoch_mean']:.1f}")

    # 打印测试集准确率
    print(f"\n{'实验目录':<40} {'Test1 Acc':<12} {'Test2 Acc':<12} {'Test3 Acc':<12}")
    print("-" * 80)
    for a in analyzed:
        dir_name = Path(a['dir']).name
        print(f"{dir_name:<40} "
              f"{a['test1_acc_mean']:.3f}        "
              f"{a['test2_acc_mean']:.3f}        "
              f"{a['test3_acc_mean']:.3f}")

    # 分析趋势
    print("\n" + "=" * 80)
    print("趋势分析")
    print("=" * 80)

    if len(analyzed) >= 2:
        first = analyzed[0]
        last = analyzed[-1]

        loss_change = ((last['train_loss_mean'] - first['train_loss_mean']) /
                       first['train_loss_mean'] * 100)
        acc_change = ((last['train_acc_mean'] - first['train_acc_mean']) /
                      first['train_acc_mean'] * 100)

        print(f"\n从首次实验到最新实验:")
        print(f"  训练 Loss 变化: {first['train_loss_mean']:.4f} → {last['train_loss_mean']:.4f} "
              f"({loss_change:+.1f}%)")
        print(f"  训练准确率变化: {first['train_acc_mean']:.3f} → {last['train_acc_mean']:.3f} "
              f"({acc_change:+.1f}%)")
        print(f"  训练轮数变化: {first['epoch_mean']:.1f} → {last['epoch_mean']:.1f}")

        if loss_change > 10:
            print(f"\n⚠️  警告: 训练 Loss 上升了 {loss_change:.1f}%，这可能是过拟合的迹象")
        if acc_change > 0 and loss_change > 0:
            print(f"⚠️  警告: 准确率提升但 Loss 上升，可能存在类别不平衡问题")


def analyze_predictions(exp_dir: str) -> Dict:
    """分析预测结果的类别分布"""
    results = load_experiment_results(exp_dir)
    predictions = results.get('predictions', {})

    if not predictions:
        print("警告: 没有预测结果")
        return None

    # 统计预测类别分布
    pred_classes = [p['predicted_class'] for p in predictions.values()]
    true_classes = [p['true_class'] for p in predictions.values()]

    pred_dist = np.bincount(pred_classes, minlength=4)[1:]  # 类别1-3
    true_dist = np.bincount(true_classes, minlength=4)[1:]

    # 计算每个类别的准确率
    class_correct = {1: 0, 2: 0, 3: 0}
    class_total = {1: 0, 2: 0, 3: 0}

    for p in predictions.values():
        true_cls = p['true_class']
        class_total[true_cls] += 1
        if p['correct']:
            class_correct[true_cls] += 1

    class_acc = {c: class_correct[c] / class_total[c] if class_total[c] > 0 else 0
                 for c in [1, 2, 3]}

    return {
        'pred_distribution': pred_dist,
        'true_distribution': true_dist,
        'class_accuracy': class_acc,
        'class_total': class_total,
    }


def plot_prediction_distribution(exp_dir: str, output_path: Optional[str] = None) -> None:
    """绘制预测类别分布"""
    analysis = analyze_predictions(exp_dir)
    if analysis is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 类别分布对比
    x = ['类别1', '类别2', '类别3']
    width = 0.35

    axes[0].bar([i - width/2 for i in range(3)], analysis['true_distribution'],
                width, label='真实分布', alpha=0.8)
    axes[0].bar([i + width/2 for i in range(3)], analysis['pred_distribution'],
                width, label='预测分布', alpha=0.8)
    axes[0].set_xlabel('类别')
    axes[0].set_ylabel('样本数量')
    axes[0].set_title('真实类别 vs 预测类别分布')
    axes[0].set_xticks(range(3))
    axes[0].set_xticklabels(x)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # 各类别准确率
    class_acc = analysis['class_accuracy']
    class_total = analysis['class_total']

    colors = ['#2ecc71' if class_acc[c] > 0.5 else '#e74c3c' for c in [1, 2, 3]]
    bars = axes[1].bar(x, [class_acc[c] for c in [1, 2, 3]], color=colors, alpha=0.8)

    # 添加样本数标注
    for i, (c, bar) in enumerate(zip([1, 2, 3], bars)):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2, height + 0.02,
                     f"n={class_total[c]}", ha='center', fontsize=9)

    axes[1].set_xlabel('类别')
    axes[1].set_ylabel('准确率')
    axes[1].set_title('各类别准确率')
    axes[1].set_ylim(0, 1.1)
    axes[1].axhline(y=1/3, color='gray', linestyle='--', alpha=0.5, label='随机基线')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存到: {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    """主函数"""
    print("眼动追踪 Transformer - 训练曲线分析")
    print("=" * 80)

    # 查找所有实验
    experiments = find_all_experiments()
    print(f"\n找到 {len(experiments)} 个实验结果")

    if not experiments:
        print("没有找到任何实验结果")
        return

    # 比较所有实验
    compare_experiments(experiments)

    # 分析最新实验的预测分布
    latest_exp = max(experiments, key=lambda x: x.get('experiment_info', {}).get('timestamp', ''))
    exp_dir = latest_exp['_dir']
    print(f"\n分析最新实验: {Path(exp_dir).name}")

    output_dir = Path(exp_dir).parent / 'analysis'
    output_dir.mkdir(exist_ok=True)

    plot_prediction_distribution(
        exp_dir,
        output_path=output_dir / 'prediction_distribution.png'
    )

    # 输出分析报告
    report_path = output_dir / 'analysis_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 训练分析报告\n\n")
        f.write("## 实验对比\n\n")

        analyzed = [analyze_single_experiment(exp) for exp in experiments]
        analyzed = [a for a in analyzed if a is not None]

        for a in analyzed:
            dir_name = Path(a['dir']).name
            f.write(f"### {dir_name}\n")
            f.write(f"- 训练 Loss: {a['train_loss_mean']:.4f} ± {a['train_loss_std']:.4f}\n")
            f.write(f"- 训练准确率: {a['train_acc_mean']:.3f}\n")
            f.write(f"- 最佳 Epoch: {a['epoch_mean']:.1f}\n")
            f.write(f"- Test1 准确率: {a['test1_acc_mean']:.3f}\n")
            f.write(f"- Test2 准确率: {a['test2_acc_mean']:.3f}\n")
            f.write(f"- Test3 准确率: {a['test3_acc_mean']:.3f}\n\n")

        pred_analysis = analyze_predictions(exp_dir)
        if pred_analysis:
            f.write("## 预测分析\n\n")
            f.write("### 类别分布\n")
            f.write("| 类别 | 真实分布 | 预测分布 |\n")
            f.write("|------|---------|---------|\n")
            for i, c in enumerate([1, 2, 3], 1):
                f.write(f"| {c} | {pred_analysis['true_distribution'][i-1]} "
                        f"| {pred_analysis['pred_distribution'][i-1]} |\n")

            f.write("\n### 各类别准确率\n")
            f.write("| 类别 | 准确率 | 样本数 |\n")
            f.write("|------|--------|--------|\n")
            for c in [1, 2, 3]:
                f.write(f"| {c} | {pred_analysis['class_accuracy'][c]:.3f} "
                        f"| {pred_analysis['class_total'][c]} |\n")

    print(f"\n分析报告已保存到: {report_path}")


if __name__ == '__main__':
    main()
