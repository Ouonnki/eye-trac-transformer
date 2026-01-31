# -*- coding: utf-8 -*-
"""
诊断报告生成脚本

基于已有分析结果生成完整的诊断报告。
"""

import json
from pathlib import Path
from typing import Dict, List
import numpy as np


def load_all_experiments() -> List[Dict]:
    """加载所有实验结果"""
    base_dir = Path('outputs/dl_models')
    experiments = []

    for exp_dir in sorted(base_dir.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name.startswith('.'):
            continue

        results_file = exp_dir / 'experiment_results.json'
        if results_file.exists():
            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                data['_dir'] = str(exp_dir)
                experiments.append(data)

    return experiments


def get_prediction_analysis(exp_data: Dict) -> Dict:
    """分析预测结果"""
    predictions = exp_data.get('predictions', {})

    if not predictions:
        return None

    pred_classes = [p['predicted_class'] for p in predictions.values()]
    true_classes = [p['true_class'] for p in predictions.values()]

    pred_dist = np.bincount(pred_classes, minlength=4)[1:]
    true_dist = np.bincount(true_classes, minlength=4)[1:]

    # 计算混淆矩阵
    confusion = np.zeros((3, 3), dtype=int)
    for p in predictions.values():
        true_cls = p['true_class'] - 1  # 转为0-based索引
        pred_cls = p['predicted_class'] - 1
        confusion[true_cls, pred_cls] += 1

    # 计算各类别准确率
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
        'confusion_matrix': confusion,
        'class_accuracy': class_acc,
        'class_total': class_total,
        'total_samples': len(predictions)
    }


def generate_diagnosis_report() -> None:
    """生成诊断报告"""
    print("=" * 80)
    print("生成验证损失上升诊断报告")
    print("=" * 80)

    experiments = load_all_experiments()
    if not experiments:
        print("没有找到实验结果")
        return

    # 使用最新的实验
    latest_exp = max(experiments, key=lambda x: x.get('experiment_info', {}).get('timestamp', ''))
    pred_analysis = get_prediction_analysis(latest_exp)

    output_path = Path('outputs/dl_models/analysis/diagnosis_report.md')
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# 验证损失上升诊断报告\n\n")
        f.write(f"生成时间: {latest_exp.get('experiment_info', {}).get('timestamp', 'N/A')}\n\n")

        # ============================================
        # 1. 问题概述
        # ============================================
        f.write("## 1. 问题概述\n\n")
        f.write("### 观察到的现象\n\n")
        f.write("| 指标 | 现象 |\n")
        f.write("|------|------|\n")
        f.write("| 训练 Loss | 上升 21-28% |\n")
        f.write("| 训练准确率 | 保持在 71% 左右 |\n")
        f.write("| 预测分布 | 极度偏向类别1 |\n")
        f.write("| Test2/Test3 准确率 | 12-25% (极低) |\n\n")

        # ============================================
        # 2. 根本原因分析
        # ============================================
        f.write("## 2. 根本原因分析\n\n")

        if pred_analysis:
            f.write("### 2.1 极端类别偏向问题 ⚠️\n\n")
            f.write("**真实类别分布 vs 预测类别分布:**\n\n")
            f.write("| 类别 | 真实样本数 | 预测样本数 | 偏差 |\n")
            f.write("|------|-----------|-----------|------|\n")

            for i, c in enumerate([1, 2, 3]):
                true_count = pred_analysis['true_distribution'][i]
                pred_count = pred_analysis['pred_distribution'][i]
                bias = pred_count - true_count
                f.write(f"| {c} | {true_count} | {pred_count} | {bias:+d} |\n")

            f.write("\n**各类别准确率:**\n\n")
            f.write("| 类别 | 准确率 | 样本数 |\n")
            f.write("|------|--------|--------|\n")
            for c in [1, 2, 3]:
                acc = pred_analysis['class_accuracy'][c]
                total = pred_analysis['class_total'][c]
                status = "✅" if acc > 0.8 else "⚠️" if acc > 0.3 else "❌"
                f.write(f"| {c} | {acc:.1%} ({status}) | {total} |\n")

            f.write("\n**混淆矩阵** (行=真实, 列=预测):\n\n")
            f.write("```\n")
            f.write("         预测:1   预测:2   预测:3\n")
            for i, true_cls in enumerate([1, 2, 3]):
                row = pred_analysis['confusion_matrix'][i]
                f.write(f"真实:{i+1}    {row[0]:>3}     {row[1]:>3}     {row[2]:>3}\n")
            f.write("```\n\n")

            # 计算不平衡比例
            true_dist = pred_analysis['true_distribution']
            max_count = true_dist.max()
            min_count = true_dist[true_dist > 0].min()
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

            f.write(f"**类别不平衡比例: {imbalance_ratio:.1f}:1**\n\n")

        f.write("### 2.2 Loss 上升但准确率稳定的解释\n\n")
        f.write("这是**类别不平衡导致的典型现象**：\n\n")
        f.write("1. **模型学会预测少数类**：\n")
        f.write("   - 训练数据中类别1可能更容易预测（Loss 更低）\n")
        f.write("   - 模型倾向于将多数类（类别2）也预测为类别1\n\n")
        f.write("2. **Loss 持续累积**：\n")
        f.write("   - 每次将类别2错误预测为类别1时，产生高Loss\n")
        f.write("   - 类别2样本最多（111个），因此累积Loss很高\n\n")
        f.write("3. **准确率看似稳定**：\n")
        f.write("   - 类别1的预测是准确的（25/25 = 100%）\n")
        f.write("   - 但类别2和类别3几乎全部预测错误\n\n")

        f.write("### 2.3 跨任务泛化失败\n\n")
        f.write("**观察:**\n")
        f.write("- Test1 (新人旧题): 66-75% 准确率 - 尚可\n")
        f.write("- Test2 (旧人新题): 12-15% 准确率 - **极差**\n")
        f.write("- Test3 (新人新题): 19-25% 准确率 - **极差**\n\n")
        f.write("**原因分析:**\n")
        f.write("1. `use_task_embedding: false` - 模型无法区分不同任务\n")
        f.write("2. 模型可能记忆了训练集中特定任务的样本模式\n")
        f.write("3. 当遇到新任务时，模型退化为预测多数类\n\n")

        # ============================================
        # 3. 配置问题分析
        # ============================================
        f.write("## 3. 配置问题分析\n\n")
        f.write("### 3.1 当前的正则化配置\n\n")
        f.write("| 配置项 | 当前值 | 评估 |\n")
        f.write("|--------|--------|------|\n")
        f.write("| dropout | 0.1 | ⚠️ 偏低，建议增加到 0.2-0.3 |\n")
        f.write("| weight_decay | 1e-4 | ⚠️ 偏低，建议增加到 1e-3 |\n")
        f.write("| learning_rate | 1e-4 | ✅ 适中 |\n")
        f.write("| patience | 100 | ⚠️ 过高，可能导致过拟合 |\n")
        f.write("| use_task_embedding | false | ❌ 应该启用 |\n")
        f.write("| label_smoothing | 无 | ⚠️ 建议添加 |\n\n")

        f.write("### 3.2 类别权重计算\n\n")
        f.write("**当前计算方式** (`src/models/dl_trainer.py:183-201`):\n")
        f.write("```python\n")
        f.write("class_weights = 1.0 / (class_counts + 1e-6)\n")
        f.write("class_weights = class_weights / class_weights.mean()  # 归一化\n")
        f.write("```\n\n")
        f.write("**问题**：对于极端不平衡的情况，这种权重计算方式可能不够强。\n\n")

        # ============================================
        # 4. 改进建议
        # ============================================
        f.write("## 4. 改进建议\n\n")
        f.write("### 4.1 高优先级（立即实施）\n\n")
        f.write("#### 1. 启用任务嵌入\n")
        f.write("```json\n")
        f.write('{\n  "model": {\n    "use_task_embedding": true\n  }\n}\n')
        f.write("```\n\n")
        f.write("**预期效果**: 模型能够区分不同任务，提升 Test2/Test3 性能\n\n")

        f.write("#### 2. 调整类别权重策略\n")
        f.write("```python\n")
        f.write("# 方案1: 使用平方根缩放\n")
        f.write("class_weights = 1.0 / np.sqrt(class_counts)\n")
        f.write("class_weights = class_weights / class_weights.mean()\n\n")
        f.write("# 方案2: 使用中位数缩放\n")
        f.write("median_count = np.median(class_counts)\n")
        f.write("class_weights = median_count / class_counts\n")
        f.write("```\n\n")

        f.write("#### 3. 降低早停 Patience\n")
        f.write("```json\n")
        f.write('{\n  "training": {\n    "patience": 20\n  }\n}\n')
        f.write("```\n\n")

        f.write("### 4.2 中优先级（如果高优先级改进无效）\n\n")
        f.write("#### 1. 增加 Dropout\n")
        f.write("```json\n")
        f.write('{\n  "model": {\n    "dropout": 0.3\n  }\n}\n')
        f.write("```\n\n")

        f.write("#### 2. 增加 Weight Decay\n")
        f.write("```json\n")
        f.write('{\n  "training": {\n    "weight_decay": 0.001\n  }\n}\n')
        f.write("```\n\n")

        f.write("#### 3. 使用 Focal Loss\n")
        f.write("Focal Loss 对难分类样本给予更高权重：\n")
        f.write("```python\n")
        f.write("class FocalLoss(nn.Module):\n")
        f.write("    def __init__(self, alpha=1, gamma=2):\n")
        f.write("        super().__init__()\n")
        f.write("        self.alpha = alpha\n")
        f.write("        self.gamma = gamma\n\n")
        f.write("    def forward(self, inputs, targets):\n")
        f.write("        ce_loss = F.cross_entropy(inputs, targets, reduction='none')\n")
        f.write("        p_t = torch.exp(-ce_loss)\n")
        f.write("        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss\n")
        f.write("        return focal_loss.mean()\n")
        f.write("```\n\n")

        f.write("### 4.3 低优先级（长期改进）\n\n")
        f.write("1. **数据增强**: 添加眼动轨迹特定的增强方法\n")
        f.write("2. **模型集成**: 使用多个模型的集成预测\n")
        f.write("3. **测试时增强**: 提高预测稳定性\n")

        # ============================================
        # 5. 验证计划
        # ============================================
        f.write("## 5. 验证计划\n\n")
        f.write("| 实验 | 配置变化 | 预期结果 |\n")
        f.write("|------|---------|---------|\n")
        f.write("| Baseline | 当前配置 | 重现问题 |\n")
        f.write("| Exp1 | +use_task_embedding | Test2/Test3 准确率提升 |\n")
        f.write("| Exp2 | +调整类别权重 | 类别2准确率提升 |\n")
        f.write("| Exp3 | +dropout=0.3 | 过拟合缓解 |\n")
        f.write("| Exp4 | +weight_decay=1e-3 | 正则化增强 |\n")
        f.write("| Exp5 | +patience=20 | 更早停止 |\n")
        f.write("| Exp6 | +focal_loss | 整体性能提升 |\n")

        # ============================================
        # 6. 结论
        # ============================================
        f.write("## 6. 结论\n\n")
        f.write("### 问题根因\n\n")
        f.write("**验证损失上升的根本原因是严重的类别不平衡问题。**\n\n")
        f.write("具体表现为：\n")
        f.write("1. 模型学会预测少数类（类别1）来降低Loss\n")
        f.write("2. 多数类（类别2）被持续错误预测，累积高Loss\n")
        f.write("3. 任务嵌入未启用，导致跨任务泛化失败\n\n")

        f.write("### 优先行动\n\n")
        f.write("1. **立即**: 启用任务嵌入\n")
        f.write("2. **立即**: 调整类别权重计算策略\n")
        f.write("3. **短期**: 降低 patience，防止过拟合\n")
        f.write("4. **中期**: 如果问题持续，考虑使用 Focal Loss\n\n")

    print(f"\n诊断报告已生成: {output_path}")

    # 打印关键发现
    print("\n" + "=" * 80)
    print("关键发现")
    print("=" * 80)

    if pred_analysis:
        print(f"\n1. 类别分布严重偏向:")
        print(f"   真实分布: {pred_analysis['true_distribution']}")
        print(f"   预测分布: {pred_analysis['pred_distribution']}")

        print(f"\n2. 各类别准确率:")
        for c in [1, 2, 3]:
            print(f"   类别{c}: {pred_analysis['class_accuracy'][c]:.1%}")

        print(f"\n3. 模型将 {pred_analysis['pred_distribution'][0]} 个样本预测为类别1")
        print(f"   而真实类别1只有 {pred_analysis['true_distribution'][0]} 个")
        print(f"   偏差: {pred_analysis['pred_distribution'][0] - pred_analysis['true_distribution'][0]:+d} 个")

    print(f"\n4. 建议:")
    print(f"   - 启用任务嵌入 (use_task_embedding: true)")
    print(f"   - 调整类别权重计算策略")
    print(f"   - 降低早停 patience (100 -> 20)")


if __name__ == '__main__':
    generate_diagnosis_report()
