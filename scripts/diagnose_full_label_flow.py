# -*- coding: utf-8 -*-
"""
完整标签流程诊断

检查训练和预测时标签的完整流程
"""

import json
from pathlib import Path
import numpy as np


def analyze_experiment():
    """分析实验结果"""
    print("="*80)
    print("完整标签流程诊断")
    print("="*80)

    # 读取实验结果
    exp_path = Path('outputs/dl_models/classification_20260130_164927/experiment_results.json')
    if not exp_path.exists():
        print("实验结果不存在")
        return

    with open(exp_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    predictions = data.get('predictions', {})

    # 统计真实和预测的类别分布
    true_labels = []
    pred_labels = []

    for subj_id, pred_data in predictions.items():
        true_labels.append(pred_data['true_class'])
        pred_labels.append(pred_data['predicted_class'])

    true_dist = np.bincount(true_labels, minlength=4)[1:]  # 类别 1-3
    pred_dist = np.bincount(pred_labels, minlength=4)[1:]

    print(f"\n测试集类别分布:")
    print(f"  真实分布: 类别1={true_dist[0]}, 类别2={true_dist[1]}, 类别3={true_dist[2]}")
    print(f"  预测分布: 类别1={pred_dist[0]}, 类别2={pred_dist[1]}, 类别3={pred_dist[2]}")

    # 分析混淆矩阵
    confusion = np.zeros((3, 3), dtype=int)
    for p in predictions.values():
        true_idx = p['true_class'] - 1  # 转为 0-based
        pred_idx = p['predicted_class'] - 1
        if 0 <= true_idx < 3 and 0 <= pred_idx < 3:
            confusion[true_idx, pred_idx] += 1

    print(f"\n混淆矩阵 (真实 x 预测, 0-based索引):")
    print(f"         预:0   预:1   预:2")
    for i in range(3):
        print(f"真:{i}     {confusion[i,0]:>3}     {confusion[i,1]:>3}     {confusion[i,2]:>3}")

    # 转换回 1-based 显示
    print(f"\n混淆矩阵 (真实 x 预测, 1-based标签):")
    print(f"         预:1   预:2   预:3")
    for i in range(3):
        print(f"真:{i+1}     {confusion[i,0]:>3}     {confusion[i,1]:>3}     {confusion[i,2]:>3}")

    # 分析模型输出
    print(f"\n" + "="*80)
    print("模型输出分析")
    print("="*80)

    # 模型输出 3 个类别的 logits (索引 0, 1, 2)
    # 如果标签是 0-based (正确)，则：
    #   真实标签0 → 对应 logit[0] → 存储为类别1
    #   真实标签1 → 对应 logit[1] → 存储为类别2
    #   真实标签2 → 对应 logit[2] → 存储为类别3

    # 预测时 argmax 得到 0, 1, 2，然后 +1 存储为类别 1, 2, 3

    # 从混淆矩阵看：
    #   真实类别1 (索引0): 25个样本 → 全部预测为类别1 (索引0)
    #   真实类别2 (索引1): 111个样本 → 110个预测为类别1 (索引0), 1个预测为类别2 (索引1)
    #   真实类别3 (索引2): 21个样本 → 20个预测为类别1 (索引0), 1个预测为类别2 (索引1)

    print(f"\n从混淆矩阵推断的模型行为:")
    print(f"  模型的 argmax 几乎总是返回索引 0")
    print(f"  这意味着 logit[0] 总是最大的")

    # 计算模型预测的分布（0-based）
    pred_dist_0based = np.bincount([p['predicted_class']-1 for p in predictions.values()], minlength=3)
    print(f"\n模型预测的 argmax 分布 (0-based):")
    print(f"  索引0: {pred_dist_0based[0]} 次 ({pred_dist_0based[0]/sum(pred_dist_0based)*100:.1f}%)")
    print(f"  索引1: {pred_dist_0based[1]} 次 ({pred_dist_0based[1]/sum(pred_dist_0based)*100:.1f}%)")
    print(f"  索引2: {pred_dist_0based[2]} 次 ({pred_dist_0based[2]/sum(pred_dist_0based)*100:.1f}%)")

    print(f"\n" + "="*80)
    print("关键发现")
    print("="*80)

    print(f"\n1. 模型几乎总是预测索引 0 (对应存储的类别1)")
    print(f"2. 这意味着模型学到的决策偏向是: logit[0] >> logit[1] ≈ logit[2]")
    print(f"3. 可能的原因:")

    # 检查是否是标签偏移问题
    print(f"\n  假设1: 标签偏移问题")
    print(f"    如果训练时标签错误地为 1,2,3 而不是 0,1,2:")
    print(f"    - 标签 3 会越界（模型只有索引 0-2）")
    print(f"    - 模型永远学不会正确预测类别3")
    print(f"    - 这符合观察到的现象！")

    print(f"\n  假设2: 类别权重计算错误")
    print(f"    如果权重计算忽略了类别3的样本:")
    print(f"    - 类别3的梯度几乎为零")
    print(f"    - 模型学不会类别3的特征")

    print(f"\n  假设3: 类别1的特征最容易学习")
    print(f"    - 类别1可能是'默认'响应")

    print(f"\n" + "="*80)
    print("建议检查")
    print("="*80)

    print(f"\n1. 检查训练日志中的 '类别权重' 输出")
    print(f"2. 检查 LightweightGazeDataset 返回的标签值")
    print(f"3. 验证 Collate 函数没有修改标签")
    print(f"4. 添加调试代码打印第一个 batch 的标签值")


if __name__ == '__main__':
    analyze_experiment()
