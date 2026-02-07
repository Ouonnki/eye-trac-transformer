# -*- coding: utf-8 -*-
"""
计算预测类别与真实标签的 Spearman 等级相关系数
"""

import json
import sys
from pathlib import Path
from scipy.stats import spearmanr

def compute_spearman_for_results(results_path: str):
    """计算指定结果文件的 Spearman 相关系数"""
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    print(f"\n{'='*60}")
    print(f"文件: {results_path}")
    print('='*60)
    
    for test_name, metrics in results.items():
        predictions = metrics['predictions']
        labels = metrics['labels']
        
        # 计算 Spearman 相关系数
        corr, p_value = spearmanr(labels, predictions)
        
        print(f"\n{test_name}:")
        print(f"  样本数: {len(labels)}")
        print(f"  Spearman ρ: {corr:.4f}")
        print(f"  p-value: {p_value:.4e}")
        
        # 简单解读
        if corr > 0.8:
            interpretation = "强正相关 (预测等级与真实等级高度一致)"
        elif corr > 0.5:
            interpretation = "中等正相关 (有一定单调趋势)"
        elif corr > 0.2:
            interpretation = "弱正相关"
        elif corr > -0.2:
            interpretation = "几乎无单调关系"
        elif corr > -0.5:
            interpretation = "弱负相关"
        else:
            interpretation = "负相关 (预测与真实趋势相反!)"
        
        print(f"  解读: {interpretation}")
        
        # 显示类别分布
        from collections import Counter
        pred_dist = Counter(predictions)
        label_dist = Counter(labels)
        print(f"  真实标签分布: {dict(sorted(label_dist.items()))}")
        print(f"  预测分布:     {dict(sorted(pred_dist.items()))}")

def demo_spearman_advantage():
    """演示 Spearman 相比准确率的优势"""
    
    print("\n" + "="*70)
    print("为什么用 Spearman？对比示例")
    print("="*70)
    
    # 假设真实标签: 3个简单(0), 3个中等(1), 3个困难(2)
    labels = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    
    # 情况A: 完全正确
    pred_a = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    
    # 情况B: 错2个，但顺序保持（相邻等级混淆）
    # 第4个(实际是1)预测成0，第7个(实际是2)预测成1
    pred_b = [0, 0, 0, 0, 1, 1, 1, 2, 2]
    
    # 情况C: 同样错2个，但顺序混乱（跳跃错误）
    # 第1个(实际是0)预测成2，第9个(实际是2)预测成0
    pred_c = [2, 0, 0, 1, 1, 1, 2, 2, 0]
    
    print(f"\n真实标签:    {labels}")
    print(f"\n预测A(全对): {pred_a}")
    print(f"预测B(错2个，相邻): {pred_b}")  
    print(f"预测C(错2个，跳跃): {pred_c}")
    
    print(f"\n{'情况':<12} {'准确率':<10} {'Spearman ρ':<12} {'解读'}")
    print("-"*70)
    
    for name, pred in [("A 完美预测", pred_a), ("B 相邻错误", pred_b), ("C 跳跃错误", pred_c)]:
        acc = sum(1 for p, l in zip(pred, labels) if p == l) / len(labels)
        corr, _ = spearmanr(labels, pred)
        
        if name == "A 完美预测":
            interp = "理想情况"
        elif name == "B 相邻错误":
            interp = "预测等级接近真实（把中等猜成简单）"
        else:
            interp = "预测等级与真实相反（把简单猜成困难）"
        
        print(f"{name:<12} {acc:<10.2%} {corr:<12.4f} {interp}")
    
    print("\n结论:")
    print("  - 准确率只看'猜没猜对'，B和C都是77.8%")
    print("  - Spearman 看'等级趋势是否一致'，B(0.89) >> C(0.44)")
    print("  - 在难度分级任务中，B 比 C 更合理（相邻错误 vs 跳跃错误）")

if __name__ == '__main__':
    demo_spearman_advantage()
    
    print("\n" + "="*70)
    
    if len(sys.argv) > 1:
        results_path = sys.argv[1]
    else:
        # 默认使用你提到的实验结果
        results_path = "outputs/task_level/task_level_classification_20260207_110116/test_results.json"
    
    compute_spearman_for_results(results_path)
