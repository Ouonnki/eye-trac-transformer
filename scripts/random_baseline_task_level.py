# -*- coding: utf-8 -*-
"""
任务级模型随机基线测试脚本

对同分布验证集 + 三个测试集生成随机预测，计算与模型训练时相同的评估指标，
用于与真实模型性能进行对比。

评估指标:
- Loss: 交叉熵损失
- Accuracy: 准确率
- F1 (Weighted): 加权F1分数
- F1 (Macro): 宏平均F1分数
- Spearman: 斯皮尔曼等级相关系数
"""

import os
import sys
import json
import pickle
import random
import logging
import argparse
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from scipy.stats import spearmanr
import torch.nn as nn

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.task_level_dataset import TaskLevelGazeDataset, TaskLevelSequenceConfig, task_level_collate_fn

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Subset(Dataset):
    """数据集子集"""
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
    
    def __len__(self):
        return len(self.indices)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='任务级模型随机基线测试脚本',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/task_level.json',
        help='配置文件路径',
    )
    return parser.parse_args()


def split_2x2(dataset, train_subjects=100, train_tasks=20, train_val_split=0.9, seed=42):
    """
    2×2划分：被试维度 × 题目维度
    
    划分方式：
    - 被试分为：前train_subjects人(训练) / 剩余(测试)
    - 题目分为：前train_tasks题(训练) / 剩余(测试)
    
    产生5个集合：
    - train_pool: 前train_subjects人 × 前train_tasks题 (用于划分train/val)
    - test1: 剩余被试 × 前train_tasks题
    - test2: 前train_subjects人 × 剩余题目
    - test3: 剩余被试 × 剩余题目
    """
    # 获取所有被试ID并排序
    subject_ids = sorted(list(set([s['subject_id'] for s in dataset.samples])))
    
    # 被试分组
    train_subject_ids = subject_ids[:train_subjects]
    test_subject_ids = subject_ids[train_subjects:]
    
    train_subject_set = set(train_subject_ids)
    test_subject_set = set(test_subject_ids)
    
    # 题目分组
    train_task_ids = set(range(1, train_tasks + 1))  # 1-20
    test_task_ids = set(range(train_tasks + 1, 31))   # 21-30
    
    # 2×2划分
    train_pool = []   # 前100人 × 前20题
    test1 = []        # 后57人 × 前20题
    test2 = []        # 前100人 × 后10题
    test3 = []        # 后57人 × 后10题
    
    for i, sample in enumerate(dataset.samples):
        sid = sample['subject_id']
        tid = sample['task_id']
        
        if sid in train_subject_set and tid in train_task_ids:
            train_pool.append(i)
        elif sid in test_subject_set and tid in train_task_ids:
            test1.append(i)
        elif sid in train_subject_set and tid in test_task_ids:
            test2.append(i)
        elif sid in test_subject_set and tid in test_task_ids:
            test3.append(i)
    
    # 从train_pool中按被试划分train/val
    rng = np.random.RandomState(seed)
    train_pool_subjects = sorted(list(set([dataset.samples[i]['subject_id'] for i in train_pool])))
    rng.shuffle(train_pool_subjects)
    
    n_train = int(len(train_pool_subjects) * train_val_split)
    train_subject_set = set(train_pool_subjects[:n_train])
    val_subject_set = set(train_pool_subjects[n_train:])
    
    train_indices = [i for i in train_pool if dataset.samples[i]['subject_id'] in train_subject_set]
    val_indices = [i for i in train_pool if dataset.samples[i]['subject_id'] in val_subject_set]
    
    return train_indices, val_indices, test1, test2, test3


def evaluate_random_baseline(loader, num_classes=3, class_weights=None, device='cpu'):
    """
    评估随机基线模型
    
    Args:
        loader: DataLoader
        num_classes: 类别数量
        class_weights: 类别权重（用于计算带权重的损失）
        device: 设备
    
    Returns:
        metrics: 包含各项评估指标的字典
    """
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0.0
    
    # 损失函数
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    
    # 遍历数据集
    for batch in loader:
        labels = batch['labels'].to(device)
        batch_size = labels.size(0)
        
        # 生成随机预测 (均匀分布)
        random_preds = torch.randint(0, num_classes, (batch_size,), device=device)
        random_probs = torch.nn.functional.one_hot(
            random_preds,
            num_classes=num_classes
        ).float()
        
        # 为了计算损失，生成随机logits（每个类别的概率相等）
        # 使用很小的随机值，让softmax输出接近均匀分布
        random_logits = torch.randn(batch_size, num_classes, device=device) * 0.1
        
        # 计算损失
        loss = criterion(random_logits, labels)
        total_loss += loss.item() * batch_size
        
        # 收集预测和标签
        all_preds.extend(random_preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(random_probs.cpu().numpy())
    
    # 计算指标
    avg_loss = total_loss / len(all_labels)
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    
    # F1分数
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # Spearman等级相关系数
    spearman_corr, spearman_p = spearmanr(all_labels, all_preds)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'spearman': spearman_corr,
        'spearman_p': spearman_p,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
    }


def evaluate_stratified_random(loader, train_class_dist, num_classes=3, device='cpu'):
    """
    评估分层随机基线（按照训练集类别分布进行采样）
    
    Args:
        loader: DataLoader
        train_class_dist: 训练集类别分布（概率）
        num_classes: 类别数量
        device: 设备
    
    Returns:
        metrics: 包含各项评估指标的字典
    """
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0.0
    
    criterion = nn.CrossEntropyLoss()
    
    # 遍历数据集
    for batch in loader:
        labels = batch['labels'].to(device)
        batch_size = labels.size(0)
        
        # 按照训练集类别分布进行随机采样
        random_preds = torch.tensor(
            np.random.choice(num_classes, size=batch_size, p=train_class_dist),
            device=device
        )
        random_probs = torch.nn.functional.one_hot(
            random_preds,
            num_classes=num_classes
        ).float()
        
        # 为了计算损失，生成符合类别分布的logits
        random_logits = torch.randn(batch_size, num_classes, device=device) * 0.1
        # 根据类别分布调整logits偏置
        for i, prob in enumerate(train_class_dist):
            random_logits[:, i] += np.log(prob + 1e-10)
        
        # 计算损失
        loss = criterion(random_logits, labels)
        total_loss += loss.item() * batch_size
        
        # 收集预测和标签
        all_preds.extend(random_preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(random_probs.cpu().numpy())
    
    # 计算指标
    avg_loss = total_loss / len(all_labels)
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    
    # F1分数
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # Spearman等级相关系数
    spearman_corr, spearman_p = spearmanr(all_labels, all_preds)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'spearman': spearman_corr,
        'spearman_p': spearman_p,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
    }


def main():
    args = parse_args()

    # 加载配置
    config_path = Path(args.config)
    with open(config_path) as f:
        config = json.load(f)
    
    # 设置随机种子（保证结果可复现）
    random.seed(config['experiment']['random_seed'])
    np.random.seed(config['experiment']['random_seed'])
    torch.manual_seed(config['experiment']['random_seed'])
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(config['experiment']['output_dir']) / f"random_baseline_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")
    
    # 加载预处理数据
    logger.info("加载预处理数据...")
    with open(config['data']['processed_data_path'], 'rb') as f:
        processed_data = pickle.load(f)
    logger.info(f"加载了 {len(processed_data)} 个被试的数据")
    
    # 创建数据集
    seq_config = TaskLevelSequenceConfig(
        max_seq_len=config['sequence']['max_seq_len'],
        max_segments=config['sequence']['max_segments'],
    )
    dataset = TaskLevelGazeDataset(
        processed_data=processed_data,
        config=seq_config,
        fit_normalizer=True,
    )
    
    # 收集所有标签
    all_labels = np.array([s['label'] for s in dataset.samples])
    logger.info(f"数据集总样本数: {len(dataset)}")
    logger.info(f"类别分布: {np.bincount(all_labels)}")
    
    # 2×2划分数据集
    logger.info("2×2划分数据集...")
    train_indices, val_indices, test1_indices, test2_indices, test3_indices = split_2x2(
        dataset,
        train_subjects=config['experiment']['train_subjects'],
        train_tasks=config['experiment']['train_tasks'],
        train_val_split=config['training']['train_val_split'],
        seed=config['experiment']['random_seed']
    )
    
    val_dataset = Subset(dataset, val_indices)
    test1_dataset = Subset(dataset, test1_indices)
    test2_dataset = Subset(dataset, test2_indices)
    test3_dataset = Subset(dataset, test3_indices)
    
    # 统计各测试集的样本数
    def get_stats(indices):
        subjects = set([dataset.samples[i]['subject_id'] for i in indices])
        tasks = set([dataset.samples[i]['task_id'] for i in indices])
        return len(subjects), len(tasks), len(indices)
    
    val_s, val_t, val_n = get_stats(val_indices)
    test1_s, test1_t, test1_n = get_stats(test1_indices)
    test2_s, test2_t, test2_n = get_stats(test2_indices)
    test3_s, test3_t, test3_n = get_stats(test3_indices)
    
    logger.info(f"验证集(同分布): {val_s}被试 × {val_t}题 = {val_n}样本")
    logger.info(f"测试集1: {test1_s}被试 × {test1_t}题 = {test1_n}样本 (新被试+旧题)")
    logger.info(f"测试集2: {test2_s}被试 × {test2_t}题 = {test2_n}样本 (旧被试+新题)")
    logger.info(f"测试集3: {test3_s}被试 × {test3_t}题 = {test3_n}样本 (新被试+新题)")
    
    # 创建DataLoader
    batch_size = config['training']['batch_size']
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        collate_fn=task_level_collate_fn, num_workers=4, pin_memory=True
    )
    test1_loader = DataLoader(
        test1_dataset, batch_size=batch_size,
        collate_fn=task_level_collate_fn, num_workers=4, pin_memory=True
    )
    test2_loader = DataLoader(
        test2_dataset, batch_size=batch_size,
        collate_fn=task_level_collate_fn, num_workers=4, pin_memory=True
    )
    test3_loader = DataLoader(
        test3_dataset, batch_size=batch_size,
        collate_fn=task_level_collate_fn, num_workers=4, pin_memory=True
    )
    
    # 计算训练集类别分布（用于分层随机基线）
    train_labels = [dataset.samples[i]['label'] for i in train_indices]
    train_dist = np.bincount(train_labels, minlength=3)
    train_class_dist = train_dist / train_dist.sum()
    logger.info(f"训练集类别分布: {train_dist}")
    logger.info(f"训练集类别概率: {train_class_dist}")
    
    # 类别权重
    if config['training']['use_class_weights']:
        class_weights = torch.tensor(config['training']['class_weights'], dtype=torch.float32)
    else:
        class_weights = None
    
    # 定义评估集（同分布验证集 + 三个测试分布）
    eval_sets = [
        ("Val (同分布)", val_loader),
        ("Test1 (新被试+旧题)", test1_loader),
        ("Test2 (旧被试+新题)", test2_loader),
        ("Test3 (新被试+新题)", test3_loader),
    ]
    
    # ==================== 均匀随机基线 ====================
    print("\n" + "="*100)
    print("均匀随机基线 (Uniform Random Baseline)")
    print("预测策略: 均匀随机选择类别 (0, 1, 2)，每个类别概率 = 1/3")
    print("="*100)
    print(f"{'数据集':<25} {'Loss':<10} {'Acc':<10} {'F1(Weighted)':<15} {'F1(Macro)':<12} {'Spearman':<10}")
    print("-"*100)
    
    uniform_results = {}
    for name, loader in eval_sets:
        metrics = evaluate_random_baseline(
            loader, 
            num_classes=config['model']['num_classes'],
            class_weights=class_weights,
            device=device
        )
        uniform_results[name] = metrics
        print(f"{name:<25} {metrics['loss']:<10.4f} {metrics['accuracy']:<10.4f} "
              f"{metrics['f1_weighted']:<15.4f} {metrics['f1_macro']:<12.4f} {metrics['spearman']:<10.4f}")
    
    print("="*100)
    
    # ==================== 分层随机基线 ====================
    print("\n" + "="*100)
    print("分层随机基线 (Stratified Random Baseline)")
    print(f"预测策略: 按照训练集类别分布采样 (类别0={train_class_dist[0]:.3f}, "
          f"类别1={train_class_dist[1]:.3f}, 类别2={train_class_dist[2]:.3f})")
    print("="*100)
    print(f"{'数据集':<25} {'Loss':<10} {'Acc':<10} {'F1(Weighted)':<15} {'F1(Macro)':<12} {'Spearman':<10}")
    print("-"*100)
    
    stratified_results = {}
    for name, loader in eval_sets:
        metrics = evaluate_stratified_random(
            loader,
            train_class_dist=train_class_dist,
            num_classes=config['model']['num_classes'],
            device=device
        )
        stratified_results[name] = metrics
        print(f"{name:<25} {metrics['loss']:<10.4f} {metrics['accuracy']:<10.4f} "
              f"{metrics['f1_weighted']:<15.4f} {metrics['f1_macro']:<12.4f} {metrics['spearman']:<10.4f}")
    
    print("="*100)
    
    # ==================== 详细报告 ====================
    for name, _ in eval_sets:
        print(f"\n{'='*80}")
        print(f"{name} - 均匀随机基线详细报告")
        print("="*80)
        
        metrics = uniform_results[name]
        print(f"样本数: {len(metrics['labels'])}")
        print(f"Loss: {metrics['loss']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 (Weighted): {metrics['f1_weighted']:.4f}")
        print(f"F1 (Macro): {metrics['f1_macro']:.4f}")
        print(f"Spearman ρ: {metrics['spearman']:.4f} (p={metrics['spearman_p']:.4e})")
        print()
        print(classification_report(
            metrics['labels'],
            metrics['predictions'],
            target_names=['Class 0', 'Class 1', 'Class 2'],
            digits=4
        ))
        print("混淆矩阵:")
        print(confusion_matrix(metrics['labels'], metrics['predictions']))
    
    # ==================== 保存结果 ====================
    results = {
        'timestamp': timestamp,
        'config': config,
        'uniform_random': {},
        'stratified_random': {},
    }
    
    for name in uniform_results:
        results['uniform_random'][name] = {
            'loss': uniform_results[name]['loss'],
            'accuracy': uniform_results[name]['accuracy'],
            'f1_weighted': uniform_results[name]['f1_weighted'],
            'f1_macro': uniform_results[name]['f1_macro'],
            'spearman': uniform_results[name]['spearman'],
            'spearman_p': uniform_results[name]['spearman_p'],
            'predictions': [int(p) for p in uniform_results[name]['predictions']],
            'labels': [int(l) for l in uniform_results[name]['labels']],
            'probabilities': [
                [float(x) for x in prob]
                for prob in uniform_results[name].get('probabilities', [])
            ],
        }
        results['stratified_random'][name] = {
            'loss': stratified_results[name]['loss'],
            'accuracy': stratified_results[name]['accuracy'],
            'f1_weighted': stratified_results[name]['f1_weighted'],
            'f1_macro': stratified_results[name]['f1_macro'],
            'spearman': stratified_results[name]['spearman'],
            'spearman_p': stratified_results[name]['spearman_p'],
            'predictions': [int(p) for p in stratified_results[name]['predictions']],
            'labels': [int(l) for l in stratified_results[name]['labels']],
            'probabilities': [
                [float(x) for x in prob]
                for prob in stratified_results[name].get('probabilities', [])
            ],
        }
    
    # 保存为JSON
    with open(output_dir / 'random_baseline_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # 保存为CSV格式（便于对比）
    import csv
    with open(output_dir / 'random_baseline_summary.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Baseline Type', 'Test Set', 'Loss', 'Accuracy', 'F1(Weighted)', 'F1(Macro)', 'Spearman'])
        for name in uniform_results:
            m = uniform_results[name]
            writer.writerow(['Uniform Random', name, m['loss'], m['accuracy'], m['f1_weighted'], m['f1_macro'], m['spearman']])
        for name in stratified_results:
            m = stratified_results[name]
            writer.writerow(['Stratified Random', name, m['loss'], m['accuracy'], m['f1_weighted'], m['f1_macro'], m['spearman']])
    
    logger.info(f"\n随机基线测试完成! 结果保存在: {output_dir}")
    logger.info("产出文件:")
    logger.info(f"  - random_baseline_results.json: 完整结果（含预测和标签）")
    logger.info(f"  - random_baseline_summary.csv: 汇总表格（便于对比）")


if __name__ == '__main__':
    main()
