# -*- coding: utf-8 -*-
"""
任务级模型训练脚本
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
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import spearmanr

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.task_level_model import TaskLevelEncoder
from src.models.task_level_dataset import TaskLevelGazeDataset, TaskLevelSequenceConfig, task_level_collate_fn
from src.models.task_level_trainer import TaskLevelTrainer

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def str2bool(value):
    """解析命令行布尔值"""
    if isinstance(value, bool):
        return value
    val = value.lower()
    if val in {'true', '1', 'yes', 'y', 't'}:
        return True
    if val in {'false', '0', 'no', 'n', 'f'}:
        return False
    raise argparse.ArgumentTypeError(f'Invalid boolean value: {value}')


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='任务级模型训练脚本',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/task_level.json',
        help='配置文件路径',
    )
    parser.add_argument(
        '--use-task-embedding',
        type=str2bool,
        default=None,
        help='覆盖配置中的 use_task_embedding（true/false）',
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='覆盖配置中的 experiment.name',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='覆盖配置中的 experiment.output_dir',
    )
    return parser.parse_args()


def plot_training_curves(history: dict, output_path: Path, early_stop_epoch: int = None, early_stop_metric: str = 'f1_macro'):
    """绘制训练曲线（改进版：同时显示Weighted和Macro F1）"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training History', fontsize=14, fontweight='bold')
    
    epochs = range(1, len(history['train_loss']) + 1)
    colors = {'train': '#1f77b4', 'val': '#ff7f0e', 'val_macro': '#d62728'}
    
    # 1. Loss Curve
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], color=colors['train'], linewidth=1.5, label='Train')
    ax.plot(epochs, history['val_loss'], color=colors['val'], linewidth=1.5, label='Val')
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Loss', fontsize=10)
    ax.set_title('Loss', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(epochs))
    
    # 2. Accuracy Curve
    ax = axes[0, 1]
    ax.plot(epochs, history['train_acc'], color=colors['train'], linewidth=1.5, label='Train')
    ax.plot(epochs, history['val_acc'], color=colors['val'], linewidth=1.5, label='Val')
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Accuracy', fontsize=10)
    ax.set_title('Accuracy', fontsize=11, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(epochs))
    ax.set_ylim(0, 1)
    
    # 3. F1 (Weighted) Curve
    ax = axes[1, 0]
    ax.plot(epochs, history['train_f1'], color=colors['train'], linewidth=1.5, label='Train')
    ax.plot(epochs, history['val_f1'], color=colors['val'], linewidth=1.5, label='Val')
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('F1 Score (Weighted)', fontsize=10)
    ax.set_title('F1 Score (Weighted)', fontsize=11, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(epochs))
    ax.set_ylim(0, 1)
    
    # 4. F1 (Macro) Curve - 最重要的指标
    ax = axes[1, 1]
    ax.plot(epochs, history['train_f1_macro'], color=colors['train'], linewidth=1.5, label='Train')
    ax.plot(epochs, history['val_f1_macro'], color=colors['val'], linewidth=2, label='Val (Weighted)')
    ax.plot(epochs, history['val_f1_macro'], color=colors['val_macro'], linewidth=1.5, 
            linestyle='--', label='Val (Macro)')
    
    # 标记最佳epoch
    if 'val_f1_macro' in history and len(history['val_f1_macro']) > 0:
        best_epoch = np.argmax(history['val_f1_macro']) + 1
        best_f1 = max(history['val_f1_macro'])
        ax.axvline(x=best_epoch, color='green', linestyle=':', alpha=0.7, label=f'Best ({best_epoch})')
        ax.scatter([best_epoch], [best_f1], color='green', s=100, zorder=5, marker='*')
        ax.annotate(f'{best_f1:.3f}', xy=(best_epoch, best_f1), 
                   xytext=(best_epoch+2, best_f1+0.05), fontsize=9, color='green')
    
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('F1 Score', fontsize=10)
    ax.set_title(f'F1 Score (Macro) - Early Stop Metric', fontsize=11, fontweight='bold', color='darkred')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(epochs))
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"训练曲线已保存: {output_path}")


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """计算类别权重"""
    classes, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    weights = total / (len(classes) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def compute_sample_weights(labels: np.ndarray, mode: str = "effective_num", beta: float = 0.999) -> np.ndarray:
    """
    计算样本采样权重用于平衡采样
    
    Args:
        labels: 样本标签数组
        mode: 权重计算模式
            - 'inverse': 简单反比 (1 / count)
            - 'inverse_sqrt': 反比平方根 (1 / sqrt(count))
            - 'effective_num': 有效样本数 (论文推荐，默认)
        beta: effective_num 模式的超参数，越接近1对少数类越友好
    
    Returns:
        每个样本的采样权重
    """
    classes, counts = np.unique(labels, return_counts=True)
    
    # 计算类别权重
    if mode == "inverse":
        # 简单反比
        class_weights = 1.0 / counts
    elif mode == "inverse_sqrt":
        # 反比平方根（比简单反比更温和）
        class_weights = 1.0 / np.sqrt(counts)
    elif mode == "effective_num":
        # 有效样本数 (Class-Balanced Loss 论文)
        # 公式: (1 - beta) / (1 - beta ^ n_k)
        effective_num = (1.0 - np.power(beta, counts)) / (1.0 - beta)
        class_weights = 1.0 / effective_num
    else:
        raise ValueError(f"Unknown balanced sampler mode: {mode}")
    
    # 归一化类别权重
    class_weights = class_weights / class_weights.sum() * len(classes)
    
    # 为每个样本分配权重
    sample_weights = np.array([class_weights[int(label)] for label in labels])
    
    return sample_weights


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
    
    Args:
        dataset: 数据集
        train_subjects: 训练集被试数量(前N人)
        train_tasks: 训练集题目数量(前N题)
        train_val_split: 从train_pool中划分train/val的比例
        seed: 随机种子
    
    Returns:
        train_indices, val_indices, test1_indices, test2_indices, test3_indices
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


class Subset(Dataset):
    """数据集子集"""
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
    
    def __len__(self):
        return len(self.indices)


def main():
    args = parse_args()

    # 加载配置
    config_path = Path(args.config)
    with open(config_path) as f:
        config = json.load(f)

    # 命令行覆盖配置（保持默认行为兼容）
    if args.use_task_embedding is not None:
        config['model']['use_task_embedding'] = args.use_task_embedding
        logger.info(f"命令行覆盖: model.use_task_embedding={args.use_task_embedding}")
    if args.experiment_name:
        config['experiment']['name'] = args.experiment_name
        logger.info(f"命令行覆盖: experiment.name={args.experiment_name}")
    if args.output_dir:
        config['experiment']['output_dir'] = args.output_dir
        logger.info(f"命令行覆盖: experiment.output_dir={args.output_dir}")
    
    set_seed(config['experiment']['random_seed'])
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 创建带时间戳的输出目录 (隔离每次训练)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = config['experiment'].get('name', 'exp')
    output_dir = Path(config['experiment']['output_dir']) / f"{exp_name}_{timestamp}"
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
    
    # 2×2划分数据集（被试 × 题目）
    logger.info("2×2划分数据集...")
    train_indices, val_indices, test1_indices, test2_indices, test3_indices = split_2x2(
        dataset,
        train_subjects=config['experiment']['train_subjects'],
        train_tasks=config['experiment']['train_tasks'],
        train_val_split=config['training']['train_val_split'],
        seed=config['experiment']['random_seed']
    )
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test1_dataset = Subset(dataset, test1_indices)
    test2_dataset = Subset(dataset, test2_indices)
    test3_dataset = Subset(dataset, test3_indices)
    
    # 统计各集合的被试数和样本数
    def get_stats(indices):
        subjects = set([dataset.samples[i]['subject_id'] for i in indices])
        tasks = set([dataset.samples[i]['task_id'] for i in indices])
        return len(subjects), len(tasks), len(indices)
    
    train_s, train_t, train_n = get_stats(train_indices)
    val_s, val_t, val_n = get_stats(val_indices)
    test1_s, test1_t, test1_n = get_stats(test1_indices)
    test2_s, test2_t, test2_n = get_stats(test2_indices)
    test3_s, test3_t, test3_n = get_stats(test3_indices)
    
    logger.info(f"训练集:   {train_s}被试 × {train_t}题 = {train_n}样本")
    logger.info(f"验证集:   {val_s}被试 × {val_t}题 = {val_n}样本")
    logger.info(f"测试集1:  {test1_s}被试 × {test1_t}题 = {test1_n}样本 (新被试+旧题)")
    logger.info(f"测试集2:  {test2_s}被试 × {test2_t}题 = {test2_n}样本 (旧被试+新题)")
    logger.info(f"测试集3:  {test3_s}被试 × {test3_t}题 = {test3_n}样本 (新被试+新题)")
    
    # 计算/配置类别权重
    train_labels = [dataset.samples[i]['label'] for i in train_indices]
    train_dist = np.bincount(train_labels, minlength=3)
    logger.info(f"训练集类别分布: {train_dist}")
    
    if config['training']['use_class_weights']:
        if config['training'].get('class_weights_mode', 'auto') == 'manual':
            # 手动配置权重
            class_weights = torch.tensor(config['training']['class_weights'], dtype=torch.float32)
            logger.info(f"使用手动类别权重: {class_weights}")
        else:
            # 自动计算权重
            class_weights = compute_class_weights(np.array(train_labels))
            logger.info(f"使用自动计算类别权重: {class_weights}")
    else:
        class_weights = None
        logger.info("不使用类别权重")
    logger.info(f"类别权重: {class_weights}")
    
    # 创建DataLoader
    # 检查是否使用平衡采样
    use_balanced_sampler = config['training'].get('use_balanced_sampler', False)
    
    if use_balanced_sampler:
        # 计算样本采样权重
        sampler_mode = config['training'].get('balanced_sampler_mode', 'effective_num')
        sampler_beta = config['training'].get('balanced_sampler_beta', 0.999)
        
        train_labels_arr = np.array(train_labels)
        sample_weights = compute_sample_weights(
            train_labels_arr,
            mode=sampler_mode,
            beta=sampler_beta
        )
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.float32),
            num_samples=len(train_dataset),
            replacement=True
        )
        logger.info(f"使用平衡采样 (mode={sampler_mode}, beta={sampler_beta})")
        logger.info(f"采样权重分布: 类别0={sample_weights[train_labels_arr==0].mean():.4f}, "
                   f"类别1={sample_weights[train_labels_arr==1].mean():.4f}, "
                   f"类别2={sample_weights[train_labels_arr==2].mean():.4f}")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            sampler=sampler,
            collate_fn=task_level_collate_fn,
            num_workers=4,
            pin_memory=True,
        )
    else:
        logger.info("不使用平衡采样 (使用随机打乱)")
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            collate_fn=task_level_collate_fn,
            num_workers=4,
            pin_memory=True,
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        collate_fn=task_level_collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    test1_loader = DataLoader(
        test1_dataset,
        batch_size=config['training']['batch_size'],
        collate_fn=task_level_collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    test2_loader = DataLoader(
        test2_dataset,
        batch_size=config['training']['batch_size'],
        collate_fn=task_level_collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    test3_loader = DataLoader(
        test3_dataset,
        batch_size=config['training']['batch_size'],
        collate_fn=task_level_collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    
    # 创建模型
    logger.info("创建模型...")
    use_task_embedding = config['model'].get('use_task_embedding', True)
    logger.info(f"使用任务嵌入: {use_task_embedding}")
    model = TaskLevelEncoder(
        input_dim=config['sequence']['input_dim'],
        max_seq_len=config['sequence']['max_seq_len'],
        max_segments=config['sequence']['max_segments'],
        segment_d_model=config['model']['segment_d_model'],
        segment_nhead=config['model']['segment_nhead'],
        segment_num_layers=config['model']['segment_num_layers'],
        attention_dim=config['model']['attention_dim'],
        task_embedding_dim=config['model']['task_embedding_dim'],
        use_task_embedding=use_task_embedding,
        dropout=config['model']['dropout'],
        num_classes=config['model']['num_classes'],
    ).to(device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"总参数量: {total_params:,}, 可训练: {trainable_params:,}")
    
    # 创建训练器
    use_focal_loss = config['training'].get('use_focal_loss', False)
    focal_loss_alpha = config['training'].get('focal_loss_alpha', None)
    focal_loss_gamma = config['training'].get('focal_loss_gamma', 2.0)
    
    label_smoothing = config['training'].get('label_smoothing', 0.0)
    
    trainer = TaskLevelTrainer(
        model=model,
        device=device,
        num_classes=config['model']['num_classes'],
        class_weights=class_weights if config['training']['use_class_weights'] else None,
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
        grad_clip=config['training']['grad_clip'],
        use_focal_loss=use_focal_loss,
        focal_loss_alpha=focal_loss_alpha,
        focal_loss_gamma=focal_loss_gamma,
        label_smoothing=label_smoothing,
    )
    
    # 训练循环
    early_stop_metric = config['training'].get('early_stop_metric', 'f1_weighted')
    best_val_metric = 0.0
    patience_counter = 0
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [], 'train_f1_macro': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_f1_macro': []
    }
    
    logger.info(f"早停监控指标: {early_stop_metric}")
    
    logger.info("开始训练...")
    print("\n" + "="*85)
    print(f"{'Epoch':<6} {'Train':<28} {'Val':<28}")
    print(f"       {'Loss':<8}{'Acc':<8}{'F1w|mac':<12} {'Loss':<8}{'Acc':<8}{'F1w|mac':<12}")
    print("="*85)
    
    for epoch in range(1, config['training']['epochs'] + 1):
        # 训练
        train_metrics = trainer.train_epoch(train_loader, epoch, config['training']['epochs'])
        
        # 验证
        val_metrics = trainer.evaluate(val_loader, desc=f"Epoch {epoch}/{config['training']['epochs']} [Val]")
        
        # 记录历史
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['train_f1'].append(train_metrics['f1_weighted'])
        history['train_f1_macro'].append(train_metrics['f1_macro'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1_weighted'])
        history['val_f1_macro'].append(val_metrics['f1_macro'])
        
        # 打印一行结果 (同时显示Weighted F1和Macro F1)
        print(f"{epoch:<6} "
              f"{train_metrics['loss']:<8.4f} {train_metrics['accuracy']:<8.4f} {train_metrics['f1_weighted']:<8.4f}|{train_metrics['f1_macro']:<8.4f}  "
              f"{val_metrics['loss']:<8.4f} {val_metrics['accuracy']:<8.4f} {val_metrics['f1_weighted']:<8.4f}|{val_metrics['f1_macro']:<8.4f}")
        
        # 学习率调整
        trainer.scheduler.step(val_metrics['loss'])
        
        # 早停检查
        current_metric = val_metrics[early_stop_metric]
        if current_metric > best_val_metric:
            best_val_metric = current_metric
            patience_counter = 0
            # 保存最佳模型
            best_model_path = output_dir / 'best_model.pt'
            trainer.save_checkpoint(best_model_path, epoch, best_val_metric)
        else:
            patience_counter += 1
            if patience_counter >= config['training']['patience']:
                print(f"\n早停! {config['training']['patience']}个epoch没有改善 (Best Val {early_stop_metric}: {best_val_metric:.4f})")
                break
    
    print("="*85)
    
    # 加载最佳模型进行测试
    print(f"\n加载最佳模型 (Best Val {early_stop_metric}: {best_val_metric:.4f}) 进行测试...")
    trainer.load_checkpoint(output_dir / 'best_model.pt')
    
    # 定义评估集（包含同分布验证集 + 三个测试分布）
    eval_sets = [
        ("Val (同分布)", val_loader),
        ("Test1 (新被试+旧题)", test1_loader),
        ("Test2 (旧被试+新题)", test2_loader),
        ("Test3 (新被试+新题)", test3_loader),
    ]
    
    print("\n" + "="*85)
    print("评估结果汇总")
    print("="*85)
    print(f"{'数据集':<25} {'Loss':<10} {'Acc':<10} {'F1(Weighted)':<15} {'F1(Macro)':<12} {'Spearman':<10}")
    print("-"*95)
    
    all_eval_results = {}
    
    for name, loader in eval_sets:
        metrics = trainer.evaluate(loader, desc=name)
        
        # 计算 Spearman 等级相关系数
        spearman_corr, spearman_p = spearmanr(metrics['labels'], metrics['predictions'])
        metrics['spearman'] = spearman_corr
        metrics['spearman_p'] = spearman_p
        
        all_eval_results[name] = metrics
        print(f"{name:<25} {metrics['loss']:<10.4f} {metrics['accuracy']:<10.4f} {metrics['f1_weighted']:<15.4f} {metrics['f1_macro']:<12.4f} {metrics['spearman']:<10.4f}")
    
    print("="*95)
    
    # 详细评估报告
    for name, metrics in all_eval_results.items():
        print(f"\n{'='*70}")
        print(f"{name} - 详细报告")
        print("="*70)
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
    
    # 保存评估结果（包含同分布验证集 + 三个测试分布）
    test_results = {}
    for name, metrics in all_eval_results.items():
        test_results[name] = {
            'loss': metrics['loss'],
            'accuracy': metrics['accuracy'],
            'f1_weighted': metrics['f1_weighted'],
            'f1_macro': metrics['f1_macro'],
            'spearman': metrics['spearman'],
            'spearman_p': metrics['spearman_p'],
            'predictions': [int(p) for p in metrics['predictions']],
            'labels': [int(l) for l in metrics['labels']],
            'probabilities': [
                [float(x) for x in prob]
                for prob in metrics.get('probabilities', [])
            ],
        }
    
    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    # 保存配置和历史
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # 绘制并保存训练曲线
    best_epoch = np.argmax(history[f'val_{early_stop_metric}']) + 1 if history[f'val_{early_stop_metric}'] else 0
    plot_training_curves(history, output_dir / 'training_curves.png', best_epoch, early_stop_metric)
    
    logger.info(f"\n训练完成! 结果保存在: {output_dir}")
    logger.info("产出文件:")
    logger.info(f"  - best_model.pt: 最佳模型检查点")
    logger.info(f"  - config.json: 配置文件")
    logger.info(f"  - history.json: 训练历史")
    logger.info(f"  - training_curves.png: 训练曲线图")
    logger.info(f"  - test_results.json: 测试结果")


if __name__ == '__main__':
    main()
