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
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix

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


def main():
    # 加载配置
    config_path = Path('configs/task_level.json')
    with open(config_path) as f:
        config = json.load(f)
    
    set_seed(config['experiment']['random_seed'])
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 创建输出目录
    output_dir = Path(config['experiment']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # 收集所有标签用于划分和计算权重
    all_labels = np.array([s['label'] for s in dataset.samples])
    logger.info(f"数据集总样本数: {len(dataset)}")
    logger.info(f"类别分布: {np.bincount(all_labels)}")
    
    # 数据集划分
    total_size = len(dataset)
    train_size = int(total_size * config['data']['train_split'])
    val_size = int(total_size * config['data']['val_split'])
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    logger.info(f"数据集划分: 训练{train_size} / 验证{val_size} / 测试{test_size}")
    
    # 计算类别权重
    train_labels = [dataset.samples[i]['label'] for i in train_dataset.indices]
    class_weights = compute_class_weights(np.array(train_labels))
    logger.info(f"类别权重: {class_weights}")
    
    # 创建DataLoader
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
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        collate_fn=task_level_collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    
    # 创建模型
    logger.info("创建模型...")
    model = TaskLevelEncoder(
        input_dim=config['sequence']['input_dim'],
        max_seq_len=config['sequence']['max_seq_len'],
        max_segments=config['sequence']['max_segments'],
        segment_d_model=config['model']['segment_d_model'],
        segment_nhead=config['model']['segment_nhead'],
        segment_num_layers=config['model']['segment_num_layers'],
        attention_dim=config['model']['attention_dim'],
        task_embedding_dim=config['model']['task_embedding_dim'],
        dropout=config['model']['dropout'],
        num_classes=config['model']['num_classes'],
    ).to(device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"总参数量: {total_params:,}, 可训练: {trainable_params:,}")
    
    # 创建训练器
    trainer = TaskLevelTrainer(
        model=model,
        device=device,
        num_classes=config['model']['num_classes'],
        class_weights=class_weights if config['training']['use_class_weights'] else None,
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
        grad_clip=config['training']['grad_clip'],
    )
    
    # 训练循环
    best_val_acc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    logger.info("开始训练...")
    for epoch in range(config['training']['epochs']):
        logger.info(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        
        # 训练
        train_metrics = trainer.train_epoch(train_loader)
        logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        
        # 验证
        val_metrics = trainer.evaluate(val_loader)
        logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        
        # 记录历史
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        
        # 学习率调整
        trainer.scheduler.step(val_metrics['loss'])
        
        # 早停检查
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            patience_counter = 0
            # 保存最佳模型
            best_model_path = output_dir / 'best_model.pt'
            trainer.save_checkpoint(best_model_path, epoch, best_val_acc)
            logger.info(f"保存最佳模型 (val_acc={best_val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config['training']['patience']:
                logger.info(f"早停! {config['training']['patience']}个epoch没有改善")
                break
    
    # 加载最佳模型进行测试
    logger.info("\n加载最佳模型进行测试...")
    trainer.load_checkpoint(output_dir / 'best_model.pt')
    test_metrics = trainer.evaluate(test_loader)
    logger.info(f"Test - Loss: {test_metrics['loss']:.4f}, Acc: {test_metrics['accuracy']:.4f}")
    
    # 详细评估报告
    logger.info("\n分类报告:")
    print(classification_report(
        test_metrics['labels'], 
        test_metrics['predictions'],
        target_names=['Class 0', 'Class 1', 'Class 2']
    ))
    
    logger.info("混淆矩阵:")
    print(confusion_matrix(test_metrics['labels'], test_metrics['predictions']))
    
    # 保存配置和历史
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"\n训练完成! 结果保存在: {output_dir}")


if __name__ == '__main__':
    main()
