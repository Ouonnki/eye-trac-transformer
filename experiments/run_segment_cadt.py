#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
片段级CADT实验脚本

使用CADT域适应改善跨被试泛化性能。

使用方法:
    python experiments/run_segment_cadt.py --config configs/seg.json --target_domain test1
"""

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import UnifiedConfig
from src.models.dl_dataset import SequenceConfig, SegmentGazeDataset
from src.models.segment_cadt_trainer import SegmentCADTTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data_splits(data_path: str):
    """
    加载数据分割

    Args:
        data_path: 预处理数据路径

    Returns:
        数据分割字典
    """
    import pickle

    path = Path(data_path)
    if not path.exists():
        # 尝试默认路径
        default_path = Path('data/processed/segment_data.pkl')
        if default_path.exists():
            path = default_path
        else:
            raise FileNotFoundError(f"数据文件不存在: {data_path}")

    logger.info(f"加载数据: {path}")
    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data


def run_segment_cadt_experiment(
    config_path: str,
    data_path: str,
    target_domain: str = 'test1',
):
    """
    运行片段级CADT实验

    Args:
        config_path: 配置文件路径
        data_path: 预处理数据路径
        target_domain: 目标域 ('test1', 'test2', 'test3')
    """
    # 加载配置
    config = UnifiedConfig.from_json(config_path)
    seq_config = SequenceConfig.from_config(config)

    logger.info(f"配置加载完成: {config_path}")
    logger.info(f"目标域: {target_domain}")
    logger.info(f"预训练轮数: {config.cadt.pre_train_epochs}")
    logger.info(f"KL权重: {config.cadt.cadt_kl_weight}")
    logger.info(f"Reset模式: {config.cadt.reset_mode}")

    # 加载数据
    splits = load_data_splits(data_path)

    # 创建数据集
    logger.info("创建源域数据集 (train)...")
    train_dataset = SegmentGazeDataset.from_processed_data(
        splits['train'],
        seq_config,
        fit_normalizer=True,
        domain='train',
    )

    logger.info(f"创建目标域数据集 ({target_domain})...")
    test_dataset = SegmentGazeDataset.from_processed_data(
        splits[target_domain],
        seq_config,
        normalizer_stats={
            'dt_mean': train_dataset.feature_extractor.dt_mean,
            'dt_std': train_dataset.feature_extractor.dt_std,
            'velocity_mean': train_dataset.feature_extractor.velocity_mean,
            'velocity_std': train_dataset.feature_extractor.velocity_std,
            'acceleration_mean': train_dataset.feature_extractor.acceleration_mean,
            'acceleration_std': train_dataset.feature_extractor.acceleration_std,
        },
        domain=target_domain,
    )

    logger.info(f"源域样本数: {len(train_dataset)}")
    logger.info(f"目标域样本数: {len(test_dataset)}")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.device.num_workers,
        pin_memory=config.device.pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.device.num_workers,
        pin_memory=config.device.pin_memory,
    )

    # 创建训练器
    logger.info("初始化CADT训练器...")
    trainer = SegmentCADTTrainer(
        config=config,
        seq_config=seq_config,
        target_domain=target_domain,
    )

    # 训练
    output_dir = Path(config.experiment.output_dir) / f'cadt_{target_domain}'
    logger.info(f"开始训练，输出目录: {output_dir}")

    history = trainer.train(train_loader, test_loader, str(output_dir))

    # 最终评估
    final_metrics = trainer.evaluate(test_loader)
    logger.info(f"{target_domain} 最终准确率: {final_metrics['accuracy']:.4f}")

    # 保存配置快照
    config.save_snapshot(str(output_dir))

    return history, final_metrics


def main():
    parser = argparse.ArgumentParser(description='片段级CADT域适应实验')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/seg.json',
        help='配置文件路径'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/processed/segment_data.pkl',
        help='预处理数据路径'
    )
    parser.add_argument(
        '--target_domain',
        type=str,
        default='test1',
        choices=['test1', 'test2', 'test3'],
        help='目标域选择'
    )
    args = parser.parse_args()

    run_segment_cadt_experiment(args.config, args.data, args.target_domain)


if __name__ == '__main__':
    main()
