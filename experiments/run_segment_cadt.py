#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
片段级CADT实验脚本

使用CADT域适应改善跨被试泛化性能。

使用方法:
    python experiments/run_segment_cadt.py --config configs/seg.json --target_domain test1
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging

import torch
from torch.utils.data import DataLoader

from src.config import UnifiedConfig
from src.models.dl_dataset import SequenceConfig, SegmentGazeDataset
from src.models.segment_cadt_trainer import SegmentCADTTrainer
from src.data.split_strategy import TwoByTwoSplitter

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
        default_path = Path('data/processed/processed_data.pkl')
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
        target_domain: 主要目标域 ('test1', 'test2', 'test3')
    """
    # 加载配置
    config = UnifiedConfig.from_json(config_path)

    # 创建序列配置
    seq_config = SequenceConfig(
        max_seq_len=100,
        max_tasks=30,
        max_segments=30,
        screen_width=1920,
        screen_height=1080,
        input_dim=config.model.input_dim,
    )

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"CADT 域适应实验 | 配置: {config_path}")
    logger.info("=" * 70)
    logger.info(f"主要目标域: {target_domain}")
    logger.info(f"预训练轮数: {config.cadt.pre_train_epochs}")
    logger.info(f"KL权重: {config.cadt.cadt_kl_weight}")
    logger.info(f"Reset模式: {config.cadt.reset_mode}")
    logger.info("")

    # 加载原始数据
    all_data = load_data_splits(data_path)
    logger.info(f"加载了 {len(all_data)} 个被试")

    # 使用 TwoByTwoSplitter 进行划分
    splitter = TwoByTwoSplitter(
        train_subjects=config.experiment.train_subjects,
        train_tasks=config.experiment.train_tasks,
        random_state=config.experiment.random_seed,
    )
    splits = splitter.split(all_data)

    # 创建源域数据集
    logger.info("创建源域数据集 (train)...")
    train_dataset = SegmentGazeDataset.from_processed_data(
        splits['train'],
        seq_config,
        domain='train',
    )

    # 拟合归一化器并应用到训练集
    all_features = [f for f in train_dataset.segments if len(f) > 0]
    if all_features:
        train_dataset.feature_extractor.fit_normalization(all_features)
        for i, features in enumerate(train_dataset.segments):
            if len(features) > 0:
                train_dataset.segments[i] = train_dataset.feature_extractor.normalize(features)

    # 收集归一化统计量
    normalizer_stats = {
        'dt_mean': train_dataset.feature_extractor.dt_mean,
        'dt_std': train_dataset.feature_extractor.dt_std,
        'velocity_mean': train_dataset.feature_extractor.velocity_mean,
        'velocity_std': train_dataset.feature_extractor.velocity_std,
        'acceleration_mean': train_dataset.feature_extractor.acceleration_mean,
        'acceleration_std': train_dataset.feature_extractor.acceleration_std,
    }

    # 创建所有测试集的数据集
    test_datasets = {}
    test_loaders = {}
    for test_domain in ['test1', 'test2', 'test3']:
        logger.info(f"创建目标域数据集 ({test_domain})...")
        test_datasets[test_domain] = SegmentGazeDataset.from_processed_data(
            splits[test_domain],
            seq_config,
            normalizer_stats=normalizer_stats,
            domain=test_domain,
        )
        test_loaders[test_domain] = DataLoader(
            test_datasets[test_domain],
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.device.num_workers,
            pin_memory=config.device.pin_memory,
        )

    logger.info(f"源域样本数: {len(train_dataset)}")
    for test_domain in ['test1', 'test2', 'test3']:
        logger.info(f"  {test_domain} 样本数: {len(test_datasets[test_domain])}")

    # 创建训练数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.device.num_workers,
        pin_memory=config.device.pin_memory,
    )

    # 创建训练器
    trainer = SegmentCADTTrainer(
        config=config,
        seq_config=seq_config,
        target_domain=target_domain,
    )

    # 训练
    output_dir = Path(config.experiment.output_dir) / f'cadt_{target_domain}'
    output_dir.mkdir(parents=True, exist_ok=True)

    history = trainer.train(train_loader, test_loaders[target_domain], str(output_dir))

    # 绘制训练曲线
    curve_path = output_dir / 'training_curves.png'
    trainer.plot_training_curves(history, str(curve_path))

    # 在所有测试集上进行最终评估（对比最终模型 vs 最佳模型）
    logger.info("")
    logger.info("=" * 70)
    logger.info("在所有测试集上进行最终评估")
    logger.info("=" * 70)

    best_model_path = output_dir / 'best_model.pth'
    has_best_model = best_model_path.exists()

    all_results = {}

    # 1. 使用训练结束时的最终模型评估
    logger.info("\n[最终模型]")
    for test_domain in ['test1', 'test2', 'test3']:
        metrics = trainer.evaluate(test_loaders[test_domain])
        all_results[f"{test_domain}_final"] = metrics
        logger.info(
            f"  {test_domain}: | "
            f"Loss={metrics['loss']:.4f} | "
            f"Acc={metrics['accuracy']:.4f} | "
            f"F1={metrics['f1']:.4f} | "
            f"Prec={metrics['precision']:.4f} | "
            f"Rec={metrics['recall']:.4f}"
        )

    # 2. 加载并使用最佳模型评估
    if has_best_model:
        logger.info(f"\n[最佳模型] 加载: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=trainer.device)
        best_epoch = checkpoint['epoch']
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"最佳模型来自 Epoch {best_epoch}")

        for test_domain in ['test1', 'test2', 'test3']:
            metrics = trainer.evaluate(test_loaders[test_domain])
            all_results[f"{test_domain}_best"] = metrics
            logger.info(
                f"  {test_domain}: | "
                f"Loss={metrics['loss']:.4f} | "
                f"Acc={metrics['accuracy']:.4f} | "
                f"F1={metrics['f1']:.4f} | "
                f"Prec={metrics['precision']:.4f} | "
                f"Rec={metrics['recall']:.4f}"
            )

    # 保存结果
    import json
    results_path = output_dir / 'final_results.json'
    with open(results_path, 'w') as f:
        serializable_results = {}
        for domain, metrics in all_results.items():
            serializable_results[domain] = {k: float(v) for k, v in metrics.items()}
        json.dump(serializable_results, f, indent=2)
    logger.info(f"\n结果已保存: {results_path}")

    # 保存配置快照
    config.save_snapshot(str(output_dir))

    return history, all_results


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
