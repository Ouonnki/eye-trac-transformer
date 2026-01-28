# -*- coding: utf-8 -*-
"""
片段级 CADT 域适应实验

使用 CADT 域适应方法提升片段级模型的跨被试泛化能力。
"""

import os
import sys
import json
import pickle
import logging
import argparse
from datetime import datetime

import numpy as np

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import UnifiedConfig
from src.models.dl_dataset import SequenceConfig, SegmentGazeDataset
from src.models.segment_cadt_trainer import SegmentCADTTrainer
from src.data.split_strategy import TwoByTwoSplitter

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def save_config_snapshot(config: UnifiedConfig, output_dir: str) -> str:
    """
    保存配置文件快照

    Args:
        config: 配置对象
        output_dir: 输出目录

    Returns:
        快照文件路径
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    snapshot_path = os.path.join(output_dir, f'config_snapshot_{timestamp}.json')
    config.to_json(snapshot_path)
    logger.info(f'配置快照已保存到 {snapshot_path}')
    return snapshot_path


def load_data(data_path: str):
    """加载预处理数据"""
    logger.info(f'加载数据: {data_path}')
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    logger.info(f'加载完成: {len(data)} 个被试')
    return data


def run_segment_cadt_experiment(config: UnifiedConfig):
    """
    运行片段级 CADT 实验

    Args:
        config: 统一配置
    """
    logger.info('=' * 60)
    logger.info('片段级 CADT 域适应实验')
    logger.info('=' * 60)

    # 创建输出目录
    os.makedirs(config.experiment.output_dir, exist_ok=True)

    # 保存配置快照
    save_config_snapshot(config, config.experiment.output_dir)

    # 加载数据
    data_path = getattr(config.experiment, 'data_path', 'data/processed/processed_data.pkl')
    data = load_data(data_path)

    # 创建划分器
    splitter = TwoByTwoSplitter(
        train_subjects=config.experiment.train_subjects,
        train_tasks=config.experiment.train_tasks,
        random_state=config.experiment.random_seed,
    )

    # 执行划分
    splits = splitter.split(data)
    logger.info(f'划分完成:')
    logger.info(f'  train: {len(splits["train"])} 被试')
    logger.info(f'  test1: {len(splits["test1"])} 被试')
    logger.info(f'  test2: {len(splits["test2"])} 被试')
    logger.info(f'  test3: {len(splits["test3"])} 被试')

    # 创建序列配置
    seq_config = SequenceConfig()

    # 创建数据集
    logger.info('创建数据集...')
    source_dataset = SegmentGazeDataset.from_processed_data(splits['train'], seq_config)
    target_dataset = SegmentGazeDataset.from_processed_data(
        splits[config.cadt.target_domain], seq_config
    )

    logger.info(f'源域 (train): {len(source_dataset)} 片段')
    logger.info(f'目标域 ({config.cadt.target_domain}): {len(target_dataset)} 片段')

    # 创建训练器并训练
    trainer = SegmentCADTTrainer(config, seq_config)
    best_metrics = trainer.train(source_dataset, target_dataset)

    logger.info('=' * 60)
    logger.info('训练完成，开始评估所有划分')
    logger.info('=' * 60)

    # 评估所有划分
    results = {}
    for split_name in ['train', 'test1', 'test2', 'test3']:
        eval_dataset = SegmentGazeDataset.from_processed_data(splits[split_name], seq_config)
        metrics = trainer.evaluate(eval_dataset)
        results[split_name] = metrics
        logger.info(f'{split_name}: Acc={metrics["accuracy"]:.4f}, F1={metrics["f1"]:.4f}, '
                   f'Precision={metrics["precision"]:.4f}, Recall={metrics["recall"]:.4f}')

    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = os.path.join(config.experiment.output_dir, f'segment_cadt_results_{timestamp}.json')

    # 转换为可序列化格式
    serializable_results = {}
    for split_name, split_results in results.items():
        serializable_results[split_name] = {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in split_results.items()
        }

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            'target_domain': config.cadt.target_domain,
            'best_epoch': best_metrics.get('epoch', 0) if best_metrics else 0,
            'results': serializable_results,
        }, f, indent=2, ensure_ascii=False)

    logger.info(f'结果已保存到 {result_file}')

    # 打印汇总
    logger.info('=' * 60)
    logger.info('实验汇总')
    logger.info('=' * 60)
    logger.info(f'目标域: {config.cadt.target_domain}')
    logger.info(f'划分       | Accuracy | F1       | Precision | Recall')
    logger.info('-' * 55)
    for split_name, metrics in results.items():
        logger.info(f'{split_name:10} | {metrics["accuracy"]:.4f}   | {metrics["f1"]:.4f}   | '
                   f'{metrics["precision"]:.4f}    | {metrics["recall"]:.4f}')

    return results


def main():
    parser = argparse.ArgumentParser(description='片段级 CADT 域适应实验')
    parser.add_argument('--config', type=str, required=True,
                        help='配置文件路径')
    args = parser.parse_args()

    # 加载配置
    logger.info(f'加载配置: {args.config}')
    config = UnifiedConfig.from_json(args.config)

    # 运行实验
    run_segment_cadt_experiment(config)


if __name__ == '__main__':
    main()
