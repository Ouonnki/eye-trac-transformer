# -*- coding: utf-8 -*-
"""
片段级Transformer实验

使用片段级编码器进行眼动注意力预测实验。
与被试级模型的区别：
- 输入：单个片段而非被试的所有数据
- 样本量：大幅增加（被试×任务×片段）
- 推理：聚合片段预测为被试级预测

使用 2×2 矩阵划分验证跨被试和跨任务泛化能力。

需要先运行 scripts/preprocess_data.py 预处理数据。
"""

import os
import sys
import logging
import json
import pickle
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.dl_dataset import SequenceConfig, SegmentGazeDataset, segment_collate_fn
from src.models.segment_trainer import SegmentTrainer
from src.data.split_strategy import TwoByTwoSplitter
from src.config import UnifiedConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_processed_data(data_path: str) -> List[Dict]:
    """加载预处理好的数据"""
    logger.info(f'加载预处理数据: {data_path}')
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    logger.info(f'已加载 {len(data)} 个被试')
    return data


def create_segment_dataset_from_split(
    split_data: List[Dict],
    seq_config: SequenceConfig,
    normalizer_stats: Optional[Dict] = None,
) -> SegmentGazeDataset:
    """
    从划分后的数据创建片段级数据集

    Args:
        split_data: 划分后的数据列表（预处理后的格式）
        seq_config: 序列配置
        normalizer_stats: 归一化统计量（用于保持一致性）

    Returns:
        SegmentGazeDataset 实例
    """
    # 需要将预处理数据转换为 SubjectData 格式
    # 这里我们创建一个轻量级的适配
    from src.data.schemas import SubjectData, TaskTrial, SearchSegment, GazePoint

    subjects = []
    for subject_dict in split_data:
        # 构造 trial 列表
        trials = []
        for task_dict in subject_dict['tasks']:
            segments = []
            for segment_features in task_dict['segments']:
                # 将特征转换回 GazePoint 列表（近似逆变换）
                gaze_points = []
                for i, feat in enumerate(segment_features):
                    # 反归一化（近似）
                    x = feat[0] * seq_config.screen_width
                    y = feat[1] * seq_config.screen_height
                    # 时间戳使用近似值
                    timestamp = pd.Timestamp('2024-01-01') + pd.Timedelta(milliseconds=i * 10)
                    gaze_points.append(GazePoint(
                        timestamp=timestamp,
                        x=float(x),
                        y=float(y),
                        pupil_size=0.0,
                    ))
                segments.append(SearchSegment(
                    gaze_points=gaze_points,
                    target_number=int(i % 10),
                    search_time=len(segment_features) * 10,
                ))

            trials.append(TaskTrial(
                task_id=task_dict['task_id'],
                segments=segments,
            ))

        subject = SubjectData(
            subject_id=subject_dict['subject_id'],
            trials=trials,
            total_score=subject_dict['label'],
        )
        subjects.append(subject)

    # 创建数据集
    dataset = SegmentGazeDataset(
        subjects=subjects,
        config=seq_config,
        fit_normalizer=False,  # 不重新拟合，使用提供的统计量
    )

    # 如果提供了归一化统计量，应用它们
    if normalizer_stats:
        dataset.feature_extractor.dt_mean = normalizer_stats['dt_mean']
        dataset.feature_extractor.dt_std = normalizer_stats['dt_std']
        dataset.feature_extractor.velocity_mean = normalizer_stats['velocity_mean']
        dataset.feature_extractor.velocity_std = normalizer_stats['velocity_std']
        dataset.feature_extractor.acceleration_mean = normalizer_stats['acceleration_mean']
        dataset.feature_extractor.acceleration_std = normalizer_stats['acceleration_std']

    return dataset


def evaluate_on_split(
    trainer: SegmentTrainer,
    test_data: List[Dict],
    seq_config: SequenceConfig,
    split_name: str,
    normalizer_stats: Optional[Dict] = None,
) -> Dict:
    """
    在特定划分上评估模型

    Args:
        trainer: 训练好的训练器
        test_data: 测试数据
        seq_config: 序列配置
        split_name: 划分名称
        normalizer_stats: 归一化统计量

    Returns:
        评估结果字典
    """
    logger.info(f'评估 {split_name}...')

    # 创建测试数据集
    test_dataset = create_segment_dataset_from_split(test_data, seq_config, normalizer_stats)
    logger.info(f'{split_name}: {len(test_dataset)} 个片段, {len(set(test_dataset.segment_subject_ids))} 个被试')

    # 评估
    results = trainer.evaluate(test_dataset, aggregate=True)

    logger.info(f'{split_name} 结果:')
    if trainer.config.task.type == 'classification':
        logger.info(f"  Accuracy: {results['accuracy']:.4f}")
        logger.info(f"  F1: {results['f1']:.4f}")
        logger.info(f"  Precision: {results['precision']:.4f}")
        logger.info(f"  Recall: {results['recall']:.4f}")
    else:
        logger.info(f"  R2: {results['r2']:.4f}")
        logger.info(f"  MAE: {results['mae']:.4f}")
        logger.info(f"  RMSE: {results['rmse']:.4f}")

    return results


def run_2x2_experiment(
    data: List[Dict],
    config: UnifiedConfig,
    seq_config: SequenceConfig,
) -> Dict:
    """
    运行 2×2 划分实验

    划分说明：
    - train: 前100人 × 前20题（训练集）
    - test1: 后49人 × 前20题（新人旧题，跨被试泛化）
    - test2: 前100人 × 后10题（旧人新题，跨任务泛化）
    - test3: 后49人 × 后10题（新人新题，双重泛化）

    Args:
        data: 预处理后的数据
        config: 统一配置
        seq_config: 序列配置

    Returns:
        实验结果字典
    """
    logger.info('='*60)
    logger.info('片段级模型 2×2 划分实验')
    logger.info('='*60)

    # 创建划分器
    splitter = TwoByTwoSplitter(
        train_subjects=config.experiment.train_subjects,
        train_tasks=config.experiment.train_tasks,
        random_state=config.experiment.random_seed,
    )

    # 执行划分
    splits = splitter.split(data)

    # 计算归一化统计量（只在训练集上）
    logger.info('计算归一化统计量...')
    train_dataset_full = create_segment_dataset_from_split(splits['train'], seq_config)
    train_dataset_full.feature_extractor.fit_normalization(
        [f for f in train_dataset_full.segments if len(f) > 0]
    )
    normalizer_stats = {
        'dt_mean': train_dataset_full.feature_extractor.dt_mean,
        'dt_std': train_dataset_full.feature_extractor.dt_std,
        'velocity_mean': train_dataset_full.feature_extractor.velocity_mean,
        'velocity_std': train_dataset_full.feature_extractor.velocity_std,
        'acceleration_mean': train_dataset_full.feature_extractor.acceleration_mean,
        'acceleration_std': train_dataset_full.feature_extractor.acceleration_std,
    }
    logger.info(f'归一化统计量: {normalizer_stats}')

    # 使用归一化后的训练集
    train_dataset = create_segment_dataset_from_split(splits['train'], seq_config, normalizer_stats)
    logger.info(f'训练集: {len(train_dataset)} 个片段')

    # 创建验证集（从训练集划分20%）
    val_size = len(train_dataset) // 5
    train_size = len(train_dataset) - val_size
    from torch.utils.data import Subset
    import random
    indices = list(range(len(train_dataset)))
    random.shuffle(indices)
    train_subset = Subset(train_dataset, indices[train_size:])
    val_subset = Subset(train_dataset, indices[:val_size])

    logger.info(f'训练子集: {len(train_subset)} 片段, 验证子集: {len(val_subset)} 片段')

    # 创建训练器
    trainer = SegmentTrainer(config, seq_config)

    # 训练
    train_result = trainer.train(train_subset, val_subset, fold=0)

    # 在四个划分上评估
    results = {
        'train': evaluate_on_split(trainer, splits['train'], seq_config, 'train', normalizer_stats),
        'test1': evaluate_on_split(trainer, splits['test1'], seq_config, 'test1 (新人旧题)', normalizer_stats),
        'test2': evaluate_on_split(trainer, splits['test2'], seq_config, 'test2 (旧人新题)', normalizer_stats),
        'test3': evaluate_on_split(trainer, splits['test3'], seq_config, 'test3 (新人新题)', normalizer_stats),
    }

    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = os.path.join(config.experiment.output_dir, f'segment_2x2_results_{timestamp}.json')
    with open(result_file, 'w') as f:
        # 转换numpy类型为Python类型
        serializable_results = {}
        for split_name, split_results in results.items():
            serializable_results[split_name] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in split_results.items()
                if k not in ['predictions', 'labels', 'subject_ids']
            }
        json.dump(serializable_results, f, indent=2)
    logger.info(f'结果已保存: {result_file}')

    return results


def main():
    """主函数"""
    # 加载配置
    config = UnifiedConfig.from_json('configs/default.json')

    # 修改输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config.experiment.output_dir = f'outputs/segment_models/{timestamp}'

    # 序列配置（数据属性，使用默认值）
    seq_config = SequenceConfig()

    logger.info(f'配置: {config.to_dict()}')

    # 加载数据
    data_path = 'data/processed/processed_data.pkl'
    if not os.path.exists(data_path):
        logger.error(f'预处理数据不存在: {data_path}')
        logger.info('请先运行 scripts/preprocess_data.py')
        return

    data = load_processed_data(data_path)

    # 运行实验
    results = run_2x2_experiment(data, config, seq_config)

    # 打印汇总
    logger.info('='*60)
    logger.info('实验汇总')
    logger.info('='*60)

    if config.task.type == 'classification':
        logger.info('划分       | Accuracy | F1       | Precision | Recall')
        logger.info('-' * 55)
        for split_name, split_results in results.items():
            logger.info(
                f'{split_name:10} | {split_results["accuracy"]:.4f}   | '
                f'{split_results["f1"]:.4f}   | {split_results["precision"]:.4f}    | '
                f'{split_results["recall"]:.4f}'
            )
    else:
        logger.info('划分       | R2       | MAE      | RMSE')
        logger.info('-' * 40)
        for split_name, split_results in results.items():
            logger.info(
                f'{split_name:10} | {split_results["r2"]:.4f}   | '
                f'{split_results["mae"]:.4f}   | {split_results["rmse"]:.4f}'
            )


if __name__ == '__main__':
    main()
