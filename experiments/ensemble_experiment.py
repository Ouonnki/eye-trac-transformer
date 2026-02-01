# -*- coding: utf-8 -*-
"""
集成模型实验

结合层级模型和片段模型进行2×2划分实验评估。

主要功能：
1. 在验证集(test2)上搜索最优权重
2. 评估集成模型在所有划分上的表现
3. 比较不同集成策略的效果
"""

import os
import sys
import logging
import json
import pickle
import argparse
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import UnifiedConfig
from src.models.dl_dataset import SequenceConfig, SegmentGazeDataset, collate_fn
from src.models.ensemble import EnsemblePredictor, EnsembleConfig, search_optimal_weights
from src.data.split_strategy import TwoByTwoSplitter

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LightweightGazeDataset(Dataset):
    """
    轻量级眼动数据集（复用层级模型训练时的数据集类）

    直接使用预处理好的numpy数据，不需要SubjectData对象。
    """

    def __init__(
        self,
        data: List[Dict],
        config: SequenceConfig,
        fit_normalizer: bool = False,
        normalizer_stats: Optional[Dict] = None,
        task_type: str = 'classification',
        use_task_embedding: bool = False,  # 默认关闭
    ):
        """
        初始化

        Args:
            data: 预处理后的数据列表
            config: 序列配置
            fit_normalizer: 是否拟合归一化器
            normalizer_stats: 已有的归一化统计量
            task_type: 任务类型 ('classification' 或 'regression')
            use_task_embedding: 是否使用任务嵌入
        """
        self.data = data
        self.config = config
        self.task_type = task_type
        self.use_task_embedding = use_task_embedding

        # 归一化统计量
        if normalizer_stats is not None:
            self.stats = normalizer_stats
        else:
            self.stats = {
                'dt_mean': 0.0, 'dt_std': 1.0,
                'velocity_mean': 0.0, 'velocity_std': 1.0,
                'acceleration_mean': 0.0, 'acceleration_std': 1.0,
            }

        if fit_normalizer:
            self._fit_normalizer()

        # 预归一化所有数据
        self._prenormalize_all()

    def _fit_normalizer(self):
        """计算归一化统计量"""
        all_dt = []
        all_velocity = []
        all_acceleration = []

        for subject_data in self.data:
            for task in subject_data['tasks']:
                for features in task['segments']:
                    if len(features) > 1:
                        all_dt.extend(features[1:, 2].tolist())
                        all_velocity.extend(features[1:, 3].tolist())
                        all_acceleration.extend(features[1:, 4].tolist())

        if all_dt:
            self.stats['dt_mean'] = float(np.mean(all_dt))
            self.stats['dt_std'] = float(np.std(all_dt)) + 1e-8
        if all_velocity:
            self.stats['velocity_mean'] = float(np.mean(all_velocity))
            self.stats['velocity_std'] = float(np.std(all_velocity)) + 1e-8
        if all_acceleration:
            self.stats['acceleration_mean'] = float(np.mean(all_acceleration))
            self.stats['acceleration_std'] = float(np.std(all_acceleration)) + 1e-8

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """应用归一化"""
        normalized = features.copy()
        if len(features) > 1:
            normalized[1:, 2] = (features[1:, 2] - self.stats['dt_mean']) / self.stats['dt_std']
            normalized[1:, 3] = (features[1:, 3] - self.stats['velocity_mean']) / self.stats['velocity_std']
            normalized[1:, 4] = (features[1:, 4] - self.stats['acceleration_mean']) / self.stats['acceleration_std']
        return normalized

    def _prenormalize_all(self):
        """预归一化所有数据"""
        for subject_data in self.data:
            for task in subject_data['tasks']:
                for i, features in enumerate(task['segments']):
                    if len(features) > 1:
                        task['segments'][i] = self._normalize(features.copy())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        subject_data = self.data[idx]

        # 初始化张量
        segments = np.zeros(
            (self.config.max_tasks, self.config.max_segments,
             self.config.max_seq_len, self.config.input_dim),
            dtype=np.float32
        )
        segment_lengths = np.zeros(
            (self.config.max_tasks, self.config.max_segments),
            dtype=np.int64
        )
        segment_mask = np.zeros(
            (self.config.max_tasks, self.config.max_segments),
            dtype=np.bool_
        )
        task_lengths = np.zeros(self.config.max_tasks, dtype=np.int64)
        task_mask = np.zeros(self.config.max_tasks, dtype=np.bool_)

        # 任务条件张量
        task_conditions = np.zeros((self.config.max_tasks, 5), dtype=np.int64)

        # 填充数据
        for t_idx, task in enumerate(subject_data['tasks'][:self.config.max_tasks]):
            num_segments = min(len(task['segments']), self.config.max_segments)
            task_lengths[t_idx] = num_segments
            task_mask[t_idx] = True

            for s_idx, features in enumerate(task['segments'][:self.config.max_segments]):
                seq_len = min(len(features), self.config.max_seq_len)
                if seq_len > 0:
                    segments[t_idx, s_idx, :seq_len, :] = features[:seq_len]
                    segment_lengths[t_idx, s_idx] = seq_len
                    segment_mask[t_idx, s_idx] = True

            # 处理任务条件
            if self.use_task_embedding and 'task_conditions' in task:
                tc = task['task_conditions']
                # grid_scale 映射: 9→1, 16→2, 25→3, 36→4
                grid_to_scale = {9: 1, 16: 2, 25: 3, 36: 4}
                grid_scale = grid_to_scale.get(tc.get('grid_size', 25), 3)
                # continuous_thinking: 0 if number_range[1] < 99 else 1
                number_range = tc.get('number_range', [1, 25])
                continuous_thinking = 0 if number_range[1] < 99 else 1
                task_conditions[t_idx] = [
                    grid_scale,
                    continuous_thinking,
                    int(tc.get('click_disappear', False)),
                    int(tc.get('has_distractor', False)),
                    int(tc.get('distractor_count', 0) > 0),
                ]

        # 根据任务类型返回不同的标签
        if self.task_type == 'classification':
            label = torch.tensor(subject_data['category'] - 1, dtype=torch.long)
        else:
            label = torch.tensor(subject_data['label'], dtype=torch.float32)

        result = {
            'segments': torch.from_numpy(segments),
            'segment_lengths': torch.from_numpy(segment_lengths),
            'segment_mask': torch.from_numpy(segment_mask),
            'task_lengths': torch.from_numpy(task_lengths),
            'task_mask': torch.from_numpy(task_mask),
            'label': label,
            'subject_id': subject_data['subject_id'],
        }

        if self.use_task_embedding:
            result['task_conditions'] = torch.from_numpy(task_conditions)

        return result


def load_processed_data(data_path: str) -> List[Dict]:
    """加载预处理好的数据"""
    logger.info(f'加载预处理数据: {data_path}')
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    logger.info(f'已加载 {len(data)} 个被试')
    return data


def create_datasets(
    data: List[Dict],
    seq_config: SequenceConfig,
    config: UnifiedConfig,
    normalizer_stats: Optional[Dict] = None,
) -> tuple:
    """
    创建层级和片段数据集

    Args:
        data: 预处理后的数据
        seq_config: 序列配置
        config: 统一配置
        normalizer_stats: 归一化统计量

    Returns:
        (hierarchical_dataset, segment_dataset)
    """
    # 使用 LightweightGazeDataset 作为层级数据集
    hierarchical_dataset = LightweightGazeDataset(
        data, seq_config,
        normalizer_stats=normalizer_stats,
        task_type=config.task.type,
        use_task_embedding=config.model.use_task_embedding,
    )
    # 使用 SegmentGazeDataset.from_processed_data 创建片段数据集
    segment_dataset = SegmentGazeDataset.from_processed_data(
        data, seq_config, normalizer_stats
    )
    return hierarchical_dataset, segment_dataset


def evaluate_on_split(
    predictor: EnsemblePredictor,
    split_data: List[Dict],
    seq_config: SequenceConfig,
    config: UnifiedConfig,
    split_name: str,
    normalizer_stats: Optional[Dict] = None,
) -> Dict:
    """
    在特定划分上评估集成模型

    Args:
        predictor: 集成预测器
        split_data: 划分数据
        seq_config: 序列配置
        config: 统一配置
        split_name: 划分名称
        normalizer_stats: 归一化统计量

    Returns:
        评估结果字典
    """
    logger.info(f'评估 {split_name}...')

    # 创建数据集
    hier_dataset, seg_dataset = create_datasets(
        split_data, seq_config, config, normalizer_stats
    )

    logger.info(f'{split_name}: {len(hier_dataset)} 个被试, {len(seg_dataset)} 个片段')

    # 评估
    metrics = predictor.evaluate(hier_dataset, seg_dataset)

    # 获取详细预测结果
    results = predictor.predict(hier_dataset, seg_dataset, return_individual=True)

    # 打印结果
    logger.info(f'{split_name} 结果:')
    logger.info(f"  [集成] Accuracy: {metrics['ensemble_accuracy']:.4f}, F1: {metrics['ensemble_f1']:.4f}")
    logger.info(f"  [层级] Accuracy: {metrics['hier_accuracy']:.4f}, F1: {metrics['hier_f1']:.4f}")
    logger.info(f"  [片段] Accuracy: {metrics['seg_accuracy']:.4f}, F1: {metrics['seg_f1']:.4f}")
    logger.info(f"  平均权重: hier={metrics['avg_hier_weight']:.2f}, seg={metrics['avg_seg_weight']:.2f}")

    return {
        'metrics': metrics,
        'predictions': results,
        'confusion_matrix': confusion_matrix(results['labels'], results['ensemble_preds']).tolist(),
    }


def run_ensemble_experiment(
    data: List[Dict],
    config: UnifiedConfig,
    seq_config: SequenceConfig,
    hierarchical_weight_path: str,
    segment_weight_path: str,
    output_dir: str,
    strategy: str = "condition_aware",
) -> Dict:
    """
    运行集成实验

    Args:
        data: 预处理后的数据
        config: 统一配置
        seq_config: 序列配置
        hierarchical_weight_path: 层级模型权重路径
        segment_weight_path: 片段模型权重路径
        output_dir: 输出目录
        strategy: 集成策略

    Returns:
        实验结果
    """
    logger.info('=' * 60)
    logger.info('集成模型实验')
    logger.info(f'策略: {strategy}')
    logger.info('=' * 60)

    os.makedirs(output_dir, exist_ok=True)

    # 创建划分器
    splitter = TwoByTwoSplitter(
        train_subjects=config.experiment.train_subjects,
        train_tasks=config.experiment.train_tasks,
        random_state=config.experiment.random_seed,
    )

    # 执行划分
    splits = splitter.split(data)

    # 获取训练集归一化统计量
    train_hier_dataset = LightweightGazeDataset(
        splits['train'], seq_config, fit_normalizer=True,
        task_type=config.task.type,
        use_task_embedding=config.model.use_task_embedding,
    )
    normalizer_stats = train_hier_dataset.stats

    # 创建集成配置
    ensemble_config = EnsembleConfig(
        hierarchical_weight_path=hierarchical_weight_path,
        segment_weight_path=segment_weight_path,
        strategy=strategy,
        device=config.device.device,
        batch_size=config.training.batch_size,
    )

    # 创建集成预测器
    predictor = EnsemblePredictor(config, seq_config, ensemble_config)
    predictor.load_models()

    results = {
        'config': {
            'strategy': strategy,
            'hierarchical_weight_path': hierarchical_weight_path,
            'segment_weight_path': segment_weight_path,
        },
        'splits': {},
    }

    # 如果是固定权重策略，先在test2上搜索最优权重
    if strategy == "fixed":
        logger.info('在 test2 上搜索最优权重...')
        val_hier, val_seg = create_datasets(
            splits['test2'], seq_config, config, normalizer_stats
        )
        best_weight, best_f1, search_results = search_optimal_weights(
            predictor, val_hier, val_seg
        )
        results['weight_search'] = {
            'best_hier_weight': float(best_weight),
            'best_f1': float(best_f1),
            'search_results': search_results,
        }
        # 更新权重
        predictor.ensemble_config.hierarchical_weight = best_weight
        predictor.ensemble_config.segment_weight = 1.0 - best_weight
        logger.info(f'使用最优权重: hier={best_weight:.2f}, seg={1-best_weight:.2f}')

    # 在所有划分上评估
    for split_name in ['train', 'test1', 'test2', 'test3']:
        split_results = evaluate_on_split(
            predictor, splits[split_name], seq_config, config,
            split_name, normalizer_stats
        )
        results['splits'][split_name] = split_results

    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(output_dir, f'ensemble_results_{timestamp}.json')

    # 移除不可序列化的数据
    results_to_save = {
        'config': results['config'],
        'weight_search': results.get('weight_search'),
        'splits': {
            name: {
                'metrics': split_results['metrics'],
                'confusion_matrix': split_results['confusion_matrix'],
            }
            for name, split_results in results['splits'].items()
        }
    }

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, indent=2, ensure_ascii=False)
    logger.info(f'结果已保存到: {results_file}')

    # 打印汇总
    logger.info('=' * 60)
    logger.info('实验结果汇总')
    logger.info('=' * 60)
    summary_data = []
    for split_name in ['train', 'test1', 'test2', 'test3']:
        m = results['splits'][split_name]['metrics']
        summary_data.append({
            'Split': split_name,
            'Ensemble Acc': f"{m['ensemble_accuracy']:.3f}",
            'Ensemble F1': f"{m['ensemble_f1']:.3f}",
            'Hier Acc': f"{m['hier_accuracy']:.3f}",
            'Seg Acc': f"{m['seg_accuracy']:.3f}",
        })
    df = pd.DataFrame(summary_data)
    logger.info('\n' + df.to_string(index=False))

    return results


def compare_strategies(
    data: List[Dict],
    config: UnifiedConfig,
    seq_config: SequenceConfig,
    hierarchical_weight_path: str,
    segment_weight_path: str,
    output_dir: str,
) -> Dict:
    """
    比较不同集成策略

    Args:
        data: 预处理后的数据
        config: 统一配置
        seq_config: 序列配置
        hierarchical_weight_path: 层级模型权重路径
        segment_weight_path: 片段模型权重路径
        output_dir: 输出目录

    Returns:
        比较结果
    """
    strategies = ["fixed", "condition_aware"]
    all_results = {}

    for strategy in strategies:
        logger.info(f'\n\n{"="*60}')
        logger.info(f'测试策略: {strategy}')
        logger.info('=' * 60)

        strategy_output = os.path.join(output_dir, strategy)
        results = run_ensemble_experiment(
            data, config, seq_config,
            hierarchical_weight_path, segment_weight_path,
            strategy_output, strategy
        )
        all_results[strategy] = results

    # 打印比较结果
    logger.info('\n\n' + '=' * 80)
    logger.info('策略比较汇总')
    logger.info('=' * 80)

    comparison_data = []
    for strategy in strategies:
        for split_name in ['train', 'test1', 'test2', 'test3']:
            m = all_results[strategy]['splits'][split_name]['metrics']
            comparison_data.append({
                'Strategy': strategy,
                'Split': split_name,
                'Ensemble Acc': m['ensemble_accuracy'],
                'Ensemble F1': m['ensemble_f1'],
                'Hier Acc': m['hier_accuracy'],
                'Seg Acc': m['seg_accuracy'],
            })

    df = pd.DataFrame(comparison_data)
    pivot = df.pivot(index='Split', columns='Strategy', values=['Ensemble Acc', 'Ensemble F1'])
    logger.info('\n' + pivot.to_string())

    return all_results


def main():
    parser = argparse.ArgumentParser(description='集成模型实验')
    parser.add_argument('--config', type=str, default='configs/default.json',
                       help='配置文件路径')
    parser.add_argument('--data', type=str, default='data/processed/processed_data.pkl',
                       help='预处理数据路径')
    parser.add_argument('--hier-weights', type=str, required=True,
                       help='层级模型权重路径')
    parser.add_argument('--seg-weights', type=str, required=True,
                       help='片段模型权重路径')
    parser.add_argument('--output', type=str, default='outputs/ensemble',
                       help='输出目录')
    parser.add_argument('--strategy', type=str, default='condition_aware',
                       choices=['fixed', 'condition_aware', 'compare'],
                       help='集成策略（compare 表示比较所有策略）')

    args = parser.parse_args()

    # 加载配置
    config = UnifiedConfig.from_json(args.config)

    # 创建序列配置
    seq_config = SequenceConfig(
        max_seq_len=100,
        max_tasks=30,
        max_segments=30,
        input_dim=config.model.input_dim,
    )

    # 加载数据
    data = load_processed_data(args.data)

    # 运行实验
    if args.strategy == 'compare':
        results = compare_strategies(
            data, config, seq_config,
            args.hier_weights, args.seg_weights,
            args.output
        )
    else:
        results = run_ensemble_experiment(
            data, config, seq_config,
            args.hier_weights, args.seg_weights,
            args.output, args.strategy
        )

    logger.info('实验完成！')


if __name__ == '__main__':
    main()
