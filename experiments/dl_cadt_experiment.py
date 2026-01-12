# -*- coding: utf-8 -*-
"""
CADT 域适应 Transformer 分类实验

基于 CADT 思想实现的 Transformer 域适应实验。
支持将三个测试集（test1/test2/test3）分别设置为目标域。

用法：
    # 以 test1（跨被试泛化）作为目标域训练
    TARGET_DOMAIN=test1 python experiments/dl_cadt_experiment.py

    # 以 test2（跨任务泛化）作为目标域训练
    TARGET_DOMAIN=test2 python experiments/dl_cadt_experiment.py

    # 以 test3（双重泛化）作为目标域训练
    TARGET_DOMAIN=test3 python experiments/dl_cadt_experiment.py

    # 调整超参数
    TARGET_DOMAIN=test1 KL_WEIGHT=0.5 DIS_WEIGHT=2.0 python experiments/dl_cadt_experiment.py

环境变量：
    TARGET_DOMAIN: 目标域选择 (test1/test2/test3)
    KL_WEIGHT: 原型聚类损失权重
    DIS_WEIGHT: 域对抗损失权重
    PRE_TRAIN_EPOCHS: 预训练阶段 epoch 数
    MODEL_MODE: 模型配置模式 (full/light)
"""

import os
import sys
import logging
import json
import pickle
from datetime import datetime
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.dl_dataset import SequenceConfig, collate_fn
from src.models.dl_cadt_config import CADTConfig
from src.models.dl_cadt_trainer import CADTTrainer
from src.data.split_strategy import TwoByTwoSplitter
from experiments.experiment_config import ExperimentConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LightweightGazeDataset(Dataset):
    """
    轻量级眼动数据集

    直接使用预处理好的numpy数据。
    """

    def __init__(
        self,
        data: List[Dict],
        config: SequenceConfig,
        fit_normalizer: bool = False,
        normalizer_stats: Dict = None,
        task_type: str = 'classification',
    ):
        """
        初始化

        Args:
            data: 预处理后的数据列表
            config: 序列配置
            fit_normalizer: 是否拟合归一化器
            normalizer_stats: 已有的归一化统计量
            task_type: 任务类型
        """
        self.data = data
        self.config = config
        self.task_type = task_type

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
                        # dt, velocity, acceleration 分别在索引 2, 3, 4
                        all_dt.extend(features[:, 2].flatten())
                        all_velocity.extend(features[:, 3].flatten())
                        all_acceleration.extend(features[:, 4].flatten())

        self.stats = {
            'dt_mean': np.mean(all_dt) if all_dt else 0.0,
            'dt_std': np.std(all_dt) if all_dt else 1.0,
            'velocity_mean': np.mean(all_velocity) if all_velocity else 0.0,
            'velocity_std': np.std(all_velocity) if all_velocity else 1.0,
            'acceleration_mean': np.mean(all_acceleration) if all_acceleration else 0.0,
            'acceleration_std': np.std(all_acceleration) if all_acceleration else 1.0,
        }

        # 避免除零
        for key in ['dt_std', 'velocity_std', 'acceleration_std']:
            if self.stats[key] < 1e-8:
                self.stats[key] = 1.0

    def _prenormalize_all(self):
        """预归一化所有数据"""
        self.normalized_data = []

        for subject_data in self.data:
            normalized_subject = {
                'subject_id': subject_data['subject_id'],
                'label': subject_data['label'],
                'category': subject_data.get('category', 2),
                'tasks': [],
            }

            for task in subject_data['tasks']:
                normalized_task = {
                    'task_id': task['task_id'],
                    'segments': [],
                }

                for features in task['segments']:
                    if len(features) == 0:
                        normalized_task['segments'].append(np.zeros((1, 7), dtype=np.float32))
                        continue

                    norm_features = features.copy().astype(np.float32)
                    # 归一化 dt, velocity, acceleration
                    norm_features[:, 2] = (norm_features[:, 2] - self.stats['dt_mean']) / self.stats['dt_std']
                    norm_features[:, 3] = (norm_features[:, 3] - self.stats['velocity_mean']) / self.stats['velocity_std']
                    norm_features[:, 4] = (norm_features[:, 4] - self.stats['acceleration_mean']) / self.stats['acceleration_std']

                    normalized_task['segments'].append(norm_features)

                normalized_subject['tasks'].append(normalized_task)

            self.normalized_data.append(normalized_subject)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        subject_data = self.normalized_data[idx]

        # 构建输出张量
        max_tasks = self.config.max_tasks
        max_segments = self.config.max_segments
        max_seq_len = self.config.max_seq_len

        segments = np.zeros((max_tasks, max_segments, max_seq_len, 7), dtype=np.float32)
        segment_mask = np.zeros((max_tasks, max_segments), dtype=bool)
        task_mask = np.zeros(max_tasks, dtype=bool)
        segment_lengths = np.zeros((max_tasks, max_segments), dtype=np.int64)
        task_lengths = np.zeros(max_tasks, dtype=np.int64)  # 每个任务的有效片段数

        for t_idx, task in enumerate(subject_data['tasks'][:max_tasks]):
            task_mask[t_idx] = True
            num_segments = min(len(task['segments']), max_segments)
            task_lengths[t_idx] = num_segments

            for s_idx, seg_features in enumerate(task['segments'][:max_segments]):
                segment_mask[t_idx, s_idx] = True
                seq_len = min(len(seg_features), max_seq_len)
                segment_lengths[t_idx, s_idx] = seq_len

                if seq_len > 0:
                    segments[t_idx, s_idx, :seq_len, :] = seg_features[:seq_len]

        # 标签处理
        if self.task_type == 'classification':
            label = subject_data['category'] - 1  # 转为 0-indexed
        else:
            label = subject_data['label']

        return {
            'subject_id': subject_data['subject_id'],
            'segments': torch.from_numpy(segments),
            'segment_mask': torch.from_numpy(segment_mask),
            'task_mask': torch.from_numpy(task_mask),
            'segment_lengths': torch.from_numpy(segment_lengths),
            'task_lengths': torch.from_numpy(task_lengths),
            'label': torch.tensor(label, dtype=torch.long if self.task_type == 'classification' else torch.float32),
        }


def load_processed_data(path: str) -> List[Dict]:
    """加载预处理后的数据"""
    logger.info(f'加载预处理数据: {path}')

    with open(path, 'rb') as f:
        data = pickle.load(f)

    logger.info(f'加载了 {len(data)} 个被试的数据')
    return data


def run_cadt_experiment(
    data: List[Dict],
    config: CADTConfig,
    experiment_config: ExperimentConfig,
) -> Dict:
    """
    运行 CADT 域适应实验

    Args:
        data: 预处理后的数据
        config: CADT 配置
        experiment_config: 实验配置

    Returns:
        实验结果字典
    """
    logger.info('=' * 60)
    logger.info('开始 CADT 域适应实验')
    logger.info(f'目标域: {config.target_domain}')
    logger.info('=' * 60)

    # 2×2 数据划分
    splitter = TwoByTwoSplitter(
        train_subjects=experiment_config.train_subjects,
        train_tasks=experiment_config.train_tasks,
        random_state=experiment_config.random_seed,
    )
    splits = splitter.split(data)

    # 打印划分信息
    for split_name, split_data in splits.items():
        logger.info(f'{split_name}: {len(split_data)} 个被试')

    # 创建序列配置
    seq_config = SequenceConfig(
        max_seq_len=config.max_seq_len,
        max_tasks=config.max_tasks,
        max_segments=config.max_segments,
    )

    # 创建数据集
    source_dataset = LightweightGazeDataset(
        splits['train'], seq_config,
        fit_normalizer=True,
        task_type='classification',
    )
    target_dataset = LightweightGazeDataset(
        splits[config.target_domain], seq_config,
        normalizer_stats=source_dataset.stats,
        task_type='classification',
    )

    # 创建训练器并训练
    trainer = CADTTrainer(config)
    best_metrics = trainer.train(
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        val_dataset=target_dataset,
    )

    # 评估所有测试集
    results = {
        'target_domain': config.target_domain,
        'best_metrics': best_metrics,
        'test_results': {},
    }

    for test_name in ['test1', 'test2', 'test3']:
        test_dataset = LightweightGazeDataset(
            splits[test_name], seq_config,
            normalizer_stats=source_dataset.stats,
            task_type='classification',
        )
        test_metrics = trainer.evaluate(test_dataset)
        results['test_results'][test_name] = test_metrics
        logger.info(f'{test_name}: Accuracy={test_metrics["accuracy"]:.4f}, F1={test_metrics["f1"]:.4f}')

    return results


def print_results(results: Dict):
    """打印实验结果"""
    print('\n' + '=' * 60)
    print('CADT 域适应实验结果')
    print('=' * 60)
    print(f'目标域: {results["target_domain"]}')
    print(f'最佳验证准确率: {results["best_metrics"]["accuracy"]:.4f}')
    print(f'最佳验证 F1: {results["best_metrics"]["f1"]:.4f}')
    print(f'最佳 Epoch: {results["best_metrics"]["epoch"]}')
    print()
    print('各测试集结果:')
    for test_name, metrics in results['test_results'].items():
        marker = ' (目标域)' if test_name == results['target_domain'] else ''
        print(f'  {test_name}{marker}: Accuracy={metrics["accuracy"]:.4f}, F1={metrics["f1"]:.4f}')
    print('=' * 60)


def save_results(results: Dict, output_dir: str, config: CADTConfig):
    """保存实验结果"""
    os.makedirs(output_dir, exist_ok=True)

    # 保存 JSON 结果
    results_path = os.path.join(output_dir, f'cadt_results_{config.target_domain}.json')

    serializable_results = {
        'target_domain': results['target_domain'],
        'best_metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v
                         for k, v in results['best_metrics'].items()},
        'test_results': {
            name: {k: float(v) if isinstance(v, (np.floating, float)) else v
                   for k, v in metrics.items()}
            for name, metrics in results['test_results'].items()
        },
        'config': str(config),
    }

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    logger.info(f'结果已保存: {results_path}')


def main():
    """主函数"""
    print('=' * 70)
    print('CADT Domain Adaptation Transformer Experiment')
    print('=' * 70)

    # ============================================================
    # 路径配置
    # ============================================================
    processed_data_path = 'data/processed/processed_data.pkl'
    base_output_dir = 'outputs/cadt_res'

    # ============================================================
    # 配置
    # ============================================================
    config = CADTConfig.from_env()

    # 打印配置
    model_mode = os.environ.get('MODEL_MODE', 'light')
    print(f'\n{"="*50}')
    print(f'MODEL_MODE: {model_mode}')
    print(f'目标域: {config.target_domain}')
    print(f'模型维度: segment_d={config.segment_d_model}, task_d={config.task_d_model}')
    print(f'序列配置: max_seq={config.max_seq_len}, max_tasks={config.max_tasks}, max_segments={config.max_segments}')
    print(f'CADT 参数: kl_w={config.cadt_kl_weight}, dis_w={config.cadt_dis_weight}')
    print(f'预训练 Epochs: {config.pre_train_epochs}')
    print(f'总 Epochs: {config.epochs}')
    print(f'批次大小: {config.batch_size}')
    print(f'混合精度: {config.use_amp}')
    print(f'梯度检查点: {config.use_gradient_checkpointing}')
    print(f'{"="*50}')

    # GPU 信息
    if torch.cuda.is_available():
        print(f'\nGPU: {torch.cuda.get_device_name(0)}')
        print(f'显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    else:
        print('\n警告: 没有可用的 GPU!')

    # 检查数据文件
    if not os.path.exists(processed_data_path):
        print(f'\n错误: 预处理数据不存在: {processed_data_path}')
        print('请先运行预处理脚本:')
        print('  python scripts/preprocess_data.py --data_dir <数据目录>')
        return

    # 加载数据
    data = load_processed_data(processed_data_path)

    if len(data) == 0:
        print('错误: 没有加载到数据!')
        return

    # 统计
    total_tasks = sum(len(d['tasks']) for d in data)
    total_segments = sum(
        sum(len(t['segments']) for t in d['tasks'])
        for d in data
    )
    print(f'\n数据统计:')
    print(f'  被试数: {len(data)}')
    print(f'  任务数: {total_tasks}')
    print(f'  片段数: {total_segments}')

    # 实验配置
    experiment_config = ExperimentConfig.from_env()

    # 输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(base_output_dir, f'cadt_{config.target_domain}_{timestamp}')
    config.output_dir = output_dir

    # ============================================================
    # 运行实验
    # ============================================================
    results = run_cadt_experiment(data, config, experiment_config)

    # 打印结果
    print_results(results)

    # 保存结果
    save_results(results, output_dir, config)

    print('\n实验完成!')


if __name__ == '__main__':
    main()
