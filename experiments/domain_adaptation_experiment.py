# -*- coding: utf-8 -*-
"""
域迁移实验脚本

使用 CADT 方法进行跨任务泛化实验。
支持分类/回归双模式，分类模式下启用域迁移。

使用示例：
    # 分类模式 + 域迁移（推荐）
    TASK_TYPE=classification MODEL_MODE=full python experiments/domain_adaptation_experiment.py

    # 回归模式（无域迁移）
    TASK_TYPE=regression MODEL_MODE=full python experiments/domain_adaptation_experiment.py

    # 正式实验（多次重复）
    EXPERIMENT_MODE=formal TASK_TYPE=classification python experiments/domain_adaptation_experiment.py

环境变量：
    - MODEL_MODE: 'full', 'medium', 'light'
    - TASK_TYPE: 'classification', 'regression'
    - EXPERIMENT_MODE: 'quick', 'formal'
    - TARGET_DOMAIN: 'test1', 'test2', 'test3', 'all'
    - ENABLE_DOMAIN_ADAPTATION: 'true', 'false'
"""

import sys
import os
import logging
import json
import pickle
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, f1_score, confusion_matrix,
)
from tqdm import tqdm

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.models.dl_dataset import SequenceConfig, collate_fn
from src.models.domain_trainer import (
    DomainAdaptationTrainer,
    DomainAdaptationConfig,
)
from experiments.experiment_config import ExperimentConfig
from src.data.split_strategy import TwoByTwoSplitter

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LightweightGazeDataset(Dataset):
    """
    轻量级眼动数据集

    直接使用预处理好的numpy数据，支持分类和回归双模式。
    """

    def __init__(
        self,
        data: List[Dict],
        config: SequenceConfig,
        fit_normalizer: bool = False,
        normalizer_stats: Optional[Dict] = None,
    ):
        """
        初始化

        Args:
            data: 预处理后的数据列表
            config: 序列配置
            fit_normalizer: 是否拟合归一化器
            normalizer_stats: 已有的归一化统计量
        """
        self.data = data
        self.config = config

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
        """归一化特征"""
        if len(features) > 1:
            features[1:, 2] = (features[1:, 2] - self.stats['dt_mean']) / self.stats['dt_std']
            features[1:, 3] = (features[1:, 3] - self.stats['velocity_mean']) / self.stats['velocity_std']
            features[1:, 4] = (features[1:, 4] - self.stats['acceleration_mean']) / self.stats['acceleration_std']
        return features

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
        task_ids = np.zeros(self.config.max_tasks, dtype=np.int64)

        # 填充数据
        for t_idx, task in enumerate(subject_data['tasks'][:self.config.max_tasks]):
            num_segments = min(len(task['segments']), self.config.max_segments)
            task_lengths[t_idx] = num_segments
            task_mask[t_idx] = True
            task_ids[t_idx] = task.get('task_id', t_idx)

            for s_idx, features in enumerate(task['segments'][:self.config.max_segments]):
                seq_len = min(len(features), self.config.max_seq_len)
                if seq_len > 0:
                    segments[t_idx, s_idx, :seq_len, :] = features[:seq_len]
                    segment_lengths[t_idx, s_idx] = seq_len
                    segment_mask[t_idx, s_idx] = True

        # 根据任务类型选择标签
        if self.config.task_type == 'classification':
            # 分类模式：使用 category 标签
            label = torch.tensor(
                subject_data.get('category', 0),
                dtype=torch.long
            )
        else:
            # 回归模式：使用 label（总分）
            label = torch.tensor(
                subject_data['label'],
                dtype=torch.float32
            )

        return {
            'segments': torch.from_numpy(segments),
            'segment_lengths': torch.from_numpy(segment_lengths),
            'segment_mask': torch.from_numpy(segment_mask),
            'task_lengths': torch.from_numpy(task_lengths),
            'task_mask': torch.from_numpy(task_mask),
            'task_ids': torch.from_numpy(task_ids),
            'label': label,
            'category': torch.tensor(subject_data.get('category', 0), dtype=torch.long),
            'subject_id': subject_data['subject_id'],
        }


def load_processed_data(data_path: str) -> List[Dict]:
    """加载预处理好的数据"""
    logger.info(f'加载预处理数据: {data_path}')
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    logger.info(f'加载了 {len(data)} 个被试')
    return data


def get_model_config(
    model_mode: str,
    experiment_config: ExperimentConfig,
    output_dir: str,
) -> DomainAdaptationConfig:
    """
    获取模型配置

    Args:
        model_mode: 'full', 'medium', 'light'
        experiment_config: 实验配置
        output_dir: 输出目录

    Returns:
        训练配置
    """
    base_params = {
        'task_type': experiment_config.task_type,
        'num_classes': experiment_config.num_classes,
        'enable_domain_adaptation': experiment_config.enable_domain_adaptation,
        'pretrain_epochs': experiment_config.pretrain_epochs,
        'center_alignment_weight': experiment_config.center_alignment_weight,
        'domain_adversarial_weight': experiment_config.domain_adversarial_weight,
        'center_diversity_weight': experiment_config.center_diversity_weight,
        'output_dir': output_dir,
        'save_figures': experiment_config.save_figures,
        'figure_dpi': experiment_config.figure_dpi,
    }

    if model_mode == 'full':
        # 完整模型配置（需要较大显存，约24GB+）
        return DomainAdaptationConfig(
            input_dim=7,
            segment_d_model=128,
            segment_nhead=8,
            segment_num_layers=6,
            task_d_model=256,
            task_nhead=8,
            task_num_layers=4,
            attention_dim=64,
            dropout=0.1,
            max_seq_len=100,
            max_tasks=30,
            max_segments=30,
            batch_size=16,
            learning_rate=1e-4,
            weight_decay=1e-4,
            warmup_epochs=5,
            epochs=100,
            patience=15,
            grad_clip=1.0,
            use_multi_gpu=False,
            use_amp=True,
            use_gradient_checkpointing=True,
            num_workers=4,
            pin_memory=True,
            save_best=True,
            **base_params,
        )
    elif model_mode == 'medium':
        # 中等配置（适合RTX 3090 24GB）
        return DomainAdaptationConfig(
            input_dim=7,
            segment_d_model=96,
            segment_nhead=6,
            segment_num_layers=4,
            task_d_model=192,
            task_nhead=6,
            task_num_layers=3,
            attention_dim=48,
            dropout=0.1,
            max_seq_len=80,
            max_tasks=25,
            max_segments=25,
            batch_size=8,
            learning_rate=8e-5,
            weight_decay=1e-4,
            warmup_epochs=4,
            epochs=100,
            patience=15,
            grad_clip=1.0,
            use_multi_gpu=False,
            use_amp=True,
            use_gradient_checkpointing=True,
            num_workers=4,
            pin_memory=True,
            save_best=True,
            **base_params,
        )
    else:  # 'light'
        # 轻量配置
        return DomainAdaptationConfig(
            input_dim=7,
            segment_d_model=64,
            segment_nhead=4,
            segment_num_layers=2,
            task_d_model=128,
            task_nhead=4,
            task_num_layers=2,
            attention_dim=32,
            dropout=0.1,
            max_seq_len=50,
            max_tasks=20,
            max_segments=20,
            batch_size=4,
            learning_rate=5e-5,
            weight_decay=1e-4,
            warmup_epochs=3,
            epochs=100,
            patience=15,
            grad_clip=1.0,
            use_multi_gpu=False,
            use_amp=True,
            use_gradient_checkpointing=False,
            num_workers=2,
            pin_memory=True,
            save_best=True,
            **base_params,
        )


def run_2x2_experiment(
    data: List[Dict],
    training_config: DomainAdaptationConfig,
    experiment_config: ExperimentConfig,
) -> Dict:
    """
    运行 2×2 实验设计

    Args:
        data: 预处理后的数据
        training_config: 训练配置
        experiment_config: 实验配置

    Returns:
        实验结果字典
    """
    start_time = time.time()

    logger.info('运行 2×2 实验设计...')
    logger.info(f'任务类型: {experiment_config.task_type}')
    logger.info(f'训练被试数: {experiment_config.train_subjects}')
    logger.info(f'训练任务数: {experiment_config.train_tasks}')
    logger.info(f'重复次数: {experiment_config.n_repeats}')
    if experiment_config.task_type == 'classification':
        logger.info(f'域迁移: {experiment_config.enable_domain_adaptation}')
        logger.info(f'目标域: {experiment_config.target_domain}')

    # 序列配置
    seq_config = SequenceConfig(
        max_seq_len=training_config.max_seq_len,
        max_tasks=training_config.max_tasks,
        max_segments=training_config.max_segments,
        input_dim=training_config.input_dim,
        task_type=experiment_config.task_type,
        num_classes=experiment_config.num_classes,
    )

    # 创建划分器
    splitter = TwoByTwoSplitter(
        train_subjects=experiment_config.train_subjects,
        train_tasks=experiment_config.train_tasks,
        random_state=experiment_config.random_seed,
    )

    # 多次重复实验的结果收集
    all_repeat_results = []

    for repeat in range(experiment_config.n_repeats):
        logger.info(f'\n{"="*60}')
        logger.info(f'Repeat {repeat + 1}/{experiment_config.n_repeats}')
        logger.info(f'{"="*60}')

        # 执行划分
        splits = splitter.split(data)
        split_summary = splitter.get_split_summary(splits)

        logger.info('划分摘要:')
        for name, info in split_summary.items():
            if name != 'config':
                logger.info(f'  {name}: {info["samples"]} 样本, {info["subjects"]} 被试, {info["tasks"]} 任务')

        # 创建数据集
        train_dataset = LightweightGazeDataset(
            data=splits['train'],
            config=seq_config,
            fit_normalizer=True,
        )

        # 所有测试集使用相同的归一化参数
        test_datasets = {}
        for split_name in ['test1', 'test2', 'test3']:
            if splits[split_name]:
                test_datasets[split_name] = LightweightGazeDataset(
                    data=splits[split_name],
                    config=seq_config,
                    fit_normalizer=False,
                    normalizer_stats=train_dataset.stats,
                )

        # 获取目标域数据集（用于域迁移）
        if experiment_config.target_domain == 'all':
            target_data = splits['test1'] + splits['test2'] + splits['test3']
        else:
            target_data = splits.get(experiment_config.target_domain, splits['test2'])

        target_dataset = LightweightGazeDataset(
            data=target_data,
            config=seq_config,
            fit_normalizer=False,
            normalizer_stats=train_dataset.stats,
        )

        # 创建训练器并训练
        trainer = DomainAdaptationTrainer(training_config)

        if (experiment_config.task_type == 'classification'
                and experiment_config.enable_domain_adaptation):
            # 分类模式 + 域迁移
            train_metrics = trainer.train_with_domain_adaptation(
                train_dataset=train_dataset,
                target_dataset=target_dataset,
                val_dataset=test_datasets.get('test1', test_datasets.get('test2')),
                fold=repeat,
            )
        else:
            # 标准训练
            train_metrics = trainer.train(
                train_dataset,
                test_datasets.get('test1', test_datasets.get('test2')),
                fold=repeat,
            )

        # 评估所有测试集
        repeat_metrics = {'train': train_metrics}
        repeat_predictions = {}

        for split_name, dataset in test_datasets.items():
            if experiment_config.task_type == 'classification':
                # 分类模式评估
                eval_result = trainer.evaluate_with_confusion_matrix(dataset)
                metrics = {
                    'accuracy': eval_result['accuracy'],
                    'f1_weighted': eval_result['f1_weighted'],
                    'f1_macro': eval_result['f1_macro'],
                }
                predictions = eval_result['predictions']
                labels = eval_result['labels']
            else:
                # 回归模式评估
                predictions, labels, _ = trainer.predict(dataset)
                metrics = {
                    'r2': r2_score(labels, predictions),
                    'mae': mean_absolute_error(labels, predictions),
                    'rmse': np.sqrt(mean_squared_error(labels, predictions)),
                }

            repeat_metrics[split_name] = metrics

            # 收集预测结果
            for i, sample in enumerate(splits[split_name]):
                repeat_predictions[sample['subject_id']] = {
                    'subject_id': sample['subject_id'],
                    'split': split_name,
                    'true_label': float(labels[i]),
                    'predicted_label': float(predictions[i]),
                }

            # 打印结果
            if experiment_config.task_type == 'classification':
                logger.info(
                    f'{split_name}: Acc={metrics["accuracy"]:.4f}, '
                    f'F1={metrics["f1_weighted"]:.4f}'
                )
            else:
                logger.info(
                    f'{split_name}: R²={metrics["r2"]:.4f}, '
                    f'MAE={metrics["mae"]:.4f}'
                )

        all_repeat_results.append({
            'metrics': repeat_metrics,
            'predictions': repeat_predictions,
            'split_summary': split_summary,
        })

    # 汇总结果
    duration = time.time() - start_time

    # 计算平均指标
    avg_metrics = {}
    for split_name in ['train', 'test1', 'test2', 'test3']:
        split_metrics = [
            r['metrics'].get(split_name, {})
            for r in all_repeat_results
            if split_name in r['metrics']
        ]
        if split_metrics:
            if experiment_config.task_type == 'classification':
                avg_metrics[split_name] = {
                    'accuracy': np.mean([m.get('accuracy', 0) for m in split_metrics]),
                    'accuracy_std': np.std([m.get('accuracy', 0) for m in split_metrics]) if len(split_metrics) > 1 else 0,
                    'f1_weighted': np.mean([m.get('f1_weighted', 0) for m in split_metrics]),
                    'f1_weighted_std': np.std([m.get('f1_weighted', 0) for m in split_metrics]) if len(split_metrics) > 1 else 0,
                }
            else:
                avg_metrics[split_name] = {
                    'r2': np.mean([m.get('r2', 0) for m in split_metrics]),
                    'r2_std': np.std([m.get('r2', 0) for m in split_metrics]) if len(split_metrics) > 1 else 0,
                    'mae': np.mean([m.get('mae', 0) for m in split_metrics]),
                    'mae_std': np.std([m.get('mae', 0) for m in split_metrics]) if len(split_metrics) > 1 else 0,
                    'rmse': np.mean([m.get('rmse', 0) for m in split_metrics]),
                    'rmse_std': np.std([m.get('rmse', 0) for m in split_metrics]) if len(split_metrics) > 1 else 0,
                }

    # 添加描述
    metric_descriptions = {
        'train': '训练集',
        'test1': '跨被试泛化（新人旧题）',
        'test2': '跨任务泛化（旧人新题）',
        'test3': '双重泛化（新人新题）',
    }

    for split_name, desc in metric_descriptions.items():
        if split_name in avg_metrics:
            avg_metrics[split_name]['description'] = desc

    results = {
        'experiment_info': {
            'mode': experiment_config.mode,
            'task_type': experiment_config.task_type,
            'split_type': experiment_config.split_type,
            'train_subjects': experiment_config.train_subjects,
            'train_tasks': experiment_config.train_tasks,
            'n_repeats': experiment_config.n_repeats,
            'enable_domain_adaptation': experiment_config.enable_domain_adaptation,
            'target_domain': experiment_config.target_domain,
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': duration,
        },
        'split_summary': all_repeat_results[0]['split_summary'] if all_repeat_results else {},
        'metrics': avg_metrics,
        'repeat_results': [r['metrics'] for r in all_repeat_results],
        'predictions': all_repeat_results[-1]['predictions'] if all_repeat_results else {},
    }

    return results


def print_results(results: Dict, task_type: str) -> None:
    """打印结果"""
    print('\n' + '=' * 70)
    if task_type == 'classification':
        print('EXPERIMENT RESULTS: Domain Adaptive Hierarchical Transformer (Classification)')
    else:
        print('EXPERIMENT RESULTS: Hierarchical Transformer Network (Regression)')
    print('=' * 70)

    info = results['experiment_info']
    print(f'\n实验模式: {info["mode"]}')
    print(f'任务类型: {info["task_type"]}')
    print(f'划分策略: {info["split_type"]}')
    print(f'训练被试数: {info["train_subjects"]}')
    print(f'训练任务数: {info["train_tasks"]}')
    print(f'重复次数: {info["n_repeats"]}')

    if task_type == 'classification':
        print(f'域迁移: {info["enable_domain_adaptation"]}')
        print(f'目标域: {info["target_domain"]}')

    print(f'耗时: {info["duration_seconds"]:.1f}秒')

    print('\n' + '-' * 50)
    print('各测试集指标:')
    print('-' * 50)

    metrics = results['metrics']
    for split_name in ['train', 'test1', 'test2', 'test3']:
        if split_name in metrics:
            m = metrics[split_name]
            desc = m.get('description', split_name)
            print(f'\n  {split_name} ({desc}):')

            if task_type == 'classification':
                if 'accuracy_std' in m and m['accuracy_std'] > 0:
                    print(f'    Accuracy: {m["accuracy"]:.4f} +/- {m["accuracy_std"]:.4f}')
                    print(f'    F1 (weighted): {m["f1_weighted"]:.4f} +/- {m["f1_weighted_std"]:.4f}')
                else:
                    print(f'    Accuracy: {m["accuracy"]:.4f}')
                    print(f'    F1 (weighted): {m["f1_weighted"]:.4f}')
            else:
                if 'r2_std' in m and m['r2_std'] > 0:
                    print(f'    R2:   {m["r2"]:.4f} +/- {m["r2_std"]:.4f}')
                    print(f'    MAE:  {m["mae"]:.4f} +/- {m["mae_std"]:.4f}')
                    print(f'    RMSE: {m["rmse"]:.4f} +/- {m["rmse_std"]:.4f}')
                else:
                    print(f'    R2:   {m["r2"]:.4f}')
                    print(f'    MAE:  {m["mae"]:.4f}')
                    print(f'    RMSE: {m["rmse"]:.4f}')

    print('\n' + '=' * 70)


def save_results(
    results: Dict,
    output_dir: str,
    experiment_config: ExperimentConfig,
) -> None:
    """保存结果"""
    os.makedirs(output_dir, exist_ok=True)

    # 转换 numpy 类型为 Python 原生类型
    def convert_to_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj

    serializable_results = convert_to_serializable(results)

    # 保存结果 JSON
    results_path = os.path.join(output_dir, 'experiment_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    logger.info(f'结果已保存: {results_path}')

    # 保存预测结果 CSV
    if 'predictions' in results and results['predictions']:
        predictions_list = list(results['predictions'].values())
        predictions_df = pd.DataFrame(predictions_list)
        predictions_path = os.path.join(output_dir, 'predictions.csv')
        predictions_df.to_csv(predictions_path, index=False)
        logger.info(f'预测结果已保存: {predictions_path}')

    # 保存配置快照
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(experiment_config.to_dict(), f, indent=2, ensure_ascii=False)
    logger.info(f'配置已保存: {config_path}')


def main():
    """主函数"""
    print('=' * 70)
    print('Domain Adaptive Hierarchical Transformer')
    print('支持分类/回归双模式，分类模式下启用 CADT 域迁移')
    print('=' * 70)

    # 路径配置
    processed_data_path = 'data/processed/processed_data.pkl'
    output_base_dir = 'outputs/domain_adaptation'

    # 模型模式
    MODEL_MODE = os.environ.get('MODEL_MODE', 'full')

    # 实验配置（从环境变量读取）
    experiment_config = ExperimentConfig.from_env()

    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = experiment_config.experiment_name or f'{experiment_config.task_type}_{timestamp}'
    output_dir = os.path.join(output_base_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    # 打印配置
    print(f'\n{"="*50}')
    print(f'配置模式: {MODEL_MODE}')
    print(f'{"="*50}')
    print(f'预处理数据: {processed_data_path}')
    print(f'输出目录: {output_dir}')

    # 打印实验配置
    experiment_config.print_config()

    # GPU信息
    if torch.cuda.is_available():
        print(f'\nGPU: {torch.cuda.get_device_name(0)}')
        print(f'显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    else:
        print('\n警告: 没有可用的GPU!')

    # 检查预处理数据是否存在
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

    # 检查 category 字段
    has_category = all('category' in d for d in data)
    if experiment_config.task_type == 'classification' and not has_category:
        print('\n错误: 分类模式需要 category 字段，但数据中不存在!')
        print('请确保预处理时包含了分类标签。')
        return

    print(f'\n数据统计:')
    print(f'  被试: {len(data)}')
    print(f'  任务: {total_tasks}')
    print(f'  片段: {total_segments}')
    if has_category:
        category_counts = {}
        for d in data:
            cat = d.get('category', -1)
            category_counts[cat] = category_counts.get(cat, 0) + 1
        print(f'  类别分布: {category_counts}')

    # 获取训练配置
    training_config = get_model_config(MODEL_MODE, experiment_config, output_dir)

    # 打印模型配置
    print(f'\n模型配置:')
    print(f'  Batch Size: {training_config.batch_size}')
    print(f'  Max Tasks: {training_config.max_tasks}, Max Segments: {training_config.max_segments}')
    print(f'  Max Seq Len: {training_config.max_seq_len}')
    print(f'  Model: segment_d={training_config.segment_d_model}, task_d={training_config.task_d_model}')
    print(f'  Layers: segment={training_config.segment_num_layers}, task={training_config.task_num_layers}')

    # 运行实验
    if experiment_config.split_type == '2x2':
        results = run_2x2_experiment(data, training_config, experiment_config)
    else:
        print('K-Fold 暂不支持，请使用 2x2 划分策略')
        return

    # 打印结果
    print_results(results, experiment_config.task_type)

    # 保存结果
    save_results(results, output_dir, experiment_config)

    print('\n实验完成!')
    print(f'结果保存在: {output_dir}')


if __name__ == '__main__':
    main()
