# -*- coding: utf-8 -*-
"""
深度学习Transformer实验

使用层级Transformer网络进行眼动注意力预测实验。
支持两种实验设计：
- 2×2 矩阵划分：验证跨被试和跨任务泛化能力
- K-Fold 交叉验证：传统交叉验证

需要先运行 scripts/preprocess_data.py 预处理数据。
"""

import os
import sys
import logging
import json
import pickle
import time
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, f1_score
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.dl_dataset import SequenceConfig, collate_fn
from src.models.dl_trainer import DeepLearningTrainer
from src.data.split_strategy import TwoByTwoSplitter
from src.config import UnifiedConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LightweightGazeDataset(Dataset):
    """
    轻量级眼动数据集

    直接使用预处理好的numpy数据，不需要SubjectData对象。
    """

    def __init__(
        self,
        data: List[Dict],
        config: SequenceConfig,
        fit_normalizer: bool = False,
        normalizer_stats: Optional[Dict] = None,
        task_type: str = 'classification',  # 任务类型
        use_task_embedding: bool = False,  # 是否使用任务嵌入
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

        # 预归一化所有数据（避免 __getitem__ 中重复计算）
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
        """归一化特征（原地修改，避免复制）"""
        if len(features) > 1:
            features[1:, 2] = (features[1:, 2] - self.stats['dt_mean']) / self.stats['dt_std']
            features[1:, 3] = (features[1:, 3] - self.stats['velocity_mean']) / self.stats['velocity_std']
            features[1:, 4] = (features[1:, 4] - self.stats['acceleration_mean']) / self.stats['acceleration_std']
        return features

    def _prenormalize_all(self):
        """预归一化所有数据，避免在 __getitem__ 中重复计算"""
        for subject_data in self.data:
            for task in subject_data['tasks']:
                for i, features in enumerate(task['segments']):
                    if len(features) > 1:
                        # 原地归一化
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

        # 填充数据（数据已在 __init__ 中预归一化）
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
            # 分类任务：类别标签需要是 0-indexed
            label = torch.tensor(subject_data['category'] - 1, dtype=torch.long)
        else:
            # 回归任务：使用总分
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

        # 只有在启用任务嵌入时才返回任务条件
        if self.use_task_embedding:
            result['task_conditions'] = torch.from_numpy(task_conditions)

        return result


def load_processed_data(data_path: str) -> List[Dict]:
    """加载预处理好的数据"""
    logger.info(f'Loading preprocessed data from: {data_path}')
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    logger.info(f'Loaded {len(data)} subjects')
    return data


def run_2x2_experiment(
    data: List[Dict],
    config: UnifiedConfig,
    seq_config: SequenceConfig,
) -> Dict:
    """
    运行 2×2 实验设计

    Args:
        data: 预处理后的数据
        config: 统一配置
        seq_config: 序列配置

    Returns:
        实验结果字典
    """
    start_time = time.time()

    logger.info(f'Running 2×2 experiment design...')
    logger.info(f'Train subjects: {config.experiment.train_subjects}')
    logger.info(f'Train tasks: {config.experiment.train_tasks}')
    logger.info(f'Repeats: {config.experiment.n_repeats}')

    # 创建划分器
    splitter = TwoByTwoSplitter(
        train_subjects=config.experiment.train_subjects,
        train_tasks=config.experiment.train_tasks,
        random_state=config.experiment.random_seed,
    )

    # 多次重复实验的结果收集
    all_repeat_results = []

    for repeat in range(config.experiment.n_repeats):
        logger.info(f'\n{"="*60}')
        logger.info(f'Repeat {repeat + 1}/{config.experiment.n_repeats}')
        logger.info(f'{"="*60}')

        # 执行划分
        splits = splitter.split(data)
        split_summary = splitter.get_split_summary(splits)

        logger.info(f'Split Summary:')
        for name, info in split_summary.items():
            if name != 'config':
                logger.info(f'  {name}: {info["samples"]} samples, {info["subjects"]} subjects, {info["tasks"]} tasks')

        # 创建数据集
        train_dataset = LightweightGazeDataset(
            data=splits['train'],
            config=seq_config,
            fit_normalizer=True,
            task_type=config.task.type,
            use_task_embedding=config.model.use_task_embedding,
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
                    task_type=config.task.type,
                    use_task_embedding=config.model.use_task_embedding,
                )

        # 创建训练器并训练
        trainer = DeepLearningTrainer(config, seq_config)
        train_metrics = trainer.train(
            train_dataset,
            test_datasets.get('test1', test_datasets.get('test2')),  # 使用 test1 作为验证集
            fold=repeat,
        )

        # 评估所有测试集
        repeat_metrics = {'train': train_metrics}
        repeat_predictions = {}

        for split_name, dataset in test_datasets.items():
            predictions, labels, _ = trainer.predict(dataset)

            # 根据任务类型计算指标
            if config.task.type == 'classification':
                # 分类任务：predictions 是 logits，需要 argmax
                preds = np.argmax(predictions, axis=1) if len(predictions.shape) > 1 else predictions
                metrics = {
                    'accuracy': accuracy_score(labels, preds),
                    'f1': f1_score(labels, preds, average='macro'),
                }
            else:
                # 回归任务
                metrics = {
                    'r2': r2_score(labels, predictions),
                    'mae': mean_absolute_error(labels, predictions),
                    'rmse': np.sqrt(mean_squared_error(labels, predictions)),
                }
            repeat_metrics[split_name] = metrics

            # 收集预测结果
            for i, sample in enumerate(splits[split_name]):
                if config.task.type == 'classification':
                    pred_class = int(preds[i]) + 1  # 转回 1-indexed
                    true_class = int(labels[i]) + 1
                    repeat_predictions[sample['subject_id']] = {
                        'subject_id': sample['subject_id'],
                        'split': split_name,
                        'true_class': true_class,
                        'predicted_class': pred_class,
                        'correct': pred_class == true_class,
                    }
                else:
                    repeat_predictions[sample['subject_id']] = {
                        'subject_id': sample['subject_id'],
                        'split': split_name,
                        'true_score': float(labels[i]),
                        'predicted_score': float(predictions[i]),
                        'error': float(abs(predictions[i] - labels[i])),
                    }

            logger.info(f'{split_name} Results:')
            if config.task.type == 'classification':
                logger.info(f'  Accuracy: {metrics["accuracy"]:.4f}')
                logger.info(f'  F1: {metrics["f1"]:.4f}')
            else:
                logger.info(f'  R²: {metrics["r2"]:.4f}')
                logger.info(f'  MAE: {metrics["mae"]:.4f}')
                logger.info(f'  RMSE: {metrics["rmse"]:.4f}')

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
        split_metrics = [r['metrics'].get(split_name, {}) for r in all_repeat_results if split_name in r['metrics']]
        if split_metrics:
            # 根据任务类型计算不同的平均指标
            if config.task.type == 'classification':
                avg_metrics[split_name] = {
                    'accuracy': np.mean([m['accuracy'] for m in split_metrics]),
                    'accuracy_std': np.std([m['accuracy'] for m in split_metrics]) if len(split_metrics) > 1 else 0,
                    'f1': np.mean([m['f1'] for m in split_metrics]),
                    'f1_std': np.std([m['f1'] for m in split_metrics]) if len(split_metrics) > 1 else 0,
                }
            else:
                avg_metrics[split_name] = {
                    'r2': np.mean([m['r2'] for m in split_metrics]),
                    'r2_std': np.std([m['r2'] for m in split_metrics]) if len(split_metrics) > 1 else 0,
                    'mae': np.mean([m['mae'] for m in split_metrics]),
                    'mae_std': np.std([m['mae'] for m in split_metrics]) if len(split_metrics) > 1 else 0,
                    'rmse': np.mean([m['rmse'] for m in split_metrics]),
                    'rmse_std': np.std([m['rmse'] for m in split_metrics]) if len(split_metrics) > 1 else 0,
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
        'task_type': config.task.type,  # 添加任务类型
        'experiment_info': {
            'train_subjects': config.experiment.train_subjects,
            'train_tasks': config.experiment.train_tasks,
            'n_repeats': config.experiment.n_repeats,
            'random_seed': config.experiment.random_seed,
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': duration,
        },
        'split_summary': all_repeat_results[0]['split_summary'] if all_repeat_results else {},
        'metrics': avg_metrics,
        'repeat_results': [r['metrics'] for r in all_repeat_results],
        'predictions': all_repeat_results[-1]['predictions'] if all_repeat_results else {},
    }

    return results


def print_results(results: Dict) -> None:
    """打印结果"""
    print('\n' + '=' * 70)
    print('EXPERIMENT RESULTS: Hierarchical Transformer Network')
    print('=' * 70)

    # 检测任务类型
    task_type = results.get('task_type', 'regression')

    # 检测结果类型
    if 'experiment_info' in results:
        # 2×2 实验结果
        info = results['experiment_info']
        print(f'\n划分策略: 2×2 矩阵')
        print(f'训练被试数: {info["train_subjects"]}')
        print(f'训练任务数: {info["train_tasks"]}')
        print(f'重复次数: {info["n_repeats"]}')
        if 'random_seed' in info:
            print(f'随机种子: {info["random_seed"]}')
        print(f'耗时: {info["duration_seconds"]:.1f}秒')
        print(f'任务类型: {task_type}')

        print('\n' + '-' * 50)
        print('各测试集指标:')
        print('-' * 50)

        metrics = results['metrics']
        for split_name in ['train', 'test1', 'test2', 'test3']:
            if split_name in metrics:
                m = metrics[split_name]
                desc = m.get('description', split_name)
                print(f'\n  {split_name} ({desc}):')
                # 根据指标类型显示
                if 'accuracy' in m:
                    # 分类任务
                    if 'accuracy_std' in m and m['accuracy_std'] > 0:
                        print(f'    Accuracy: {m["accuracy"]:.4f} +/- {m["accuracy_std"]:.4f}')
                        print(f'    F1:       {m["f1"]:.4f} +/- {m["f1_std"]:.4f}')
                    else:
                        print(f'    Accuracy: {m["accuracy"]:.4f}')
                        print(f'    F1:       {m["f1"]:.4f}')
                else:
                    # 回归任务
                    if 'r2_std' in m and m['r2_std'] > 0:
                        print(f'    R2:   {m["r2"]:.4f} +/- {m["r2_std"]:.4f}')
                        print(f'    MAE:  {m["mae"]:.4f} +/- {m["mae_std"]:.4f}')
                        print(f'    RMSE: {m["rmse"]:.4f} +/- {m["rmse_std"]:.4f}')
                    else:
                        print(f'    R2:   {m["r2"]:.4f}')
                        print(f'    MAE:  {m["mae"]:.4f}')
                        print(f'    RMSE: {m["rmse"]:.4f}')

    else:
        # K-Fold 交叉验证结果
        print(f'\n任务类型: {task_type}')
        print('\nOverall Metrics (Cross-Validation):')
        print('-' * 40)

        if task_type == 'classification':
            print(f"  Accuracy: {results['overall_metrics']['accuracy']:.4f}")
            print(f"  F1:       {results['overall_metrics']['f1']:.4f}")

            print('\nFold-wise Metrics:')
            print('-' * 40)
            print(f"  Accuracy: {results['fold_accuracy_mean']:.4f} +/- {results['fold_accuracy_std']:.4f}")
            print(f"  F1:       {results['fold_f1_mean']:.4f} +/- {results['fold_f1_std']:.4f}")

            print('\nPer-Fold Results:')
            print('-' * 40)
            for i, metrics in enumerate(results['fold_metrics']):
                print(f"  Fold {i+1}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
        else:
            print(f"  R2:   {results['overall_metrics']['r2']:.4f}")
            print(f"  MAE:  {results['overall_metrics']['mae']:.4f}")
            print(f"  RMSE: {results['overall_metrics']['rmse']:.4f}")

            print('\nFold-wise Metrics:')
            print('-' * 40)
            print(f"  R2:  {results['fold_r2_mean']:.4f} +/- {results['fold_r2_std']:.4f}")
            print(f"  MAE: {results['fold_mae_mean']:.4f} +/- {results['fold_mae_std']:.4f}")

            print('\nPer-Fold Results:')
            print('-' * 40)
            for i, metrics in enumerate(results['fold_metrics']):
                print(f"  Fold {i+1}: R2={metrics['r2']:.4f}, MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}")

    print('\n' + '=' * 70)


def save_results(results: Dict, output_dir: str, config: UnifiedConfig) -> None:
    """保存结果"""
    os.makedirs(output_dir, exist_ok=True)

    # 检测结果类型
    if 'experiment_info' in results:
        # 2×2 实验结果
        results_path = os.path.join(output_dir, 'experiment_results.json')

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

        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        logger.info(f'Results saved to {results_path}')

        # 保存预测结果 CSV（带 split 标记）
        if 'predictions' in results and results['predictions']:
            predictions_list = list(results['predictions'].values())
            predictions_df = pd.DataFrame(predictions_list)
            # 根据任务类型选择不同的列
            if 'true_class' in predictions_list[0]:
                # 分类任务
                predictions_df = predictions_df[['subject_id', 'split', 'true_class', 'predicted_class', 'correct']]
            else:
                # 回归任务
                predictions_df = predictions_df[['subject_id', 'split', 'true_score', 'predicted_score', 'error']]
            predictions_path = os.path.join(output_dir, 'predictions.csv')
            predictions_df.to_csv(predictions_path, index=False)
            logger.info(f'Predictions saved to {predictions_path}')

        # 保存配置快照
        config_path = os.path.join(output_dir, 'config.json')
        config.save_snapshot(output_dir)
        logger.info(f'Config saved to {config_path}')

    else:
        # K-Fold 交叉验证结果（保持兼容）
        results_path = os.path.join(output_dir, 'dl_experiment_results.json')

        # 根据任务类型构建不同的序列化结果
        task_type = results.get('task_type', 'regression')

        if task_type == 'classification':
            serializable_results = {
                'task_type': 'classification',
                'overall_metrics': results['overall_metrics'],
                'fold_accuracy_mean': results['fold_accuracy_mean'],
                'fold_accuracy_std': results['fold_accuracy_std'],
                'fold_f1_mean': results['fold_f1_mean'],
                'fold_f1_std': results['fold_f1_std'],
                'fold_metrics': results['fold_metrics'],
                'timestamp': datetime.now().isoformat(),
            }
        else:
            serializable_results = {
                'task_type': 'regression',
                'overall_metrics': results['overall_metrics'],
                'fold_r2_mean': results['fold_r2_mean'],
                'fold_r2_std': results['fold_r2_std'],
                'fold_mae_mean': results['fold_mae_mean'],
                'fold_mae_std': results['fold_mae_std'],
                'fold_metrics': results['fold_metrics'],
                'timestamp': datetime.now().isoformat(),
            }

        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        logger.info(f'Results saved to {results_path}')

        # 保存预测结果CSV
        if task_type == 'classification':
            # 分类任务：将预测类别转回 1-indexed
            predictions_array = np.array(results['predictions'])
            if len(predictions_array.shape) > 1:
                preds = np.argmax(predictions_array, axis=1) + 1  # 转回 1-indexed
            else:
                preds = predictions_array + 1
            labels = np.array(results['labels']) + 1  # 转回 1-indexed

            predictions_df = pd.DataFrame({
                'subject_id': results['subject_ids'],
                'true_class': labels.astype(int),
                'predicted_class': preds.astype(int),
                'correct': (preds == labels).astype(int),
            })
        else:
            predictions_df = pd.DataFrame({
                'subject_id': results['subject_ids'],
                'true_score': results['labels'],
                'predicted_score': results['predictions'],
            })
        predictions_path = os.path.join(output_dir, 'dl_predictions.csv')
        predictions_df.to_csv(predictions_path, index=False)
        logger.info(f'Predictions saved to {predictions_path}')


def load_config(config_path: Optional[str] = None) -> tuple[UnifiedConfig, SequenceConfig]:
    """
    加载训练配置

    从 JSON 文件加载配置，默认使用 configs/default.json

    Args:
        config_path: 配置文件路径

    Returns:
        (UnifiedConfig 实例, SequenceConfig 实例)
    """
    # 确定配置文件路径
    if config_path is None:
        config_path = os.environ.get('CONFIG', 'configs/default.json')

    # 加载配置
    logger.info(f'Loading config from: {config_path}')
    config = UnifiedConfig.from_json(config_path)

    # 创建序列配置（数据属性，暂时使用默认值）
    seq_config = SequenceConfig()

    return config, seq_config


def main():
    """主函数"""
    print('=' * 70)
    print('Hierarchical Transformer Network for Gaze Attention Prediction')
    print('=' * 70)

    # ============================================================
    # 路径配置
    # ============================================================
    processed_data_path = 'data/processed/processed_data.pkl'  # 预处理数据路径
    base_output_dir = 'outputs/dl_models'  # 基础输出目录

    # ============================================================
    # 加载配置
    # ============================================================
    config, seq_config = load_config()

    # 打印配置摘要
    print(f'\n{"="*50}')
    print(f'Task Type: {config.task.type}')
    print(f'{"="*50}')
    print(f'Processed Data: {processed_data_path}')
    print(f'Batch Size: {config.training.batch_size}')
    print(f'Model: segment_d={config.model.segment_d_model}, task_d={config.model.task_d_model}')
    print(f'Layers: segment={config.model.segment_num_layers}, task={config.model.task_num_layers}')

    # GPU信息
    if torch.cuda.is_available():
        print(f'\nGPU: {torch.cuda.get_device_name(0)}')
        print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    else:
        print('\nWARNING: No GPU available!')

    # 检查预处理数据是否存在
    if not os.path.exists(processed_data_path):
        print(f'\nERROR: 预处理数据不存在: {processed_data_path}')
        print('请先运行预处理脚本:')
        print('  python scripts/preprocess_data.py --data_dir <数据目录>')
        return

    # 加载数据
    data = load_processed_data(processed_data_path)

    if len(data) == 0:
        print('ERROR: No data loaded!')
        return

    # 统计
    total_tasks = sum(len(d['tasks']) for d in data)
    total_segments = sum(
        sum(len(t['segments']) for t in d['tasks'])
        for d in data
    )
    print(f'\nData Statistics:')
    print(f'  Subjects: {len(data)}')
    print(f'  Tasks: {total_tasks}')
    print(f'  Segments: {total_segments}')

    # 生成带时间戳的输出目录
    base_output_dir = config.experiment.output_dir
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    task_type = config.task.type
    output_dir = os.path.join(base_output_dir, f'{task_type}_{timestamp}')
    config.experiment.output_dir = output_dir

    # 打印实验配置
    print('\n' + '=' * 60)
    print('实验配置')
    print('=' * 60)
    print(f'  划分策略: 2×2 矩阵')
    print(f'  训练被试数: {config.experiment.train_subjects}')
    print(f'  训练任务数: {config.experiment.train_tasks}')
    print(f'  重复次数: {config.experiment.n_repeats}')
    print(f'  随机种子: {config.experiment.random_seed}')
    print(f'  保存图表: {config.output.save_figures}')
    print(f'  输出目录: {config.experiment.output_dir}')
    print('=' * 60)

    # ============================================================
    # 运行实验
    # ============================================================
    # 运行 2×2 矩阵划分实验
    results = run_2x2_experiment(data, config, seq_config)

    # 打印结果
    print_results(results)

    # 保存结果
    save_results(results, output_dir, config)

    print('\nExperiment completed!')


if __name__ == '__main__':
    main()
