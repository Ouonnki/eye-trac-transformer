# -*- coding: utf-8 -*-
"""
深度学习Transformer实验

使用层级Transformer网络进行眼动注意力预测实验。
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
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.dl_dataset import SequenceConfig, create_subject_splits, collate_fn
from src.models.dl_trainer import TrainingConfig, DeepLearningTrainer

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

        return {
            'segments': torch.from_numpy(segments),
            'segment_lengths': torch.from_numpy(segment_lengths),
            'segment_mask': torch.from_numpy(segment_mask),
            'task_lengths': torch.from_numpy(task_lengths),
            'task_mask': torch.from_numpy(task_mask),
            'label': torch.tensor(subject_data['label'], dtype=torch.float32),
            'subject_id': subject_data['subject_id'],
        }


def load_processed_data(data_path: str) -> List[Dict]:
    """加载预处理好的数据"""
    logger.info(f'Loading preprocessed data from: {data_path}')
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    logger.info(f'Loaded {len(data)} subjects')
    return data


def run_cross_validation(
    data: List[Dict],
    config: TrainingConfig,
    n_splits: int = 5,
) -> Dict:
    """运行交叉验证"""
    logger.info(f'Running {n_splits}-fold cross-validation...')

    # 获取被试ID列表
    subject_ids = [d['subject_id'] for d in data]
    subject_dict = {d['subject_id']: d for d in data}

    # 创建交叉验证划分
    splits = create_subject_splits(subject_ids, n_splits=n_splits, random_state=42)

    # 序列配置
    seq_config = SequenceConfig(
        max_seq_len=config.max_seq_len,
        max_tasks=config.max_tasks,
        max_segments=config.max_segments,
        input_dim=config.input_dim,
    )

    # 结果收集
    all_fold_metrics = []
    all_predictions = []
    all_labels = []
    all_subject_ids = []

    for fold, (train_ids, val_ids) in enumerate(splits):
        logger.info(f'\n{"="*60}')
        logger.info(f'Fold {fold + 1}/{n_splits}')
        logger.info(f'Train: {len(train_ids)} subjects, Val: {len(val_ids)} subjects')
        logger.info(f'{"="*60}')

        # 获取训练和验证数据
        train_data = [subject_dict[sid] for sid in train_ids]
        val_data = [subject_dict[sid] for sid in val_ids]

        # 创建数据集
        train_dataset = LightweightGazeDataset(
            data=train_data,
            config=seq_config,
            fit_normalizer=True,
        )

        val_dataset = LightweightGazeDataset(
            data=val_data,
            config=seq_config,
            fit_normalizer=False,
            normalizer_stats=train_dataset.stats,
        )

        # 创建训练器
        trainer = DeepLearningTrainer(config)

        # 训练
        fold_metrics = trainer.train(train_dataset, val_dataset, fold=fold)

        # 获取预测
        predictions, labels, _ = trainer.predict(val_dataset)

        # 记录结果
        all_fold_metrics.append(fold_metrics)
        all_predictions.extend(predictions)
        all_labels.extend(labels)
        all_subject_ids.extend(val_ids)

        logger.info(f'Fold {fold + 1} Results:')
        logger.info(f'  R2: {fold_metrics["r2"]:.4f}')
        logger.info(f'  MAE: {fold_metrics["mae"]:.4f}')
        logger.info(f'  RMSE: {fold_metrics["rmse"]:.4f}')

    # 汇总结果
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    overall_metrics = {
        'r2': r2_score(all_labels, all_predictions),
        'mae': mean_absolute_error(all_labels, all_predictions),
        'rmse': np.sqrt(mean_squared_error(all_labels, all_predictions)),
    }

    fold_r2s = [m['r2'] for m in all_fold_metrics]
    fold_maes = [m['mae'] for m in all_fold_metrics]

    results = {
        'overall_metrics': overall_metrics,
        'fold_metrics': all_fold_metrics,
        'fold_r2_mean': np.mean(fold_r2s),
        'fold_r2_std': np.std(fold_r2s),
        'fold_mae_mean': np.mean(fold_maes),
        'fold_mae_std': np.std(fold_maes),
        'predictions': all_predictions.tolist(),
        'labels': all_labels.tolist(),
        'subject_ids': all_subject_ids,
    }

    return results


def print_results(results: Dict) -> None:
    """打印结果"""
    print('\n' + '=' * 70)
    print('EXPERIMENT RESULTS: Hierarchical Transformer Network')
    print('=' * 70)

    print('\nOverall Metrics (Cross-Validation):')
    print('-' * 40)
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


def save_results(results: Dict, output_dir: str) -> None:
    """保存结果"""
    os.makedirs(output_dir, exist_ok=True)

    # 保存JSON结果
    results_path = os.path.join(output_dir, 'dl_experiment_results.json')

    serializable_results = {
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
    predictions_df = pd.DataFrame({
        'subject_id': results['subject_ids'],
        'true_score': results['labels'],
        'predicted_score': results['predictions'],
    })
    predictions_path = os.path.join(output_dir, 'dl_predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    logger.info(f'Predictions saved to {predictions_path}')


def main():
    """主函数"""
    print('=' * 70)
    print('Hierarchical Transformer Network for Gaze Attention Prediction')
    print('=' * 70)

    # ============================================================
    # 路径配置
    # ============================================================
    processed_data_path = 'data/processed/processed_data.pkl'  # 预处理数据路径
    output_dir = 'outputs/dl_models'

    # ============================================================
    # 训练配置
    # ============================================================
    # 选择配置模式: 'full' (完整模型), 'medium' (中等), 'light' (轻量)
    # 如果显存不足，从 full -> medium -> light 依次尝试
    MODEL_MODE = os.environ.get('MODEL_MODE', 'medium')

    if MODEL_MODE == 'full':
        # 完整模型配置（需要较大显存，约24GB+）
        config = TrainingConfig(
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
            use_gradient_checkpointing=True,  # 启用梯度检查点节省显存
            num_workers=4,
            pin_memory=True,
            output_dir=output_dir,
            save_best=True,
        )
    elif MODEL_MODE == 'medium':
        # 中等配置（适合RTX 3090 24GB）
        config = TrainingConfig(
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
            use_gradient_checkpointing=True,  # 启用梯度检查点节省显存
            num_workers=4,
            pin_memory=True,
            output_dir=output_dir,
            save_best=True,
        )
    else:  # 'light'
        # 轻量配置（显存不足时的备选方案）
        config = TrainingConfig(
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
            use_gradient_checkpointing=False,  # 轻量配置无需梯度检查点
            num_workers=2,
            pin_memory=True,
            output_dir=output_dir,
            save_best=True,
        )

    # 打印配置
    print(f'\n{"="*50}')
    print(f'Configuration Mode: {MODEL_MODE}')
    print(f'{"="*50}')
    print(f'Processed Data: {processed_data_path}')
    print(f'Output Dir: {output_dir}')
    print(f'Batch Size: {config.batch_size}')
    print(f'Max Tasks: {config.max_tasks}, Max Segments: {config.max_segments}')
    print(f'Max Seq Len: {config.max_seq_len}')
    print(f'Model: segment_d={config.segment_d_model}, task_d={config.task_d_model}')
    print(f'Layers: segment={config.segment_num_layers}, task={config.task_num_layers}')

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

    # 运行交叉验证
    results = run_cross_validation(data, config, n_splits=5)

    # 打印结果
    print_results(results)

    # 保存结果
    save_results(results, output_dir)

    print('\nExperiment completed!')


if __name__ == '__main__':
    main()
