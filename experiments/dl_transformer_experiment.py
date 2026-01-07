# -*- coding: utf-8 -*-
"""
深度学习Transformer实验

使用层级Transformer网络进行眼动注意力预测实验。
"""

import os
import sys
import logging
import json
import pickle
from datetime import datetime
from typing import List, Dict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import DataLoader as GazeDataLoader
from src.data.preprocessor import GazePreprocessor
from src.data.schemas import SubjectData
from src.segmentation.event_segmenter import AdaptiveSegmenter
from src.models.dl_dataset import (
    SequenceConfig,
    SequenceFeatureExtractor,
    HierarchicalGazeDataset,
    create_subject_splits,
)
from src.models.dl_trainer import TrainingConfig, DeepLearningTrainer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_all_subjects(
    data_dir: str,
    screen_width: int = 1920,
    screen_height: int = 1080,
    use_cache: bool = True,
    cache_dir: str = None,
) -> List[SubjectData]:
    """
    加载所有被试数据（支持缓存加速）

    Args:
        data_dir: 数据目录
        screen_width: 屏幕宽度
        screen_height: 屏幕高度
        use_cache: 是否使用缓存（首次加载后保存，后续直接读取）
        cache_dir: 缓存目录，默认为 data_dir 同级的 .cache 目录

    Returns:
        被试数据列表
    """
    # 设置缓存路径
    if cache_dir is None:
        cache_dir = Path(data_dir).parent / '.cache'
    else:
        cache_dir = Path(cache_dir)

    cache_file = cache_dir / 'processed_subjects.pkl'

    # 尝试从缓存加载
    if use_cache and cache_file.exists():
        logger.info(f'Loading from cache: {cache_file}')
        try:
            with open(cache_file, 'rb') as f:
                subjects = pickle.load(f)
            logger.info(f'Loaded {len(subjects)} subjects from cache')
            return subjects
        except Exception as e:
            logger.warning(f'Cache load failed: {e}, reloading from raw data...')

    # 从原始数据加载
    logger.info('Loading subject data from raw files...')

    loader = GazeDataLoader(data_dir)
    loader.load_labels()
    loader.load_tasks()
    preprocessor = GazePreprocessor(screen_width=screen_width, screen_height=screen_height)

    subjects = []
    subject_ids = loader.get_all_subject_ids()

    # 使用进度条显示加载进度
    for subject_id in tqdm(subject_ids, desc='Loading subjects'):
        try:
            subject = loader.load_subject(subject_id)

            for trial in subject.trials:
                preprocessor.preprocess_trial(trial)

                if trial.raw_gaze_points or hasattr(trial, 'gaze_trajectory'):
                    segmenter = AdaptiveSegmenter(
                        task_config=trial.config,
                        screen_width=screen_width,
                        screen_height=screen_height,
                    )
                    gaze_trajectory = getattr(trial, 'gaze_trajectory', None)
                    gaze_points = gaze_trajectory if gaze_trajectory else trial.raw_gaze_points
                    if gaze_points:
                        segments = segmenter.segment(gaze_points, gaze_trajectory)
                        trial.segments = segments
                    else:
                        trial.segments = []
                else:
                    trial.segments = []

            valid_trials = [t for t in subject.trials if len(t.segments) > 0]
            if len(valid_trials) > 0:
                subject.trials = valid_trials
                subjects.append(subject)

        except Exception as e:
            logger.warning(f'Failed to load subject {subject_id}: {e}')
            continue

    logger.info(f'Loaded {len(subjects)} subjects from raw data')

    # 保存到缓存
    if use_cache and len(subjects) > 0:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(subjects, f)
            logger.info(f'Cache saved to: {cache_file}')
        except Exception as e:
            logger.warning(f'Failed to save cache: {e}')

    return subjects


def run_cross_validation(
    subjects: List[SubjectData],
    config: TrainingConfig,
    n_splits: int = 5,
) -> Dict:
    """
    运行交叉验证

    Args:
        subjects: 被试数据列表
        config: 训练配置
        n_splits: 折数

    Returns:
        实验结果
    """
    logger.info(f'Running {n_splits}-fold cross-validation...')

    # 获取被试ID列表
    subject_ids = [s.subject_id for s in subjects]
    subject_dict = {s.subject_id: s for s in subjects}

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
        train_subjects = [subject_dict[sid] for sid in train_ids]
        val_subjects = [subject_dict[sid] for sid in val_ids]

        # 创建特征提取器
        feature_extractor = SequenceFeatureExtractor(seq_config)

        # 创建数据集（在训练集上拟合归一化器）
        train_dataset = HierarchicalGazeDataset(
            subjects=train_subjects,
            config=seq_config,
            feature_extractor=feature_extractor,
            fit_normalizer=True,
        )

        # 使用训练集的归一化器
        val_dataset = HierarchicalGazeDataset(
            subjects=val_subjects,
            config=seq_config,
            feature_extractor=feature_extractor,
            fit_normalizer=False,
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

    # 创建可序列化的结果副本
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
    # 路径配置（服务器上需要修改）
    # 注意：输出放到 /data 分区，因为 / 分区只剩 7.5G
    # ============================================================
    data_dir = 'data/gaze_trajectory_data'  # 修改为服务器实际数据路径
    output_dir = 'outputs/dl_models'   # 输出到 /data 分区（有186G可用）

    # ============================================================
    # 服务器优化配置
    # 硬件：单张 RTX 3090 (24GB) + 16核 Xeon + 31GB 内存
    # ============================================================
    config = TrainingConfig(
        # 模型参数（充分利用24GB显存）
        input_dim=7,
        segment_d_model=128,       # 增大：64 -> 128
        segment_nhead=8,           # 增大：4 -> 8
        segment_num_layers=6,      # 增大：4 -> 6
        task_d_model=256,          # 增大：128 -> 256
        task_nhead=8,              # 增大：4 -> 8
        task_num_layers=4,         # 增大：2 -> 4
        attention_dim=64,          # 增大：32 -> 64
        dropout=0.1,
        max_seq_len=100,
        max_tasks=30,
        max_segments=30,

        # 训练参数（针对单张3090优化）
        batch_size=16,             # 单卡保守值（可尝试增大到24）
        learning_rate=1e-4,        # 单卡标准学习率
        weight_decay=1e-4,
        warmup_epochs=5,
        epochs=100,
        patience=15,
        grad_clip=1.0,

        # 加速配置（单GPU + AMP）
        use_multi_gpu=False,       # 单GPU，关闭DataParallel
        use_amp=True,              # 混合精度（FP16加速约1.5-2倍）
        num_workers=8,             # 8个数据加载进程（16核的一半）
        pin_memory=True,           # 锁页内存加速GPU传输

        # 输出
        output_dir=output_dir,
        save_best=True,
    )

    # 打印配置信息
    print(f'\n{"="*50}')
    print('Training Configuration:')
    print(f'{"="*50}')
    print(f'Data Dir: {data_dir}')
    print(f'Output Dir: {output_dir}')
    print(f'Batch Size: {config.batch_size}')
    print(f'Learning Rate: {config.learning_rate}')
    print(f'Mixed Precision (AMP): {config.use_amp}')
    print(f'Num Workers: {config.num_workers}')
    print(f'Model: segment_d={config.segment_d_model}, task_d={config.task_d_model}')
    print(f'Epochs: {config.epochs}, Patience: {config.patience}')

    # 检查GPU配置
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f'\n{"="*40}')
        print('GPU Information:')
        print(f'{"="*40}')
        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f'GPU {i}: {props.name}')
            print(f'  Memory: {props.total_memory / 1e9:.1f} GB')
        print(f'Total GPUs: {n_gpus}')
    else:
        print('\nWARNING: No GPU available, training will be slow!')

    # 加载数据
    subjects = load_all_subjects(data_dir)

    if len(subjects) == 0:
        print('ERROR: No subjects loaded!')
        return

    # 统计信息
    total_trials = sum(len(s.trials) for s in subjects)
    total_segments = sum(
        sum(len(t.segments) for t in s.trials)
        for s in subjects
    )
    print(f'\nData Statistics:')
    print(f'  Subjects: {len(subjects)}')
    print(f'  Trials: {total_trials}')
    print(f'  Segments: {total_segments}')

    # 运行交叉验证
    results = run_cross_validation(subjects, config, n_splits=5)

    # 打印结果
    print_results(results)

    # 保存结果
    save_results(results, output_dir)

    print('\nExperiment completed!')


if __name__ == '__main__':
    main()
