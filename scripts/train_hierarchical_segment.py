# -*- coding: utf-8 -*-
"""
层级-片段联合模型训练脚本

使用示例:
    python scripts/train_hierarchical_segment.py \
        --config configs/hierarchical_segment.json \
        --data data/processed/processed_data.pkl

    # 调整损失权重
    python scripts/train_hierarchical_segment.py \
        --config configs/hierarchical_segment.json \
        --data data/processed/processed_data.pkl \
        --segment-weight 0.3 --hierarchical-weight 0.7
"""

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.config import UnifiedConfig
from src.models.dl_dataset import SequenceConfig

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HierarchicalSegmentDataset(Dataset):
    """
    层级-片段联合模型数据集

    兼容 dl_transformer_experiment.py 的预处理数据格式
    """

    def __init__(
        self,
        data: List[Dict],
        config: SequenceConfig,
        fit_normalizer: bool = False,
        normalizer_stats: Optional[Dict] = None,
    ):
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

        logger.info(f"归一化统计量: dt_mean={self.stats['dt_mean']:.4f}, "
                   f"velocity_mean={self.stats['velocity_mean']:.4f}")

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """归一化特征"""
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
                        task['segments'][i] = self._normalize(features)

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
        task_mask = np.zeros(self.config.max_tasks, dtype=np.bool_)

        # 任务条件
        task_conditions = np.zeros((self.config.max_tasks, 5), dtype=np.int64)

        # 填充数据
        for t_idx, task in enumerate(subject_data['tasks']):
            if t_idx >= self.config.max_tasks:
                break

            task_mask[t_idx] = True

            # 任务条件
            if 'task_conditions' in task:
                tc = task['task_conditions']
                task_conditions[t_idx] = [
                    tc.get('grid_scale', 3),
                    tc.get('continuous_thinking', 0),
                    tc.get('click_disappear', 0),
                    tc.get('has_distractor', 0),
                    tc.get('has_task_distractor', 0),
                ]

            # 片段
            for s_idx, features in enumerate(task['segments']):
                if s_idx >= self.config.max_segments:
                    break

                seq_len = min(len(features), self.config.max_seq_len)
                if seq_len > 0:
                    segments[t_idx, s_idx, :seq_len, :] = features[:seq_len, :self.config.input_dim]
                    segment_lengths[t_idx, s_idx] = seq_len
                    segment_mask[t_idx, s_idx] = True

        # 标签（分类任务使用 category，回归任务使用 label）
        if 'category' in subject_data:
            label = subject_data['category'] - 1  # 转换为 0-based
            label_tensor = torch.tensor(label, dtype=torch.long)
        else:
            label = subject_data.get('label', 0.0)
            label_tensor = torch.tensor(label, dtype=torch.float32)

        # 片段级标签：使用被试标签填充所有有效片段（方案1）
        # 创建一个与 segment_mask 形状相同的标签张量
        if 'category' in subject_data:
            # 分类任务：使用 int64，无效标签用 -100 (pytorch ignore_index)
            segment_labels = np.full(
                (self.config.max_tasks, self.config.max_segments),
                label,  # 使用被试标签
                dtype=np.int64
            )
            # 将无效片段的标签设为 -100（cross_entropy 的 ignore_index）
            segment_labels[~segment_mask] = -100
        else:
            # 回归任务：使用 float32，无效标签用 -1e9（会被 mask 忽略）
            segment_labels = np.full(
                (self.config.max_tasks, self.config.max_segments),
                label,  # 使用被试标签
                dtype=np.float32
            )
            # 将无效片段的标签设为 nan 或一个极值
            segment_labels[~segment_mask] = -1e9

        return {
            'segments': torch.from_numpy(segments),
            'segment_lengths': torch.from_numpy(segment_lengths),
            'segment_mask': torch.from_numpy(segment_mask),
            'task_mask': torch.from_numpy(task_mask),
            'task_conditions': torch.from_numpy(task_conditions),
            'label': label_tensor,
            'segment_labels': torch.from_numpy(segment_labels),
            'subject_id': subject_data.get('subject_id', f'subject_{idx}'),
        }


def collate_fn(batch):
    """DataLoader collate函数"""
    return {
        'segments': torch.stack([b['segments'] for b in batch]),
        'segment_lengths': torch.stack([b['segment_lengths'] for b in batch]),
        'segment_mask': torch.stack([b['segment_mask'] for b in batch]),
        'task_mask': torch.stack([b['task_mask'] for b in batch]),
        'task_conditions': torch.stack([b['task_conditions'] for b in batch]),
        'label': torch.stack([b['label'] for b in batch]),
        'segment_labels': torch.stack([b['segment_labels'] for b in batch]),
        'subject_ids': [b['subject_id'] for b in batch],
    }


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='训练层级-片段联合模型',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/hierarchical_segment.json',
        help='配置文件路径'
    )

    parser.add_argument(
        '--data', '-d',
        type=str,
        default='data/processed/processed_data.pkl',
        help='预处理数据路径 (.pkl 文件)'
    )

    parser.add_argument(
        '--segment-weight', '-sw',
        type=float,
        default=None,
        help='片段级损失权重（覆盖配置文件）'
    )

    parser.add_argument(
        '--hierarchical-weight', '-hw',
        type=float,
        default=None,
        help='层级（被试级）损失权重（覆盖配置文件）'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='输出目录（覆盖配置文件）'
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 检查数据文件
    if not os.path.exists(args.data):
        logger.error(f'数据文件不存在: {args.data}')
        logger.info('请先运行预处理脚本生成数据')
        return 1

    # 加载配置
    logger.info(f"加载配置: {args.config}")
    config = UnifiedConfig.from_json(args.config)

    # 覆盖输出目录
    if args.output_dir:
        config.experiment.output_dir = args.output_dir

    # 打印配置
    logger.info(f"模型配置: segment_d_model={config.model.segment_d_model}, "
                f"task_d_model={config.model.task_d_model}")
    logger.info(f"损失权重: segment={args.segment_weight or config.training.segment_loss_weight}, "
                f"hierarchical={args.hierarchical_weight or config.training.hierarchical_loss_weight}")

    # 加载数据
    logger.info(f'加载数据: {args.data}')
    with open(args.data, 'rb') as f:
        all_data = pickle.load(f)
    logger.info(f'加载了 {len(all_data)} 个被试')

    # 划分训练/验证集
    from sklearn.model_selection import train_test_split
    train_data, val_data = train_test_split(
        all_data,
        test_size=0.2,
        random_state=config.experiment.random_seed,
    )
    logger.info(f'训练集: {len(train_data)}，验证集: {len(val_data)}')

    # 创建序列配置
    seq_config = config.to_seq_config()

    # 创建数据集
    train_dataset = HierarchicalSegmentDataset(
        data=train_data,
        config=seq_config,
        fit_normalizer=True,
    )

    val_dataset = HierarchicalSegmentDataset(
        data=val_data,
        config=seq_config,
        fit_normalizer=False,
        normalizer_stats=train_dataset.stats,
    )

    # 导入并修改训练器以使用我们的 collate_fn
    from src.models.hierarchical_segment_trainer import HierarchicalSegmentTrainer
    from torch.utils.data import DataLoader

    class CustomHierarchicalSegmentTrainer(HierarchicalSegmentTrainer):
        """自定义训练器，使用我们的数据集"""

        def train(self, train_dataset, val_dataset, fold=0):
            """覆盖 train 方法以使用自定义 collate_fn"""
            # 创建数据加载器
            loader_kwargs = {
                'batch_size': self.config.training.batch_size,
                'collate_fn': collate_fn,  # 使用我们的 collate_fn
                'num_workers': self.config.device.num_workers,
                'pin_memory': self.config.device.pin_memory and torch.cuda.is_available(),
                'persistent_workers': False,  # 简单起见禁用
            }

            train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
            val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
            
            # 强制启用片段标签（我们的数据集始终有 segment_labels）
            self.has_segment_labels = True

            # 创建模型
            self.model = self._create_model()
            self.optimizer, self.scheduler = self._create_optimizer(self.model)

            logger.info(f'开始训练层级-片段联合模型 (Fold {fold + 1})')
            logger.info(f'训练集: {len(train_dataset)} 样本, 验证集: {len(val_dataset)} 样本')
            logger.info(f'片段级损失权重: {self.segment_loss_weight}, '
                       f'被试级损失权重: {self.hierarchical_loss_weight}')

            # 损失函数（仅用于标准损失，实际使用 model.compute_loss）
            if self.config.task.type == 'classification':
                import torch.nn as nn
                self.criterion = nn.CrossEntropyLoss()
            else:
                import torch.nn as nn
                self.criterion = nn.MSELoss()

            # 训练循环
            from tqdm import tqdm
            best_loss = float('inf')
            best_metrics = None
            patience_counter = 0

            for epoch in tqdm(range(self.config.training.epochs), desc='训练中'):
                # 训练（强制传入 has_segment_labels=True）
                train_losses = self.train_epoch(self.model, train_loader, self.optimizer, has_segment_labels=True)

                # 验证（强制传入 has_segment_labels=True）
                val_metrics = self.validate(self.model, val_loader, has_segment_labels=True)

                # 更新学习率
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']

                # 记录历史
                self.history['train_total_loss'].append(train_losses['total_loss'])
                self.history['val_total_loss'].append(val_metrics['total_loss'])
                self.history['learning_rate'].append(current_lr)

                # 保存最佳模型
                if val_metrics['total_loss'] < best_loss:
                    best_loss = val_metrics['total_loss']
                    best_metrics = val_metrics.copy()
                    best_metrics['epoch'] = epoch + 1
                    patience_counter = 0

                    # 保存模型
                    if self.config.output.save_best:
                        import os
                        os.makedirs(self.config.experiment.output_dir, exist_ok=True)
                        model_path = os.path.join(self.config.experiment.output_dir, f'model_best.pt')
                        save_state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
                        torch.save({
                            'model_state_dict': save_state,
                            'config': self.config,
                            'metrics': best_metrics,
                            'history': self.history,
                            'normalizer_stats': train_dataset.stats,
                        }, model_path)
                else:
                    patience_counter += 1

                # 早停检查
                if patience_counter >= self.config.training.patience:
                    logger.info(f'早停触发于 Epoch {epoch + 1}')
                    break

                # 日志
                if (epoch + 1) % self.config.output.summary_interval == 0:
                    logger.info(f"Epoch {epoch+1}: train_loss={train_losses['total_loss']:.4f}, "
                               f"val_loss={val_metrics['total_loss']:.4f}, "
                               f"val_subject_loss={val_metrics['subject_loss']:.4f}, "
                               f"val_segment_loss={val_metrics['segment_loss']:.4f}")

            logger.info(f'训练完成，最佳验证损失: {best_loss:.4f} @ Epoch {best_metrics["epoch"]}')
            return best_metrics

    # 创建训练器
    trainer = CustomHierarchicalSegmentTrainer(
        config=config,
        segment_loss_weight=args.segment_weight,
        hierarchical_loss_weight=args.hierarchical_weight,
    )

    # 训练
    metrics = trainer.train(train_dataset, val_dataset, fold=0)
    logger.info(f"最终指标: {metrics}")

    return 0


if __name__ == '__main__':
    exit(main())
