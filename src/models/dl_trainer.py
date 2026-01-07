# -*- coding: utf-8 -*-
"""
深度学习训练器模块

提供用于层级Transformer网络的训练、验证和评估功能。
"""

import os
import logging
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tqdm import tqdm

from src.models.dl_models import HierarchicalTransformerNetwork
from src.models.dl_dataset import HierarchicalGazeDataset, collate_fn

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """训练配置"""
    # 模型参数
    input_dim: int = 7
    segment_d_model: int = 64
    segment_nhead: int = 4
    segment_num_layers: int = 4
    task_d_model: int = 128
    task_nhead: int = 4
    task_num_layers: int = 2
    attention_dim: int = 32
    dropout: float = 0.1
    max_seq_len: int = 100
    max_tasks: int = 30
    max_segments: int = 30

    # 训练参数
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    epochs: int = 100
    patience: int = 15
    grad_clip: float = 1.0

    # 设备与加速
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_multi_gpu: bool = True  # 多GPU并行
    use_amp: bool = True  # 混合精度训练
    num_workers: int = 8  # 数据加载进程数
    pin_memory: bool = True  # 锁页内存加速

    # 输出
    output_dir: str = 'outputs/dl_models'
    save_best: bool = True


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience: int = 15, min_delta: float = 0.0, mode: str = 'min'):
        """
        初始化

        Args:
            patience: 容忍的epoch数
            min_delta: 最小改进量
            mode: 'min'表示越小越好，'max'表示越大越好
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        检查是否应该早停

        Args:
            score: 当前分数

        Returns:
            是否应该早停
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


def get_warmup_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    """
    获取带warmup的学习率调度器

    Args:
        optimizer: 优化器
        warmup_epochs: 预热epoch数
        total_epochs: 总epoch数

    Returns:
        学习率调度器
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # 线性预热
            return (epoch + 1) / warmup_epochs
        else:
            # 余弦衰减
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


class DeepLearningTrainer:
    """
    深度学习训练器

    负责模型的训练、验证、保存和加载。
    支持多GPU并行和混合精度训练。
    """

    def __init__(self, config: TrainingConfig):
        """
        初始化

        Args:
            config: 训练配置
        """
        self.config = config
        self.device = torch.device(config.device)

        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)

        # 检查GPU数量
        self.n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if self.n_gpus > 1 and config.use_multi_gpu:
            logger.info(f'Using {self.n_gpus} GPUs for training')
        elif self.n_gpus == 1:
            logger.info(f'Using single GPU: {torch.cuda.get_device_name(0)}')
        else:
            logger.info('Using CPU for training')

        # 混合精度训练
        self.use_amp = config.use_amp and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        if self.use_amp:
            logger.info('Using Automatic Mixed Precision (AMP)')

        # 模型和优化器（在train时初始化）
        self.model = None
        self.optimizer = None
        self.scheduler = None

        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_r2': [],
            'val_mae': [],
            'learning_rate': [],
        }

    def _create_model(self) -> nn.Module:
        """创建模型（支持多GPU）"""
        model = HierarchicalTransformerNetwork(
            input_dim=self.config.input_dim,
            segment_d_model=self.config.segment_d_model,
            segment_nhead=self.config.segment_nhead,
            segment_num_layers=self.config.segment_num_layers,
            task_d_model=self.config.task_d_model,
            task_nhead=self.config.task_nhead,
            task_num_layers=self.config.task_num_layers,
            attention_dim=self.config.attention_dim,
            dropout=self.config.dropout,
            max_seq_len=self.config.max_seq_len,
            max_tasks=self.config.max_tasks,
            max_segments=self.config.max_segments,
        )
        model = model.to(self.device)

        # 多GPU并行
        if self.n_gpus > 1 and self.config.use_multi_gpu:
            model = nn.DataParallel(model)
            logger.info(f'Model wrapped with DataParallel on {self.n_gpus} GPUs')

        return model

    def _create_optimizer(self, model: nn.Module) -> Tuple[AdamW, LambdaLR]:
        """创建优化器和调度器"""
        optimizer = AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = get_warmup_scheduler(
            optimizer,
            self.config.warmup_epochs,
            self.config.epochs,
        )
        return optimizer, scheduler

    def train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: AdamW,
        criterion: nn.Module,
    ) -> float:
        """
        训练一个epoch（支持混合精度）

        Args:
            model: 模型
            train_loader: 训练数据加载器
            optimizer: 优化器
            criterion: 损失函数

        Returns:
            平均训练损失
        """
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            # 移动数据到设备
            segments = batch['segments'].to(self.device, non_blocking=True)
            segment_mask = batch['segment_mask'].to(self.device, non_blocking=True)
            task_mask = batch['task_mask'].to(self.device, non_blocking=True)
            segment_lengths = batch['segment_lengths'].to(self.device, non_blocking=True)
            labels = batch['label'].to(self.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零

            # 混合精度前向传播
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(
                        segments=segments,
                        segment_mask=segment_mask,
                        task_mask=task_mask,
                        segment_lengths=segment_lengths,
                    )
                    loss = criterion(outputs['prediction'], labels)

                # 混合精度反向传播
                self.scaler.scale(loss).backward()

                # 梯度裁剪
                if self.config.grad_clip > 0:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)

                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                # 标准前向传播
                outputs = model(
                    segments=segments,
                    segment_mask=segment_mask,
                    task_mask=task_mask,
                    segment_lengths=segment_lengths,
                )
                loss = criterion(outputs['prediction'], labels)

                # 反向传播
                loss.backward()

                # 梯度裁剪
                if self.config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)

                optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def validate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module,
    ) -> Dict[str, float]:
        """
        验证模型（支持混合精度）

        Args:
            model: 模型
            val_loader: 验证数据加载器
            criterion: 损失函数

        Returns:
            验证指标字典
        """
        model.eval()
        total_loss = 0.0
        num_batches = 0

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                # 移动数据到设备
                segments = batch['segments'].to(self.device, non_blocking=True)
                segment_mask = batch['segment_mask'].to(self.device, non_blocking=True)
                task_mask = batch['task_mask'].to(self.device, non_blocking=True)
                segment_lengths = batch['segment_lengths'].to(self.device, non_blocking=True)
                labels = batch['label'].to(self.device, non_blocking=True)

                # 混合精度前向传播
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(
                            segments=segments,
                            segment_mask=segment_mask,
                            task_mask=task_mask,
                            segment_lengths=segment_lengths,
                        )
                        loss = criterion(outputs['prediction'], labels)
                else:
                    outputs = model(
                        segments=segments,
                        segment_mask=segment_mask,
                        task_mask=task_mask,
                        segment_lengths=segment_lengths,
                    )
                    loss = criterion(outputs['prediction'], labels)

                total_loss += loss.item()
                num_batches += 1

                # 收集预测和标签
                all_predictions.extend(outputs['prediction'].cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 计算指标
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        metrics = {
            'loss': total_loss / num_batches,
            'r2': r2_score(all_labels, all_predictions),
            'mae': mean_absolute_error(all_labels, all_predictions),
            'rmse': np.sqrt(mean_squared_error(all_labels, all_predictions)),
        }

        return metrics

    def train(
        self,
        train_dataset: HierarchicalGazeDataset,
        val_dataset: HierarchicalGazeDataset,
        fold: int = 0,
    ) -> Dict[str, float]:
        """
        训练模型

        Args:
            train_dataset: 训练数据集
            val_dataset: 验证数据集
            fold: 当前折数（用于保存模型）

        Returns:
            最佳验证指标
        """
        # 创建模型
        self.model = self._create_model()
        self.optimizer, self.scheduler = self._create_optimizer(self.model)

        # 创建数据加载器（支持多进程和锁页内存）
        loader_kwargs = {
            'batch_size': self.config.batch_size,
            'collate_fn': collate_fn,
            'num_workers': self.config.num_workers,
            'pin_memory': self.config.pin_memory and torch.cuda.is_available(),
            'persistent_workers': self.config.num_workers > 0,
        }

        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            **loader_kwargs,
        )
        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            **loader_kwargs,
        )

        logger.info(f'DataLoader: batch_size={self.config.batch_size}, '
                    f'num_workers={self.config.num_workers}, '
                    f'pin_memory={loader_kwargs["pin_memory"]}')

        # 损失函数
        criterion = nn.MSELoss()

        # 早停
        early_stopping = EarlyStopping(patience=self.config.patience, mode='min')

        # 最佳模型
        best_metrics = None
        best_model_state = None

        # 重置历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_r2': [],
            'val_mae': [],
            'learning_rate': [],
        }

        # 训练循环
        pbar = tqdm(range(self.config.epochs), desc=f'Fold {fold+1}')
        for epoch in pbar:
            # 训练
            train_loss = self.train_epoch(
                self.model, train_loader, self.optimizer, criterion
            )

            # 验证
            val_metrics = self.validate(self.model, val_loader, criterion)

            # 更新学习率
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_r2'].append(val_metrics['r2'])
            self.history['val_mae'].append(val_metrics['mae'])
            self.history['learning_rate'].append(current_lr)

            # 更新进度条
            pbar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'val_loss': f'{val_metrics["loss"]:.4f}',
                'val_r2': f'{val_metrics["r2"]:.4f}',
                'lr': f'{current_lr:.6f}',
            })

            # 保存最佳模型（处理DataParallel包装）
            if best_metrics is None or val_metrics['loss'] < best_metrics['loss']:
                best_metrics = val_metrics.copy()
                # 获取原始模型状态（去除DataParallel包装）
                if isinstance(self.model, nn.DataParallel):
                    best_model_state = self.model.module.state_dict().copy()
                else:
                    best_model_state = self.model.state_dict().copy()

            # 早停检查
            if early_stopping(val_metrics['loss']):
                logger.info(f'Early stopping at epoch {epoch+1}')
                break

        # 恢复最佳模型（处理DataParallel）
        if best_model_state is not None:
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(best_model_state)
            else:
                self.model.load_state_dict(best_model_state)

        # 保存模型（保存不带DataParallel的状态）
        if self.config.save_best:
            model_path = os.path.join(self.config.output_dir, f'model_fold{fold}.pt')
            save_state = best_model_state if best_model_state is not None else (
                self.model.module.state_dict() if isinstance(self.model, nn.DataParallel)
                else self.model.state_dict()
            )
            torch.save({
                'model_state_dict': save_state,
                'config': self.config,
                'metrics': best_metrics,
            }, model_path)
            logger.info(f'Model saved to {model_path}')

        return best_metrics

    def predict(
        self,
        dataset: HierarchicalGazeDataset,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        预测

        Args:
            dataset: 数据集

        Returns:
            predictions: 预测值
            labels: 真实值
            attention_weights: 注意力权重
        """
        if self.model is None:
            raise ValueError('Model not trained or loaded')

        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )

        self.model.eval()

        all_predictions = []
        all_labels = []
        all_segment_attentions = []
        all_task_attentions = []

        with torch.no_grad():
            for batch in loader:
                segments = batch['segments'].to(self.device)
                segment_mask = batch['segment_mask'].to(self.device)
                task_mask = batch['task_mask'].to(self.device)
                segment_lengths = batch['segment_lengths'].to(self.device)
                labels = batch['label']

                outputs = self.model(
                    segments=segments,
                    segment_mask=segment_mask,
                    task_mask=task_mask,
                    segment_lengths=segment_lengths,
                )

                all_predictions.extend(outputs['prediction'].cpu().numpy())
                all_labels.extend(labels.numpy())
                all_segment_attentions.append(outputs['segment_attention'].cpu().numpy())
                all_task_attentions.append(outputs['task_attention'].cpu().numpy())

        return (
            np.array(all_predictions),
            np.array(all_labels),
            {
                'segment_attention': np.concatenate(all_segment_attentions, axis=0),
                'task_attention': np.concatenate(all_task_attentions, axis=0),
            }
        )

    def load_model(self, model_path: str) -> None:
        """
        加载模型

        Args:
            model_path: 模型路径
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = self._create_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f'Model loaded from {model_path}')
