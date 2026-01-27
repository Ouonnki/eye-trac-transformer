# -*- coding: utf-8 -*-
"""
片段级训练器模块

提供片段级编码器的训练、验证和评估功能。
与被试级训练器的区别：
- 训练时：片段级损失
- 验证/测试时：聚合片段预测为被试级预测
"""

import os
import logging
import math
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, f1_score, precision_score, recall_score
)
from tqdm import tqdm

from src.models.segment_model import SegmentEncoder
from src.models.dl_dataset import SegmentGazeDataset, segment_collate_fn, SequenceConfig
from src.config import UnifiedConfig

logger = logging.getLogger(__name__)


def get_warmup_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    """获取带warmup的学习率调度器"""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


class SegmentTrainer:
    """
    片段级训练器

    关键特性：
    - 训练：片段级样本和损失
    - 验证/测试：聚合为被试级预测（投票/平均）
    - 保持被试级划分隔离，避免数据泄漏
    """

    def __init__(self, config: UnifiedConfig, seq_config: SequenceConfig):
        """
        初始化

        Args:
            config: 统一配置对象
            seq_config: 序列配置对象（数据属性）
        """
        self.config = config
        self.seq_config = seq_config
        self.device = torch.device(config.device.device)

        # 创建输出目录
        os.makedirs(config.experiment.output_dir, exist_ok=True)

        # 检查GPU
        self.n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if self.n_gpus > 1 and config.device.use_multi_gpu:
            logger.info(f'Using {self.n_gpus} GPUs')
        elif self.n_gpus == 1:
            logger.info(f'Using GPU: {torch.cuda.get_device_name(0)}')
        else:
            logger.info('Using CPU')

        # 混合精度
        self.use_amp = config.device.use_amp and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        # 模型和优化器
        self.model = None
        self.optimizer = None
        self.scheduler = None

        # 训练历史
        self._init_history()

    def _init_history(self) -> None:
        """初始化训练历史"""
        if self.config.task.type == 'classification':
            self.history = {
                'train_loss': [],
                'val_loss': [],
                'val_accuracy': [],
                'val_f1': [],
                'learning_rate': [],
            }
        else:
            self.history = {
                'train_loss': [],
                'val_loss': [],
                'val_r2': [],
                'val_mae': [],
                'learning_rate': [],
            }

    def _create_model(self) -> nn.Module:
        """创建片段编码器模型"""
        num_classes = self.config.task.num_classes if self.config.task.type == 'classification' else 1

        model = SegmentEncoder.from_config(
            config=self.config,
            seq_config=self.seq_config,
            num_classes=num_classes,
        )
        model = model.to(self.device)

        # 多GPU
        if self.n_gpus > 1 and self.config.device.use_multi_gpu:
            model = nn.DataParallel(model)

        return model

    def _create_optimizer(self, model: nn.Module) -> Tuple[AdamW, LambdaLR]:
        """创建优化器和调度器"""
        optimizer = AdamW(
            model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        scheduler = get_warmup_scheduler(
            optimizer,
            self.config.training.warmup_epochs,
            self.config.training.epochs,
        )
        return optimizer, scheduler

    def _aggregate_segment_predictions(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        subject_ids: List[str],
        method: str = 'vote',
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        将片段级预测聚合为被试级预测

        Args:
            predictions: 片段预测 (n_segments, num_classes) or (n_segments, 1)
            labels: 片段标签 (n_segments,)
            subject_ids: 被试ID列表

        Returns:
            subject_predictions: 被试级预测
            subject_labels: 被试级标签
        """
        # 按被试ID分组
        subject_preds = defaultdict(list)
        subject_labels_dict = {}

        for pred, label, sid in zip(predictions, labels, subject_ids):
            subject_preds[sid].append(pred)
            subject_labels_dict[sid] = label

        # 聚合
        subject_predictions = []
        subject_labels = []
        unique_subject_ids = list(subject_labels_dict.keys())

        for sid in unique_subject_ids:
            preds = np.array(subject_preds[sid])

            if self.config.task.type == 'classification':
                # 分类：多数投票
                if method == 'vote':
                    # 对每个片段取argmax，然后投票
                    class_votes = np.argmax(preds, axis=1)
                    # 计算每个类别的票数
                    vote_counts = np.bincount(class_votes, minlength=self.config.task.num_classes)
                    subject_pred = np.argmax(vote_counts)
                elif method == 'average':
                    # 平均概率，然后取argmax
                    avg_probs = np.mean(preds, axis=0)
                    subject_pred = np.argmax(avg_probs)
                else:
                    raise ValueError(f'Unknown aggregation method: {method}')
            else:
                # 回归：平均
                subject_pred = np.mean(preds)

            subject_predictions.append(subject_pred)
            subject_labels.append(subject_labels_dict[sid])

        return np.array(subject_predictions), np.array(subject_labels)

    def train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: AdamW,
        criterion: nn.Module,
    ) -> float:
        """训练一个epoch"""
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            features = batch['features'].to(self.device, non_blocking=True)
            lengths = batch['length'].to(self.device, non_blocking=True)
            labels = batch['label'].to(self.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(features, lengths)
                    loss = criterion(outputs, labels.squeeze(-1) if outputs.dim() > 1 else labels)

                self.scaler.scale(loss).backward()

                if self.config.training.grad_clip > 0:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.training.grad_clip)

                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                outputs = model(features, lengths)
                loss = criterion(outputs, labels.squeeze(-1) if outputs.dim() > 1 else labels)

                loss.backward()

                if self.config.training.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.training.grad_clip)

                optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    @torch.no_grad()
    def validate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module,
        aggregate: bool = True,
    ) -> Dict[str, float]:
        """
        验证模型

        Args:
            model: 模型
            val_loader: 验证数据加载器
            criterion: 损失函数
            aggregate: 是否聚合为被试级预测

        Returns:
            验证指标字典
        """
        model.eval()

        total_loss = 0.0
        num_batches = 0

        all_predictions = []
        all_labels = []
        all_subject_ids = []

        for batch in val_loader:
            features = batch['features'].to(self.device, non_blocking=True)
            lengths = batch['length'].to(self.device, non_blocking=True)
            labels = batch['label'].to(self.device, non_blocking=True)

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(features, lengths)
                    loss = criterion(outputs, labels.squeeze(-1) if outputs.dim() > 1 else labels)
            else:
                outputs = model(features, lengths)
                loss = criterion(outputs, labels.squeeze(-1) if outputs.dim() > 1 else labels)

            total_loss += loss.item()
            num_batches += 1

            # 收集预测
            all_predictions.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_subject_ids.extend(batch['subject_ids'])

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        # 是否聚合为被试级
        if aggregate and len(set(all_subject_ids)) > 1:
            predictions, labels = self._aggregate_segment_predictions(
                all_predictions, all_labels, all_subject_ids
            )
        else:
            predictions = all_predictions
            labels = all_labels

        # 计算指标
        if self.config.task.type == 'classification':
            preds = predictions if predictions.ndim == 1 else np.argmax(predictions, axis=1)
            metrics = {
                'loss': total_loss / num_batches,
                'accuracy': accuracy_score(labels, preds),
                'f1': f1_score(labels, preds, average='macro'),
                'precision': precision_score(labels, preds, average='macro', zero_division=0),
                'recall': recall_score(labels, preds, average='macro', zero_division=0),
            }
        else:
            metrics = {
                'loss': total_loss / num_batches,
                'r2': r2_score(labels, predictions),
                'mae': mean_absolute_error(labels, predictions),
                'rmse': np.sqrt(mean_squared_error(labels, predictions)),
            }

        return metrics

    def train(
        self,
        train_dataset: SegmentGazeDataset,
        val_dataset: SegmentGazeDataset,
        fold: int = 0,
    ) -> Dict[str, float]:
        """
        训练模型

        Args:
            train_dataset: 训练数据集
            val_dataset: 验证数据集
            fold: 当前折数

        Returns:
            最佳验证指标
        """
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=segment_collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=segment_collate_fn,
        )

        # 创建模型
        self.model = self._create_model()
        self.optimizer, self.scheduler = self._create_optimizer(self.model)

        # 损失函数
        if self.config.task.type == 'classification':
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()

        logger.info(f'开始训练片段级模型 (Fold {fold + 1})')
        logger.info(f'训练集: {len(train_dataset)} 片段, 验证集: {len(val_dataset)} 片段')

        best_val_loss = float('inf')
        best_val_metric = 0.0
        best_epoch = 0

        for epoch in range(self.config.training.epochs):
            # 训练
            train_loss = self.train_epoch(self.model, train_loader, self.optimizer, criterion)

            # 验证
            val_metrics = self.validate(self.model, val_loader, criterion)
            val_loss = val_metrics['loss']

            # 更新学习率
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)

            if self.config.task.type == 'classification':
                self.history['val_accuracy'].append(val_metrics['accuracy'])
                self.history['val_f1'].append(val_metrics['f1'])
                logger.info(
                    f'Epoch {epoch + 1}/{self.config.training.epochs} - '
                    f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                    f'Acc: {val_metrics["accuracy"]:.4f}, F1: {val_metrics["f1"]:.4f}'
                )
            else:
                self.history['val_r2'].append(val_metrics['r2'])
                self.history['val_mae'].append(val_metrics['mae'])
                logger.info(
                    f'Epoch {epoch + 1}/{self.config.training.epochs} - '
                    f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                    f'R2: {val_metrics["r2"]:.4f}, MAE: {val_metrics["mae"]:.4f}'
                )

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                if self.config.task.type == 'classification':
                    best_val_metric = val_metrics['accuracy']
                else:
                    best_val_metric = val_metrics['r2']
                self._save_checkpoint(fold, epoch)

            # 早停检查
            if epoch - best_epoch >= self.config.training.patience:
                logger.info(f'早停触发于 Epoch {epoch + 1}')
                break

        logger.info(f'训练完成. Best Val Loss: {best_val_loss:.4f} @ Epoch {best_epoch + 1}')

        return {'best_val_loss': best_val_loss, 'best_val_metric': best_val_metric}

    def _save_checkpoint(self, fold: int, epoch: int) -> None:
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
        }

        path = os.path.join(
            self.config.experiment.output_dir,
            f'segment_model_fold{fold}_epoch{epoch}.pt'
        )
        torch.save(checkpoint, path)

    def load_model(self, model_path: str) -> None:
        """加载模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = self._create_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f'模型已加载: {model_path}')

    def evaluate(
        self,
        test_dataset: SegmentGazeDataset,
        aggregate: bool = True,
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        评估模型

        Args:
            test_dataset: 测试数据集
            aggregate: 是否聚合为被试级预测

        Returns:
            包含指标和预测结果的字典
        """
        if self.model is None:
            raise ValueError('模型未加载，请先调用 train() 或 load_model()')

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=segment_collate_fn,
        )

        # 损失函数（仅用于计算loss）
        if self.config.task.type == 'classification':
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()

        metrics = self.validate(self.model, test_loader, criterion, aggregate=aggregate)

        # 额外收集预测结果
        all_predictions = []
        all_labels = []
        all_subject_ids = []

        self.model.eval()
        with torch.no_grad():
            for batch in test_loader:
                features = batch['features'].to(self.device, non_blocking=True)
                lengths = batch['length'].to(self.device, non_blocking=True)
                labels = batch['label']

                outputs = self.model(features, lengths)
                all_predictions.extend(outputs.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_subject_ids.extend(batch['subject_ids'])

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        if aggregate:
            predictions, labels = self._aggregate_segment_predictions(
                all_predictions, all_labels, all_subject_ids
            )
        else:
            predictions = all_predictions
            labels = all_labels

        metrics['predictions'] = predictions
        metrics['labels'] = labels
        metrics['subject_ids'] = list(set(all_subject_ids))

        return metrics
