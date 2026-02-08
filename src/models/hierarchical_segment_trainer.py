# -*- coding: utf-8 -*-
"""
层级-片段联合模型训练器

为新模型 HierarchicalSegmentTransformerNetwork 提供训练支持。
关键特性：
- 使用模型的 compute_loss 方法计算联合损失
- 同时评估被试级和片段级预测性能
- 支持加权损失训练
"""

import os
import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, f1_score
from tqdm import tqdm
from colorama import init, Fore, Style

# 初始化 colorama
init(autoreset=True)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from src.models.hierarchical_segment_model import HierarchicalSegmentTransformerNetwork
from src.models.dl_dataset import (
    HierarchicalGazeDataset,
    HierarchicalGazeDatasetWithSegmentLabels,
    hierarchical_segment_collate_fn,
    collate_fn,
    SequenceConfig,
)
from src.config import UnifiedConfig

logger = logging.getLogger(__name__)


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience: int = 15, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score: float, epoch: int = 0) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


def get_warmup_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    """获取带warmup的学习率调度器"""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


class HierarchicalSegmentTrainer:
    """
    层级-片段联合模型训练器

    特点：
    - 使用模型的 compute_loss 计算加权联合损失
    - 同时评估被试级和片段级预测
    - 支持多GPU和混合精度
    """

    def __init__(
        self,
        config: UnifiedConfig,
        seq_config: SequenceConfig = None,
        segment_loss_weight: float = None,
        hierarchical_loss_weight: float = None,
    ):
        """
        初始化

        Args:
            config: 统一配置对象
            seq_config: 序列配置对象
            segment_loss_weight: 片段级损失权重（默认从 config.training.segment_loss_weight 读取）
            hierarchical_loss_weight: 被试级损失权重（默认从 config.training.hierarchical_loss_weight 读取）
        """
        if seq_config is None:
            seq_config = config.to_seq_config()

        self.config = config
        self.seq_config = seq_config
        self.device = torch.device(config.device.device)
        
        # 从配置读取损失权重，如果没有则使用默认值
        self.segment_loss_weight = segment_loss_weight if segment_loss_weight is not None else getattr(
            config.training, 'segment_loss_weight', 0.5
        )
        self.hierarchical_loss_weight = hierarchical_loss_weight if hierarchical_loss_weight is not None else getattr(
            config.training, 'hierarchical_loss_weight', 0.5
        )

        # 创建输出目录
        os.makedirs(config.experiment.output_dir, exist_ok=True)

        # 检查GPU
        self.n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if self.n_gpus > 1 and config.device.use_multi_gpu:
            logger.info(f'使用 {self.n_gpus} GPUs 训练')
        elif self.n_gpus == 1:
            logger.info(f'使用 GPU: {torch.cuda.get_device_name(0)}')
        else:
            logger.info('使用 CPU 训练')

        # 混合精度
        self.use_amp = config.device.use_amp and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        if self.use_amp:
            logger.info('使用自动混合精度 (AMP)')

        # 模型和优化器
        self.model = None
        self.optimizer = None
        self.scheduler = None

        # 训练历史
        self._init_history()

    def _init_history(self) -> None:
        """
        初始化训练历史

        对齐任务级模型的键名，同时保持片段级特有键名
        """
        # 基础历史（对齐任务级模型）
        base_history = {
            'train_loss': [],  # 对齐任务级模型，作为 train_total_loss 的别名
            'val_loss': [],    # 对齐任务级模型，作为 val_total_loss 的别名
            'learning_rate': [],
        }

        # 片段级模型特有：详细的损失分解
        segment_history = {
            'train_total_loss': [],
            'train_subject_loss': [],
            'train_segment_loss': [],
            'val_total_loss': [],
            'val_subject_loss': [],
            'val_segment_loss': [],
        }

        if self.config.task.type == 'classification':
            # 分类任务：对齐 val_accuracy, val_f1
            self.history = {
                **base_history,
                **segment_history,
                'val_accuracy': [],  # 对齐任务级模型（被试级）
                'val_f1': [],        # 对齐任务级模型（被试级）
                # 片段级特有
                'val_subject_accuracy': [],
                'val_subject_f1': [],
                'val_segment_accuracy': [],
                'val_segment_f1': [],
            }
        else:
            # 回归任务：对齐 val_r2, val_mae
            self.history = {
                **base_history,
                **segment_history,
                'val_r2': [],  # 对齐任务级模型（被试级）
                'val_mae': [], # 对齐任务级模型（被试级）
                # 片段级特有
                'val_subject_r2': [],
                'val_subject_mae': [],
                'val_segment_r2': [],
                'val_segment_mae': [],
            }

    def _create_model(self) -> nn.Module:
        """创建模型"""
        num_classes = self.config.task.num_classes if self.config.task.type == 'classification' else 1

        model = HierarchicalSegmentTransformerNetwork.from_config(
            config=self.config,
            seq_config=self.seq_config,
            num_classes=num_classes,
            segment_loss_weight=self.segment_loss_weight,
            hierarchical_loss_weight=self.hierarchical_loss_weight,
        )
        model = model.to(self.device)

        if self.config.device.use_gradient_checkpointing:
            logger.info('使用梯度检查点节省显存')

        # 多GPU
        if self.n_gpus > 1 and self.config.device.use_multi_gpu:
            model = nn.DataParallel(model)
            logger.info(f'使用 DataParallel 在 {self.n_gpus} GPUs 上训练')

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
        segment_predictions: np.ndarray,
        segment_mask: np.ndarray,
        method: str = 'mean',
    ) -> np.ndarray:
        """
        将片段级预测聚合为被试级预测

        Args:
            segment_predictions: (batch, max_tasks, max_segments) 片段预测
            segment_mask: (batch, max_tasks, max_segments) 有效片段掩码
            method: 聚合方法 ('mean', 'median', 'attention')

        Returns:
            subject_predictions: (batch,) 被试级预测
        """
        batch_size = segment_predictions.shape[0]

        if self.config.task.type == 'classification':
            # 分类：对每个片段预测取 argmax 后投票
            segment_classes = np.argmax(segment_predictions, axis=-1)  # (batch, max_tasks, max_segments)
            subject_preds = []

            for i in range(batch_size):
                valid_mask = segment_mask[i].flatten()
                valid_preds = segment_classes[i].flatten()[valid_mask]

                if len(valid_preds) > 0:
                    # 多数投票
                    vote_counts = np.bincount(valid_preds, minlength=self.config.task.num_classes)
                    subject_pred = np.argmax(vote_counts)
                else:
                    subject_pred = 0
                subject_preds.append(subject_pred)

            return np.array(subject_preds)
        else:
            # 回归：取有效片段的平均值
            subject_preds = []
            for i in range(batch_size):
                valid_mask = segment_mask[i].flatten()
                valid_preds = segment_predictions[i].flatten()[valid_mask]

                if len(valid_preds) > 0:
                    if method == 'mean':
                        subject_pred = np.mean(valid_preds)
                    elif method == 'median':
                        subject_pred = np.median(valid_preds)
                    else:
                        subject_pred = np.mean(valid_preds)
                else:
                    subject_pred = 0.0
                subject_preds.append(subject_pred)

            return np.array(subject_preds)

    def train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: AdamW,
        has_segment_labels: bool = False,
    ) -> Dict[str, float]:
        """
        训练一个epoch

        Args:
            model: 模型
            train_loader: 训练数据加载器
            optimizer: 优化器
            has_segment_labels: 是否有片段级标签

        Returns:
            损失字典
        """
        model.train()
        total_losses = {
            'total_loss': 0.0,
            'subject_loss': 0.0,
            'segment_loss': 0.0,
        }
        num_batches = 0

        for batch in train_loader:
            # 移动数据到设备
            segments = batch['segments'].to(self.device, non_blocking=True)
            segment_mask = batch['segment_mask'].to(self.device, non_blocking=True)
            task_mask = batch['task_mask'].to(self.device, non_blocking=True)
            segment_lengths = batch['segment_lengths'].to(self.device, non_blocking=True)
            subject_labels = batch['label'].to(self.device, non_blocking=True)
            task_conditions = batch.get('task_conditions', None)
            if task_conditions is not None:
                task_conditions = task_conditions.to(self.device, non_blocking=True)

            # 片段级标签（可选）
            segment_labels = batch.get('segment_labels', None)
            if segment_labels is not None:
                segment_labels = segment_labels.to(self.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # 混合精度前向传播
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(
                        segments=segments,
                        segment_mask=segment_mask,
                        task_mask=task_mask,
                        segment_lengths=segment_lengths,
                        task_conditions=task_conditions,
                    )

                    # 使用模型的 compute_loss 方法（处理 DataParallel）
                    model_module = model.module if isinstance(model, nn.DataParallel) else model
                    losses = model_module.compute_loss(
                        outputs,
                        subject_targets=subject_labels,
                        segment_targets=segment_labels if has_segment_labels else None,
                        segment_mask=segment_mask,
                    )

                # 反向传播
                self.scaler.scale(losses['total_loss']).backward()

                if self.config.training.grad_clip > 0:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.training.grad_clip)

                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                outputs = model(
                    segments=segments,
                    segment_mask=segment_mask,
                    task_mask=task_mask,
                    segment_lengths=segment_lengths,
                    task_conditions=task_conditions,
                )

                model_module = model.module if isinstance(model, nn.DataParallel) else model
                losses = model_module.compute_loss(
                    outputs,
                    subject_targets=subject_labels,
                    segment_targets=segment_labels if has_segment_labels else None,
                    segment_mask=segment_mask,
                )

                losses['total_loss'].backward()

                if self.config.training.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.training.grad_clip)

                optimizer.step()

            total_losses['total_loss'] += losses['total_loss'].item()
            total_losses['subject_loss'] += losses['subject_loss'].item()
            total_losses['segment_loss'] += losses['segment_loss'].item()
            num_batches += 1

        return {k: v / num_batches for k, v in total_losses.items()}

    @torch.no_grad()
    def validate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        has_segment_labels: bool = False,
    ) -> Dict[str, float]:
        """
        验证模型

        Args:
            model: 模型
            val_loader: 验证数据加载器
            has_segment_labels: 是否有片段级标签

        Returns:
            验证指标字典
        """
        model.eval()

        total_losses = {
            'total_loss': 0.0,
            'subject_loss': 0.0,
            'segment_loss': 0.0,
        }
        num_batches = 0

        all_subject_predictions = []
        all_subject_labels = []
        all_segment_predictions = []
        all_segment_labels = []
        all_segment_masks = []

        for batch in val_loader:
            segments = batch['segments'].to(self.device, non_blocking=True)
            segment_mask = batch['segment_mask'].to(self.device, non_blocking=True)
            task_mask = batch['task_mask'].to(self.device, non_blocking=True)
            segment_lengths = batch['segment_lengths'].to(self.device, non_blocking=True)
            subject_labels = batch['label'].to(self.device, non_blocking=True)
            task_conditions = batch.get('task_conditions', None)
            if task_conditions is not None:
                task_conditions = task_conditions.to(self.device, non_blocking=True)

            segment_labels = batch.get('segment_labels', None)
            if segment_labels is not None:
                segment_labels = segment_labels.to(self.device, non_blocking=True)

            # 混合精度前向传播
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(
                        segments=segments,
                        segment_mask=segment_mask,
                        task_mask=task_mask,
                        segment_lengths=segment_lengths,
                        task_conditions=task_conditions,
                    )

                    model_module = model.module if isinstance(model, nn.DataParallel) else model
                    losses = model_module.compute_loss(
                        outputs,
                        subject_targets=subject_labels,
                        segment_targets=segment_labels if has_segment_labels else None,
                        segment_mask=segment_mask,
                    )
            else:
                outputs = model(
                    segments=segments,
                    segment_mask=segment_mask,
                    task_mask=task_mask,
                    segment_lengths=segment_lengths,
                    task_conditions=task_conditions,
                )

                model_module = model.module if isinstance(model, nn.DataParallel) else model
                losses = model_module.compute_loss(
                    outputs,
                    subject_targets=subject_labels,
                    segment_targets=segment_labels if has_segment_labels else None,
                    segment_mask=segment_mask,
                )

            total_losses['total_loss'] += losses['total_loss'].item()
            total_losses['subject_loss'] += losses['subject_loss'].item()
            total_losses['segment_loss'] += losses['segment_loss'].item()
            num_batches += 1

            # 收集预测
            all_subject_predictions.extend(outputs['subject_prediction'].cpu().numpy())
            all_subject_labels.extend(subject_labels.cpu().numpy())

            # 收集片段预测（只收集有效的）
            segment_preds = outputs['segment_predictions'].cpu().numpy()
            all_segment_predictions.append(segment_preds)
            if segment_labels is not None:
                all_segment_labels.append(segment_labels.cpu().numpy())
            all_segment_masks.append(segment_mask.cpu().numpy())

        # 计算平均损失
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}

        # 被试级指标
        subject_preds = np.array(all_subject_predictions)
        subject_labels = np.array(all_subject_labels)

        # 片段级指标（聚合所有batch）
        all_segment_predictions = np.concatenate(all_segment_predictions, axis=0)
        all_segment_masks = np.concatenate(all_segment_masks, axis=0)

        # 将片段预测聚合为被试级预测进行对比
        aggregated_from_segments = self._aggregate_segment_predictions(
            all_segment_predictions, all_segment_masks
        )

        if self.config.task.type == 'classification':
            # 被试级指标
            subject_preds_class = np.argmax(subject_preds, axis=1) if subject_preds.ndim > 1 else subject_preds
            metrics = {
                **avg_losses,
                'subject_accuracy': accuracy_score(subject_labels, subject_preds_class),
                'subject_f1': f1_score(subject_labels, subject_preds_class, average='macro'),
                'segment_accuracy': accuracy_score(subject_labels, aggregated_from_segments),
                'segment_f1': f1_score(subject_labels, aggregated_from_segments, average='macro'),
            }
        else:
            metrics = {
                **avg_losses,
                'subject_r2': r2_score(subject_labels, subject_preds),
                'subject_mae': mean_absolute_error(subject_labels, subject_preds),
                'segment_r2': r2_score(subject_labels, aggregated_from_segments),
                'segment_mae': mean_absolute_error(subject_labels, aggregated_from_segments),
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
            fold: 当前折数

        Returns:
            最佳验证指标
        """
        # 检查是否有片段级标签
        has_segment_labels = isinstance(train_dataset, HierarchicalGazeDatasetWithSegmentLabels) and train_dataset.has_segment_labels

        # 选择正确的 collate 函数
        if has_segment_labels:
            collate_func = hierarchical_segment_collate_fn
        else:
            collate_func = collate_fn

        # 创建数据加载器
        loader_kwargs = {
            'batch_size': self.config.training.batch_size,
            'collate_fn': collate_func,
            'num_workers': self.config.device.num_workers,
            'pin_memory': self.config.device.pin_memory and torch.cuda.is_available(),
            'persistent_workers': self.config.device.num_workers > 0,
        }

        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

        # 创建模型
        self.model = self._create_model()
        self.optimizer, self.scheduler = self._create_optimizer(self.model)

        logger.info(f'开始训练层级-片段联合模型 (Fold {fold + 1})')
        logger.info(f'训练集: {len(train_dataset)} 样本, 验证集: {len(val_dataset)} 样本')
        logger.info(f'片段级损失权重: {self.segment_loss_weight}, 被试级损失权重: {self.hierarchical_loss_weight}')
        if has_segment_labels:
            logger.info('使用片段级标签进行监督')

        # 早停
        early_stopping = EarlyStopping(patience=self.config.training.patience, mode='min')

        # 最佳模型
        best_metrics = None
        best_model_state = None

        # 重置历史
        self._init_history()

        # 训练循环
        pbar = tqdm(range(self.config.training.epochs), desc='训练中', ncols=140)

        for epoch in pbar:
            # 训练
            train_losses = self.train_epoch(self.model, train_loader, self.optimizer, has_segment_labels)

            # 验证
            val_metrics = self.validate(self.model, val_loader, has_segment_labels)

            # 更新学习率
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            # 记录历史（对齐任务级模型的键名）
            # 基础键名（对齐）
            self.history['train_loss'].append(train_losses['total_loss'])
            self.history['val_loss'].append(val_metrics['total_loss'])
            self.history['learning_rate'].append(current_lr)

            # 详细损失分解（片段级模型特有）
            self.history['train_total_loss'].append(train_losses['total_loss'])
            self.history['train_subject_loss'].append(train_losses['subject_loss'])
            self.history['train_segment_loss'].append(train_losses['segment_loss'])
            self.history['val_total_loss'].append(val_metrics['total_loss'])
            self.history['val_subject_loss'].append(val_metrics['subject_loss'])
            self.history['val_segment_loss'].append(val_metrics['segment_loss'])

            if self.config.task.type == 'classification':
                # 被试级指标（对齐任务级模型）
                self.history['val_accuracy'].append(val_metrics['subject_accuracy'])
                self.history['val_f1'].append(val_metrics['subject_f1'])
                # 片段级模型特有
                self.history['val_subject_accuracy'].append(val_metrics['subject_accuracy'])
                self.history['val_subject_f1'].append(val_metrics['subject_f1'])
                self.history['val_segment_accuracy'].append(val_metrics['segment_accuracy'])
                self.history['val_segment_f1'].append(val_metrics['segment_f1'])

                postfix_dict = {
                    'loss': f"{train_losses['total_loss']:.3f}",
                    'sub_acc': f"{val_metrics['subject_accuracy']:.3f}",
                    'seg_acc': f"{val_metrics['segment_accuracy']:.3f}",
                }
            else:
                # 被试级指标（对齐任务级模型）
                self.history['val_r2'].append(val_metrics['subject_r2'])
                self.history['val_mae'].append(val_metrics['subject_mae'])
                # 片段级模型特有
                self.history['val_subject_r2'].append(val_metrics['subject_r2'])
                self.history['val_subject_mae'].append(val_metrics['subject_mae'])
                self.history['val_segment_r2'].append(val_metrics['segment_r2'])
                self.history['val_segment_mae'].append(val_metrics['segment_mae'])

                postfix_dict = {
                    'loss': f"{train_losses['total_loss']:.3f}",
                    'sub_r2': f"{val_metrics['subject_r2']:.3f}",
                    'seg_r2': f"{val_metrics['segment_r2']:.3f}",
                }

            pbar.set_postfix(postfix_dict)

            # 保存最佳模型
            if best_metrics is None or val_metrics['total_loss'] < best_metrics['total_loss']:
                best_metrics = val_metrics.copy()
                best_metrics['epoch'] = epoch + 1

                if isinstance(self.model, nn.DataParallel):
                    best_model_state = self.model.module.state_dict().copy()
                else:
                    best_model_state = self.model.state_dict().copy()

            # 早停检查
            if early_stopping(val_metrics['total_loss'], epoch):
                logger.info(f'早停触发于 Epoch {epoch + 1}')
                break

        # 恢复最佳模型
        if best_model_state is not None:
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(best_model_state)
            else:
                self.model.load_state_dict(best_model_state)

        # 保存训练曲线图（对齐任务级模型）
        if self.config.output.save_figures:
            self.plot_training_curves(fold=fold)

        # 保存模型
        if self.config.output.save_best:
            model_path = os.path.join(self.config.experiment.output_dir, f'model_fold{fold}.pt')
            save_state = best_model_state if best_model_state is not None else (
                self.model.module.state_dict() if isinstance(self.model, nn.DataParallel)
                else self.model.state_dict()
            )
            torch.save({
                'model_state_dict': save_state,
                'config': self.config,
                'metrics': best_metrics,
                'history': self.history,
            }, model_path)
            logger.info(f'模型已保存: {model_path}')

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
            predictions: 被试级预测（对齐任务级模型的返回格式）
            labels: 真实值
            extras: 额外信息字典，包含：
                - segment_attention: 片段注意力权重
                - task_attention: 任务注意力权重
                - segment_predictions: 片段级预测
                - aggregated_predictions: 从片段聚合的预测
        """
        if self.model is None:
            raise ValueError('模型未训练或加载')

        # 选择正确的 collate 函数
        has_segment_labels = isinstance(dataset, HierarchicalGazeDatasetWithSegmentLabels)
        if has_segment_labels:
            collate_func = hierarchical_segment_collate_fn
        else:
            collate_func = collate_fn

        loader = DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            collate_fn=collate_func,
            num_workers=0,
        )

        self.model.eval()

        all_subject_predictions = []
        all_subject_labels = []
        all_segment_predictions = []
        all_segment_masks = []
        all_segment_attentions = []
        all_task_attentions = []

        with torch.no_grad():
            for batch in loader:
                segments = batch['segments'].to(self.device)
                segment_mask = batch['segment_mask'].to(self.device)
                task_mask = batch['task_mask'].to(self.device)
                segment_lengths = batch['segment_lengths'].to(self.device)
                labels = batch['label']
                task_conditions = batch.get('task_conditions', None)
                if task_conditions is not None:
                    task_conditions = task_conditions.to(self.device)

                outputs = self.model(
                    segments=segments,
                    segment_mask=segment_mask,
                    task_mask=task_mask,
                    segment_lengths=segment_lengths,
                    task_conditions=task_conditions,
                )

                # 使用 'prediction' 键（与任务级模型对齐）
                predictions = outputs.get('prediction', outputs.get('subject_prediction'))
                all_subject_predictions.extend(predictions.cpu().numpy())
                all_subject_labels.extend(labels.numpy())
                all_segment_predictions.append(outputs['segment_predictions'].cpu().numpy())
                all_segment_masks.append(segment_mask.cpu().numpy())
                all_segment_attentions.append(outputs['segment_attention'].cpu().numpy())
                all_task_attentions.append(outputs['task_attention'].cpu().numpy())

        subject_predictions = np.array(all_subject_predictions)
        subject_labels = np.array(all_subject_labels)
        segment_predictions = np.concatenate(all_segment_predictions, axis=0)
        segment_masks = np.concatenate(all_segment_masks, axis=0)

        # 从片段预测聚合
        aggregated_predictions = self._aggregate_segment_predictions(
            segment_predictions, segment_masks
        )

        # 对齐任务级模型的返回格式: (predictions, labels, attention_weights)
        extras = {
            'segment_attention': np.concatenate(all_segment_attentions, axis=0),
            'task_attention': np.concatenate(all_task_attentions, axis=0),
            'segment_predictions': segment_predictions,
            'aggregated_predictions': aggregated_predictions,
        }

        return subject_predictions, subject_labels, extras

    def plot_training_curves(self, save_path: Optional[str] = None, fold: int = 0) -> Optional[str]:
        """
        绘制训练曲线（对齐任务级模型的格式）

        与 DeepLearningTrainer 的 plot_training_curves 方法兼容，
        同时显示片段级模型的额外指标。

        Args:
            save_path: 保存路径，None 则使用默认路径
            fold: 当前折数

        Returns:
            保存的图片路径，如果失败则返回 None
        """
        if not HAS_MATPLOTLIB:
            logger.warning('matplotlib 未安装，无法生成训练曲线图')
            return None

        if not self.history.get('train_loss'):
            logger.warning('训练历史为空，无法生成图表')
            return None

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        # 创建 2×2 子图
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        task_type_label = '分类' if self.config.task.type == 'classification' else '回归'
        fig.suptitle(f'训练曲线 (Fold {fold + 1}) - {task_type_label}任务 (层级-片段联合模型)', fontsize=14, fontweight='bold')

        epochs = range(1, len(self.history['train_loss']) + 1)

        # 子图1: Loss 曲线
        ax1 = axes[0, 0]
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        loss_label = 'Loss (Joint)'  # 联合损失
        ax1.set_ylabel(loss_label)
        ax1.set_title('Loss 曲线')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 标记最佳点
        best_epoch = np.argmin(self.history['val_loss']) + 1
        best_val_loss = min(self.history['val_loss'])
        ax1.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best @ {best_epoch}')
        ax1.scatter([best_epoch], [best_val_loss], color='g', s=100, zorder=5)

        if self.config.task.type == 'classification':
            # 子图2: Accuracy 曲线
            ax2 = axes[0, 1]
            ax2.plot(epochs, self.history['val_accuracy'], 'g-', label='Subject Acc (被试级)', linewidth=2)
            ax2.plot(epochs, self.history['val_segment_accuracy'], 'b--', label='Segment Acc (片段级)', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Accuracy 曲线')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # 标记最佳 Accuracy
            best_acc_epoch = np.argmax(self.history['val_accuracy']) + 1
            best_acc = max(self.history['val_accuracy'])
            ax2.axvline(x=best_acc_epoch, color='orange', linestyle='--', alpha=0.7)
            ax2.scatter([best_acc_epoch], [best_acc], color='orange', s=100, zorder=5)
            ax2.annotate(f'Best: {best_acc:.4f}', xy=(best_acc_epoch, best_acc),
                         xytext=(10, -10), textcoords='offset points', fontsize=9)

            # 子图3: F1 曲线
            ax3 = axes[1, 0]
            ax3.plot(epochs, self.history['val_f1'], 'm-', label='Subject F1 (被试级)', linewidth=2)
            ax3.plot(epochs, self.history['val_segment_f1'], 'c--', label='Segment F1 (片段级)', linewidth=2)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('F1 Score')
            ax3.set_title('F1 曲线')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # 标记最佳 F1
            best_f1_epoch = np.argmax(self.history['val_f1']) + 1
            best_f1 = max(self.history['val_f1'])
            ax3.axvline(x=best_f1_epoch, color='purple', linestyle='--', alpha=0.7)
            ax3.scatter([best_f1_epoch], [best_f1], color='purple', s=100, zorder=5)
            ax3.annotate(f'Best: {best_f1:.4f}', xy=(best_f1_epoch, best_f1),
                         xytext=(10, 10), textcoords='offset points', fontsize=9)
        else:
            # 子图2: R² 曲线
            ax2 = axes[0, 1]
            ax2.plot(epochs, self.history['val_r2'], 'g-', label='Subject R² (被试级)', linewidth=2)
            ax2.plot(epochs, self.history['val_segment_r2'], 'b--', label='Segment R² (片段级)', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('R2')
            ax2.set_title('R2 曲线')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # 标记最佳 R²
            best_r2_epoch = np.argmax(self.history['val_r2']) + 1
            best_r2 = max(self.history['val_r2'])
            ax2.axvline(x=best_r2_epoch, color='orange', linestyle='--', alpha=0.7)
            ax2.scatter([best_r2_epoch], [best_r2], color='orange', s=100, zorder=5)
            ax2.annotate(f'Best: {best_r2:.4f}', xy=(best_r2_epoch, best_r2),
                         xytext=(10, -10), textcoords='offset points', fontsize=9)

            # 子图3: MAE 曲线
            ax3 = axes[1, 0]
            ax3.plot(epochs, self.history['val_mae'], 'm-', label='Subject MAE (被试级)', linewidth=2)
            ax3.plot(epochs, self.history['val_segment_mae'], 'c--', label='Segment MAE (片段级)', linewidth=2)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('MAE')
            ax3.set_title('MAE 曲线')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # 标记最佳 MAE
            best_mae_epoch = np.argmin(self.history['val_mae']) + 1
            best_mae = min(self.history['val_mae'])
            ax3.axvline(x=best_mae_epoch, color='purple', linestyle='--', alpha=0.7)
            ax3.scatter([best_mae_epoch], [best_mae], color='purple', s=100, zorder=5)
            ax3.annotate(f'Best: {best_mae:.4f}', xy=(best_mae_epoch, best_mae),
                         xytext=(10, 10), textcoords='offset points', fontsize=9)

        # 子图4: Learning Rate 曲线
        ax4 = axes[1, 1]
        ax4.plot(epochs, self.history['learning_rate'], 'c-', label='Learning Rate', linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.set_title('学习率曲线')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')  # 对数刻度更清晰

        plt.tight_layout()

        # 保存图片
        if save_path is None:
            save_path = os.path.join(self.config.experiment.output_dir, f'training_curves_fold{fold}.png')

        plt.savefig(save_path, dpi=self.config.output.figure_dpi, bbox_inches='tight')
        plt.close(fig)

        logger.info(f'训练曲线已保存: {save_path}')
        return save_path

    def load_model(self, model_path: str) -> None:
        """加载模型"""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model = self._create_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f'模型已加载: {model_path}')
