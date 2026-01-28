# -*- coding: utf-8 -*-
"""
片段级 CADT 域适应训练器

提供片段级 CADT 模型的两阶段训练逻辑。
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from itertools import cycle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from src.models.segment_cadt_model import SegmentCADTModel
from src.models.dl_dataset import SequenceConfig, segment_collate_fn
from src.config import UnifiedConfig

logger = logging.getLogger(__name__)


class SegmentCADTTrainer:
    """
    片段级 CADT 域适应训练器

    实现两阶段训练流程：
    1. 预训练阶段：分类损失 + 域分类损失
    2. 正式训练阶段：分类损失 + 原型聚类损失 + 域对抗损失
    """

    def __init__(self, config: UnifiedConfig, seq_config: SequenceConfig):
        """
        初始化

        Args:
            config: 统一配置对象
            seq_config: 序列配置对象
        """
        self.config = config
        self.seq_config = seq_config
        self.device = torch.device(config.device.device)

        # 创建输出目录
        os.makedirs(config.experiment.output_dir, exist_ok=True)

        # 模型和优化器
        self.model: Optional[SegmentCADTModel] = None
        self.optimizers: Dict[str, AdamW] = {}

        # 混合精度训练
        self.use_amp = config.device.use_amp and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        if self.use_amp:
            logger.info('启用混合精度训练 (AMP)')

        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': [],
            'ce_loss': [],
            'kl_loss': [],
            'dis_loss': [],
            'domain_loss': [],
        }

        # 最佳模型
        self.best_val_accuracy = 0.0
        self.best_epoch = 0

        logger.info(f'SegmentCADTTrainer 初始化完成')
        logger.info(f'目标域: {config.cadt.target_domain}')
        logger.info(f'设备: {self.device}')

    def _create_model(self) -> SegmentCADTModel:
        """创建 CADT 模型"""
        model = SegmentCADTModel.from_config(
            config=self.config,
            seq_config=self.seq_config,
            num_classes=self.config.task.num_classes,
        )
        model = model.to(self.device)

        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f'模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}')

        return model

    def _create_optimizers(self, model: SegmentCADTModel) -> Dict[str, AdamW]:
        """创建分离的优化器"""
        optimizers = {
            'encoder': AdamW(
                model.encoder.parameters(),
                lr=self.config.cadt.encoder_lr,
                weight_decay=self.config.training.weight_decay,
            ),
            'discriminator': AdamW(
                model.discriminator.parameters(),
                lr=self.config.cadt.discriminator_lr,
                weight_decay=self.config.training.weight_decay,
            ),
            'discriminator2': AdamW(
                model.discriminator2.parameters(),
                lr=self.config.cadt.discriminator_lr,
                weight_decay=self.config.training.weight_decay,
            ),
        }
        return optimizers

    def _create_data_loaders(
        self,
        source_dataset,
        target_dataset,
        val_dataset=None,
    ) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        """创建数据加载器"""
        loader_kwargs = {
            'batch_size': self.config.training.batch_size,
            'collate_fn': segment_collate_fn,
            'num_workers': self.config.device.num_workers,
            'pin_memory': self.config.device.pin_memory and torch.cuda.is_available(),
        }

        source_loader = DataLoader(source_dataset, shuffle=True, **loader_kwargs)
        target_loader = DataLoader(target_dataset, shuffle=True, **loader_kwargs)

        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

        logger.info(f'源域样本数: {len(source_dataset)}, 目标域样本数: {len(target_dataset)}')
        if val_dataset is not None:
            logger.info(f'验证集样本数: {len(val_dataset)}')

        return source_loader, target_loader, val_loader

    def _train_epoch(
        self,
        source_loader: DataLoader,
        target_loader: DataLoader,
        is_pretrain: bool = False,
    ) -> Dict[str, float]:
        """
        训练一个 epoch

        Args:
            source_loader: 源域数据加载器
            target_loader: 目标域数据加载器
            is_pretrain: 是否为预训练阶段

        Returns:
            损失字典
        """
        self.model.train()

        total_losses = {
            'ce_loss': 0.0, 'kl_loss': 0.0, 'dis_loss': 0.0,
            'domain_loss': 0.0, 'total_loss': 0.0
        }
        num_batches = 0

        target_iter = cycle(iter(target_loader))

        for source_batch in source_loader:
            target_batch = next(target_iter)

            # 清零梯度
            for opt in self.optimizers.values():
                opt.zero_grad(set_to_none=True)

            # 前向传播
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    losses = self.model.train_step(
                        source_batch, target_batch,
                        device=self.device,
                        is_pretrain=is_pretrain,
                        kl_weight=self.config.cadt.cadt_kl_weight if not is_pretrain else 0.0,
                        dis_weight=self.config.cadt.cadt_dis_weight,
                    )
                # 混合精度反向传播
                self.scaler.scale(losses['total_loss']).backward()
                self.scaler.unscale_(self.optimizers['encoder'])
                torch.nn.utils.clip_grad_norm_(self.model.encoder.parameters(), self.config.training.grad_clip)
                for opt in self.optimizers.values():
                    self.scaler.step(opt)
                self.scaler.update()
            else:
                losses = self.model.train_step(
                    source_batch, target_batch,
                    device=self.device,
                    is_pretrain=is_pretrain,
                    kl_weight=self.config.cadt.cadt_kl_weight if not is_pretrain else 0.0,
                    dis_weight=self.config.cadt.cadt_dis_weight,
                )
                # 反向传播
                losses['total_loss'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.encoder.parameters(), self.config.training.grad_clip)

                # 更新参数
                self.optimizers['encoder'].step()
                self.optimizers['discriminator2'].step()

                # 单独更新 discriminator（如果需要）
                if losses.get('need_discriminator_step', False):
                    self.optimizers['discriminator'].zero_grad(set_to_none=True)
                    losses['dis_loss_for_discriminator'].backward()
                    self.optimizers['discriminator'].step()

            # 累计损失
            for key in total_losses:
                if key in losses:
                    val = losses[key]
                    if isinstance(val, torch.Tensor):
                        val = val.item()
                    total_losses[key] += val
            num_batches += 1

        # 计算平均损失
        for key in total_losses:
            total_losses[key] /= max(num_batches, 1)

        return total_losses

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        验证

        Args:
            val_loader: 验证数据加载器

        Returns:
            指标字典
        """
        self.model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(self.device)
                lengths = batch['length'].to(self.device)
                labels = batch['label']

                output = self.model(features, lengths)
                preds = output['prediction'].argmax(dim=1).cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        return {
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds, average='macro', zero_division=0),
            'precision': precision_score(all_labels, all_preds, average='macro', zero_division=0),
            'recall': recall_score(all_labels, all_preds, average='macro', zero_division=0),
        }

    def evaluate(self, dataset) -> Dict[str, float]:
        """评估数据集"""
        loader = DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            collate_fn=segment_collate_fn,
            num_workers=self.config.device.num_workers,
        )
        return self.validate(loader)

    def train(
        self,
        source_dataset,
        target_dataset,
        val_dataset=None,
    ) -> Dict[str, float]:
        """
        两阶段训练

        Args:
            source_dataset: 源域数据集
            target_dataset: 目标域数据集
            val_dataset: 验证数据集（可选，默认使用目标域）

        Returns:
            最佳指标字典
        """
        # 创建模型和优化器
        self.model = self._create_model()
        self.optimizers = self._create_optimizers(self.model)

        # 创建数据加载器
        source_loader, target_loader, val_loader = self._create_data_loaders(
            source_dataset, target_dataset, val_dataset
        )

        if val_loader is None:
            val_loader = target_loader

        best_metrics = None

        # 阶段1：预训练
        pre_train_epochs = self.config.cadt.pre_train_epochs
        logger.info(f'阶段1：预训练 ({pre_train_epochs} epochs)')

        for epoch in tqdm(range(pre_train_epochs), desc='预训练'):
            losses = self._train_epoch(source_loader, target_loader, is_pretrain=True)

            if (epoch + 1) % 10 == 0:
                metrics = self.validate(val_loader)
                logger.debug(f'预训练 Epoch {epoch+1}: loss={losses["total_loss"]:.4f}, '
                           f'acc={metrics["accuracy"]:.4f}')

        # 初始化原型中心
        logger.info('初始化原型中心...')
        self.model.init_center_c(source_loader, self.device)

        # 重置优化器（如果配置指定）
        if self.config.cadt.reset_mode == 'optimizer':
            self.optimizers = self._create_optimizers(self.model)
            logger.info('已重置优化器')
        elif self.config.cadt.reset_mode == 'full':
            self.model = self._create_model()
            self.model.init_center_c(source_loader, self.device)
            self.optimizers = self._create_optimizers(self.model)
            logger.info('已完全重置模型和优化器')

        # 阶段2：正式训练
        main_epochs = self.config.training.epochs - pre_train_epochs
        logger.info(f'阶段2：正式训练 ({main_epochs} epochs)')

        patience_counter = 0
        for epoch in tqdm(range(main_epochs), desc='正式训练'):
            losses = self._train_epoch(source_loader, target_loader, is_pretrain=False)

            # 记录历史
            self.history['train_loss'].append(losses['total_loss'])
            self.history['ce_loss'].append(losses['ce_loss'])
            self.history['kl_loss'].append(losses['kl_loss'])
            self.history['dis_loss'].append(losses['dis_loss'])
            self.history['domain_loss'].append(losses['domain_loss'])

            # 验证
            metrics = self.validate(val_loader)
            self.history['val_accuracy'].append(metrics['accuracy'])
            self.history['val_f1'].append(metrics['f1'])

            # 更新最佳模型
            if metrics['accuracy'] > self.best_val_accuracy:
                self.best_val_accuracy = metrics['accuracy']
                self.best_epoch = pre_train_epochs + epoch + 1
                best_metrics = metrics.copy()
                best_metrics['epoch'] = self.best_epoch

                # 保存最佳模型
                self._save_best_model()
                patience_counter = 0
            else:
                patience_counter += 1

            # 早停
            if patience_counter >= self.config.training.patience:
                logger.info(f'早停触发，最佳 epoch: {self.best_epoch}')
                break

            # 定期日志
            if (epoch + 1) % 10 == 0:
                logger.info(f'Epoch {pre_train_epochs + epoch + 1}: '
                           f'loss={losses["total_loss"]:.4f}, '
                           f'ce={losses["ce_loss"]:.4f}, '
                           f'kl={losses["kl_loss"]:.4f}, '
                           f'dis={losses["dis_loss"]:.4f}, '
                           f'acc={metrics["accuracy"]:.4f}, '
                           f'f1={metrics["f1"]:.4f}')

        # 绘制训练曲线
        if HAS_MATPLOTLIB and self.config.output.save_figures:
            self._plot_training_history()

        logger.info(f'训练完成，最佳准确率: {self.best_val_accuracy:.4f} (epoch {self.best_epoch})')

        return best_metrics if best_metrics else metrics

    def _save_best_model(self) -> None:
        """保存最佳模型"""
        if not self.config.output.save_best:
            return

        model_path = os.path.join(self.config.experiment.output_dir, 'best_model.pt')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'epoch': self.best_epoch,
            'accuracy': self.best_val_accuracy,
        }, model_path)
        logger.debug(f'保存最佳模型到 {model_path}')

    def _plot_training_history(self) -> None:
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 总损失
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # 各损失分量
        axes[0, 1].plot(self.history['ce_loss'], label='CE Loss')
        axes[0, 1].plot(self.history['kl_loss'], label='KL Loss')
        axes[0, 1].plot(self.history['dis_loss'], label='Dis Loss')
        axes[0, 1].plot(self.history['domain_loss'], label='Domain Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Loss Components')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # 准确率
        axes[1, 0].plot(self.history['val_accuracy'], label='Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Validation Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # F1
        axes[1, 1].plot(self.history['val_f1'], label='F1')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].set_title('Validation F1')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        fig_path = os.path.join(self.config.experiment.output_dir, 'training_history.png')
        plt.savefig(fig_path, dpi=self.config.output.figure_dpi)
        plt.close()
        logger.info(f'训练曲线已保存到 {fig_path}')
