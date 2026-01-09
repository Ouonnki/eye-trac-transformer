# -*- coding: utf-8 -*-
"""
域迁移训练器模块

提供基于 CADT 方法的域迁移训练流程。
仅在分类模式下启用域迁移，回归模式使用标准训练。
"""

import os
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm

from src.models.dl_trainer import (
    DeepLearningTrainer,
    TrainingConfig,
    EarlyStopping,
    get_warmup_scheduler,
)
from src.models.dl_models import DomainAdaptiveHierarchicalNetwork
from src.models.dl_dataset import HierarchicalGazeDataset, collate_fn
from src.models.domain_losses import CADTLoss

logger = logging.getLogger(__name__)


@dataclass
class DomainAdaptationConfig(TrainingConfig):
    """域迁移训练配置"""

    # 域迁移开关
    enable_domain_adaptation: bool = True

    # 两阶段训练
    pretrain_epochs: int = 20  # 预训练阶段 epoch 数

    # CADT 损失权重
    center_alignment_weight: float = 10.0   # λ1
    domain_adversarial_weight: float = 1.0  # λ2
    center_diversity_weight: float = 0.1    # λ3

    # GRL alpha 调度
    grl_alpha_max: float = 1.0
    grl_alpha_warmup_epochs: int = 10


class DomainAdaptationTrainer(DeepLearningTrainer):
    """
    域迁移训练器

    基于 CADT 方法的两阶段训练：
    1. 预训练阶段：仅在源域上训练分类器
    2. 域迁移阶段：加入域对抗和类中心对齐损失

    仅在分类模式下启用域迁移。
    """

    def __init__(self, config: DomainAdaptationConfig):
        """
        初始化

        Args:
            config: 域迁移训练配置
        """
        super().__init__(config)
        self.da_config = config

        # CADT 损失函数
        self.cadt_loss = CADTLoss(
            center_weight=config.center_alignment_weight,
            adversarial_weight=config.domain_adversarial_weight,
            diversity_weight=config.center_diversity_weight,
        )

    def _create_model(self) -> nn.Module:
        """创建域适应模型"""
        if (self.config.task_type == 'classification'
                and self.da_config.enable_domain_adaptation):
            # 分类模式 + 域迁移：使用域适应模型
            model = DomainAdaptiveHierarchicalNetwork(
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
                use_gradient_checkpointing=self.config.use_gradient_checkpointing,
                task_type=self.config.task_type,
                num_classes=self.config.num_classes,
            )
            model = model.to(self.device)
            logger.info('使用域适应模型 DomainAdaptiveHierarchicalNetwork')
        else:
            # 回归模式或禁用域迁移：使用基础模型
            model = super()._create_model()
            logger.info('使用基础模型 HierarchicalTransformerNetwork')

        # 多GPU并行
        if self.n_gpus > 1 and self.config.use_multi_gpu:
            model = nn.DataParallel(model)
            logger.info(f'模型已包装为 DataParallel ({self.n_gpus} GPUs)')

        return model

    def _get_base_model(self, model: nn.Module) -> nn.Module:
        """获取去除 DataParallel 包装的基础模型"""
        if isinstance(model, nn.DataParallel):
            return model.module
        return model

    def _init_class_centers(
        self,
        model: nn.Module,
        train_loader: DataLoader,
    ) -> None:
        """
        初始化类中心

        在预训练阶段结束后，使用源域样本的平均特征初始化类中心。

        Args:
            model: 模型
            train_loader: 训练数据加载器
        """
        base_model = self._get_base_model(model)

        num_classes = self.config.num_classes
        feature_dim = self.config.task_d_model

        # 累积特征
        class_features = {i: [] for i in range(num_classes)}

        model.eval()
        with torch.no_grad():
            for batch in train_loader:
                segments = batch['segments'].to(self.device)
                segment_mask = batch['segment_mask'].to(self.device)
                task_mask = batch['task_mask'].to(self.device)
                segment_lengths = batch['segment_lengths'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = model(
                    segments=segments,
                    segment_mask=segment_mask,
                    task_mask=task_mask,
                    segment_lengths=segment_lengths,
                )

                # 收集每个类别的特征
                subject_repr = outputs['subject_repr']  # (batch, feature_dim)
                for i, label in enumerate(labels):
                    class_features[label.item()].append(
                        subject_repr[i].cpu()
                    )

        # 计算类中心
        centers = torch.zeros(num_classes, feature_dim)
        for class_id in range(num_classes):
            if class_features[class_id]:
                class_tensor = torch.stack(class_features[class_id])
                centers[class_id] = class_tensor.mean(dim=0)
            else:
                logger.warning(f'类别 {class_id} 没有样本，使用零向量初始化')

        # 设置类中心
        base_model.center_bank.init_centers(centers.to(self.device))
        logger.info(f'类中心已初始化，形状: {centers.shape}')

    def _compute_grl_alpha(self, epoch: int, total_epochs: int) -> float:
        """
        计算 GRL alpha 值

        使用预热调度，逐渐增加 alpha。

        Args:
            epoch: 当前 epoch
            total_epochs: 总 epoch 数

        Returns:
            alpha 值
        """
        warmup = self.da_config.grl_alpha_warmup_epochs
        max_alpha = self.da_config.grl_alpha_max

        if epoch < warmup:
            # 线性预热
            return max_alpha * (epoch + 1) / warmup
        else:
            # 保持最大值
            return max_alpha

    def train_epoch_with_domain_adaptation(
        self,
        model: nn.Module,
        source_loader: DataLoader,
        target_loader: DataLoader,
        optimizer: AdamW,
        epoch: int,
    ) -> Dict[str, float]:
        """
        域迁移训练一个 epoch

        Args:
            model: 模型
            source_loader: 源域数据加载器（有标签）
            target_loader: 目标域数据加载器（无标签）
            optimizer: 优化器
            epoch: 当前 epoch

        Returns:
            损失字典
        """
        model.train()
        base_model = self._get_base_model(model)

        # 设置 GRL alpha
        alpha = self._compute_grl_alpha(
            epoch - self.da_config.pretrain_epochs,
            self.config.epochs - self.da_config.pretrain_epochs,
        )
        base_model.set_discriminator_alpha(alpha)

        # 统计
        total_losses = {
            'total': 0.0,
            'ce': 0.0,
            'center': 0.0,
            'domain': 0.0,
            'dist_domain': 0.0,
            'diversity': 0.0,
        }
        num_batches = 0

        # 同时迭代源域和目标域
        target_iter = iter(target_loader)

        for source_batch in source_loader:
            # 获取目标域批次（循环使用）
            try:
                target_batch = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_batch = next(target_iter)

            # 源域数据
            s_segments = source_batch['segments'].to(self.device)
            s_segment_mask = source_batch['segment_mask'].to(self.device)
            s_task_mask = source_batch['task_mask'].to(self.device)
            s_segment_lengths = source_batch['segment_lengths'].to(self.device)
            s_labels = source_batch['label'].to(self.device)

            # 目标域数据
            t_segments = target_batch['segments'].to(self.device)
            t_segment_mask = target_batch['segment_mask'].to(self.device)
            t_task_mask = target_batch['task_mask'].to(self.device)
            t_segment_lengths = target_batch['segment_lengths'].to(self.device)

            optimizer.zero_grad(set_to_none=True)

            # 源域前向传播
            s_outputs = model(
                segments=s_segments,
                segment_mask=s_segment_mask,
                task_mask=s_task_mask,
                segment_lengths=s_segment_lengths,
            )

            # 目标域前向传播
            t_outputs = model(
                segments=t_segments,
                segment_mask=t_segment_mask,
                task_mask=t_task_mask,
                segment_lengths=t_segment_lengths,
            )

            # 获取域判别概率
            s_domain_probs, s_dist_probs = base_model.get_domain_probs(
                s_outputs['subject_repr'],
                s_labels,
            )
            t_domain_probs, t_dist_probs = base_model.get_domain_probs(
                t_outputs['subject_repr'],
                labels=None,  # 目标域无标签
            )

            # 获取源域样本对应的类中心
            s_centers = base_model.center_bank.get_center(s_labels)

            # 计算 CADT 损失
            losses = self.cadt_loss(
                logits=s_outputs['prediction'],
                labels=s_labels,
                features=s_outputs['subject_repr'],
                centers=s_centers,
                source_domain_probs=s_domain_probs,
                target_domain_probs=t_domain_probs,
                source_dist_probs=s_dist_probs,
                target_dist_probs=t_dist_probs,
                all_centers=base_model.center_bank.centers,
            )

            # 反向传播
            losses['total'].backward()

            # 梯度裁剪
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.config.grad_clip,
                )

            optimizer.step()

            # 统计
            for key in total_losses:
                total_losses[key] += losses[key].item()
            num_batches += 1

        # 平均
        for key in total_losses:
            total_losses[key] /= num_batches

        return total_losses

    def train_with_domain_adaptation(
        self,
        train_dataset: HierarchicalGazeDataset,
        target_dataset: HierarchicalGazeDataset,
        val_dataset: HierarchicalGazeDataset,
        fold: int = 0,
    ) -> Dict[str, float]:
        """
        使用域迁移训练模型

        Args:
            train_dataset: 源域训练数据集（有标签）
            target_dataset: 目标域数据集（无标签）
            val_dataset: 验证数据集
            fold: 当前折数

        Returns:
            最佳验证指标
        """
        # 检查任务类型
        if self.config.task_type != 'classification':
            logger.warning('域迁移仅支持分类模式，将使用标准训练')
            return self.train(train_dataset, val_dataset, fold)

        if not self.da_config.enable_domain_adaptation:
            logger.info('域迁移已禁用，使用标准训练')
            return self.train(train_dataset, val_dataset, fold)

        logger.info('开始域迁移训练（两阶段）')
        logger.info(f'预训练阶段: {self.da_config.pretrain_epochs} epochs')
        logger.info(f'域迁移阶段: {self.config.epochs - self.da_config.pretrain_epochs} epochs')

        # 创建模型
        self.model = self._create_model()
        self.optimizer, self.scheduler = self._create_optimizer(self.model)

        # 创建数据加载器
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
        target_loader = DataLoader(
            target_dataset,
            shuffle=True,
            **loader_kwargs,
        )
        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            **loader_kwargs,
        )

        # 损失函数（预训练阶段使用）
        criterion = self._create_criterion()

        # 早停
        early_stopping = EarlyStopping(patience=self.config.patience, mode='min')

        # 最佳模型
        best_metrics = None
        best_model_state = None

        # 重置历史
        self.history = self._init_history()
        self.history['domain_loss'] = []  # 添加域损失历史

        # 训练循环
        pbar = tqdm(range(self.config.epochs), desc=f'Fold {fold+1}')

        for epoch in pbar:
            if epoch < self.da_config.pretrain_epochs:
                # 阶段1：预训练
                phase = 'pretrain'
                train_loss = self.train_epoch(
                    self.model, train_loader, self.optimizer, criterion
                )

                # 预训练结束时初始化类中心
                if epoch == self.da_config.pretrain_epochs - 1:
                    logger.info('预训练完成，初始化类中心...')
                    self._init_class_centers(self.model, train_loader)
            else:
                # 阶段2：域迁移
                phase = 'domain'
                losses = self.train_epoch_with_domain_adaptation(
                    self.model,
                    train_loader,
                    target_loader,
                    self.optimizer,
                    epoch,
                )
                train_loss = losses['total']
                self.history['domain_loss'].append(losses['domain'])

            # 验证
            val_metrics = self.validate(self.model, val_loader, criterion)

            # 更新学习率
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['learning_rate'].append(current_lr)
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_metrics['f1'])

            # 更新进度条
            pbar.set_postfix({
                'phase': phase,
                'loss': f'{train_loss:.3f}',
                'val_loss': f'{val_metrics["loss"]:.3f}',
                'Acc': f'{val_metrics["accuracy"]:.3f}',
                'lr': f'{current_lr:.1e}',
            })

            # 保存最佳模型
            if best_metrics is None or val_metrics['loss'] < best_metrics['loss']:
                best_metrics = val_metrics.copy()
                best_metrics['epoch'] = epoch + 1
                if isinstance(self.model, nn.DataParallel):
                    best_model_state = self.model.module.state_dict().copy()
                else:
                    best_model_state = self.model.state_dict().copy()

            # 早停检查（仅在域迁移阶段）
            if epoch >= self.da_config.pretrain_epochs:
                if early_stopping(val_metrics['loss'], epoch):
                    logger.info(f'早停于 epoch {epoch+1}')
                    break

        # 恢复最佳模型
        if best_model_state is not None:
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(best_model_state)
            else:
                self.model.load_state_dict(best_model_state)

        # 保存模型
        if self.config.save_best:
            model_path = os.path.join(
                self.config.output_dir,
                f'domain_adaptive_model_fold{fold}.pt'
            )
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

        # 保存训练曲线
        if self.config.save_figures:
            self.plot_training_curves(fold=fold)

        return best_metrics

    def evaluate_with_confusion_matrix(
        self,
        dataset: HierarchicalGazeDataset,
    ) -> Dict:
        """
        评估模型并返回混淆矩阵

        Args:
            dataset: 数据集

        Returns:
            评估结果字典
        """
        if self.model is None:
            raise ValueError('模型未训练或加载')

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

                preds = torch.argmax(outputs['prediction'], dim=-1)
                all_predictions.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        return {
            'accuracy': accuracy_score(all_labels, all_predictions),
            'f1_weighted': f1_score(all_labels, all_predictions, average='weighted'),
            'f1_macro': f1_score(all_labels, all_predictions, average='macro'),
            'confusion_matrix': confusion_matrix(all_labels, all_predictions),
            'predictions': all_predictions,
            'labels': all_labels,
        }
