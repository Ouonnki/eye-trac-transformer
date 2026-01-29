# -*- coding: utf-8 -*-
"""
片段级CADT训练器

使用DANN模式（GRL）实现域适应，无需交替优化。
两阶段训练：
1. 预训练阶段(0-pre_train_epochs): 分类 + 弱域适应
2. 域适应阶段(pre_train_epochs-epochs): 初始化中心 + 强域适应

reset模式支持：
- none: 不重置，只初始化类中心（推荐）
- optimizer: 仅重置优化器（折中）
- full: 完全重置模型（激进，与原CADT一致）
"""

import logging
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from src.config import UnifiedConfig
from src.models.dl_dataset import SequenceConfig
from src.models.segment_cadt import (
    SegmentCADTModel,
    DualStreamTransformerEncoder,
)

logger = logging.getLogger(__name__)


class SegmentCADTTrainer:
    """
    片段级CADT训练器（DANN模式）

    使用GRL实现对抗训练，单步更新所有组件。
    """

    def __init__(
        self,
        config: UnifiedConfig,
        seq_config: SequenceConfig,
        target_domain: str = 'test1',
    ):
        self.config = config
        self.seq_config = seq_config
        self.target_domain = target_domain
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        n_class = config.task.num_classes
        d_model = config.model.segment_d_model

        # 创建双流编码器
        self.dual_stream = DualStreamTransformerEncoder(
            input_dim=config.model.input_dim,
            d_model=d_model,
            nhead=config.model.segment_nhead,
            num_layers=config.model.segment_num_layers,
            dim_feedforward=d_model * 4,
            dropout=config.model.dropout,
            max_seq_len=seq_config.max_seq_len,
            use_gradient_checkpointing=config.device.use_gradient_checkpointing,
            use_task_embedding=getattr(config.model, 'use_task_embedding', False),
            task_embedding_dim=config.model.task_embedding_dim,
            specific_noise_std=0.1,  # 特有流输入噪声
        )

        # 创建CADT模型
        self.model = SegmentCADTModel(
            dual_stream_encoder=self.dual_stream,
            n_class=n_class,
            d_model=d_model,
            device=self.device,
            grl_lambda=1.0,
        )

        # 优化器（DANN模式：单步更新所有组件）
        self.lr = config.cadt.encoder_lr
        self.optimizer = torch.optim.AdamW(
            list(self.model.dual_stream.parameters()) +
            list(self.model.classifier.parameters()) +
            list(self.model.discriminator.parameters()) +
            list(self.model.discriminator2.parameters()),
            lr=self.lr,
        )

        # CADT配置
        self.pre_train_epochs = config.cadt.pre_train_epochs
        self.kl_weight = config.cadt.cadt_kl_weight
        self.dis_weight = config.cadt.cadt_dis_weight
        self.reset_mode = config.cadt.reset_mode

        logger.info(f"CADT Trainer initialized with reset_mode={self.reset_mode}")

    def _reset_optimizer(self):
        """重置优化器"""
        self.optimizer = torch.optim.AdamW(
            list(self.model.dual_stream.parameters()) +
            list(self.model.classifier.parameters()) +
            list(self.model.discriminator.parameters()) +
            list(self.model.discriminator2.parameters()),
            lr=self.lr,
        )
        logger.info("Optimizer reset")

    def _reset_model(self):
        """完全重置模型"""
        self.model.reset_full()
        self._reset_optimizer()
        logger.info("Model fully reset")

    def _apply_reset_strategy(self):
        """根据配置应用reset策略"""
        if self.reset_mode == 'none':
            logger.info("Reset mode: none - 只初始化类中心，不重置模型")
        elif self.reset_mode == 'optimizer':
            logger.info("Reset mode: optimizer - 重置优化器")
            self._reset_optimizer()
        elif self.reset_mode == 'full':
            logger.info("Reset mode: full - 完全重置模型")
            self._reset_model()
        else:
            raise ValueError(f"Unknown reset_mode: {self.reset_mode}")

    def train_epoch(
        self,
        source_loader: DataLoader,
        target_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """
        训练一个epoch（DANN模式：单步更新）

        Args:
            source_loader: 源域数据加载器
            target_loader: 目标域数据加载器
            epoch: 当前epoch

        Returns:
            损失字典
        """
        self.model.train()

        # 确定阶段
        is_pretrain = epoch < self.pre_train_epochs
        change_center = is_pretrain
        kl_w = 0.0 if is_pretrain else self.kl_weight
        dis_w = 0.1 if is_pretrain else self.dis_weight

        total_loss = 0.0
        n_batches = 0

        for (source_batch, target_batch) in zip(source_loader, target_loader):
            # 准备源域数据
            source_features = source_batch['features'].to(self.device)
            source_labels = source_batch['labels'].to(self.device)

            # 准备目标域数据
            target_features = target_batch['features'].to(self.device)
            target_batch_size = target_features.size(0)

            # 域标签 (0=源域, 1=目标域)
            source_batch_size = source_features.size(0)
            source_domain_labels = torch.zeros(source_batch_size, 1).to(self.device)
            target_domain_labels = torch.ones(target_batch_size, 1).to(self.device)

            # 合并数据
            features = torch.cat([source_features, target_features], dim=0)
            # labels: 源域部分使用真实标签，目标域部分使用0填充（会被模型切片掉，仅占位）
            num_source = source_batch_size
            dummy_target_labels = torch.zeros(target_batch_size, dtype=source_labels.dtype).to(self.device)
            labels = torch.cat([source_labels, dummy_target_labels], dim=0)
            domain_labels = torch.cat([source_domain_labels, target_domain_labels], dim=0)

            # 任务条件
            task_conditions = None
            if 'task_conditions' in source_batch:
                task_conditions = {}
                for k, v in source_batch['task_conditions'].items():
                    source_v = v.to(self.device)
                    target_v = target_batch['task_conditions'][k].to(self.device)
                    task_conditions[k] = torch.cat([source_v, target_v], dim=0)

            # 前向传播（GRL自动处理对抗梯度）
            losses = self.model(
                features=features,
                lengths=None,
                labels=labels,
                domain_labels=domain_labels,
                change_center=change_center,
                kl_w=kl_w,
                dis_w=dis_w,
                num_source=num_source,  # 传入源域样本数量，用于切片
            )

            # 反向传播（单步更新所有组件）
            self.optimizer.zero_grad()
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += losses['total_loss'].item()
            n_batches += 1

        return {
            'total_loss': total_loss / n_batches,
            'ce_loss': losses['ce_loss'].item(),
            'kl_loss': losses['kl_loss'].item(),
            'hyper_sphere_loss': losses['hyper_sphere_loss'].item(),
        }

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        评估模型

        Args:
            data_loader: 数据加载器

        Returns:
            评估指标字典
        """
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch in data_loader:
            features = batch['features'].to(self.device)
            labels = batch['labels'].to(self.device)

            loss, batch_correct, batch_total = self.model.compute_loss(features, labels)
            total_loss += loss * batch_total
            correct += batch_correct
            total += batch_total

        return {
            'loss': total_loss / total if total > 0 else 0.0,
            'accuracy': correct / total if total > 0 else 0.0,
        }

    def train(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        output_dir: str,
    ) -> Dict[str, list]:
        """
        两阶段训练

        Args:
            train_loader: 源域(train)数据加载器
            test_loader: 目标域(test1/test2/test3)数据加载器
            output_dir: 输出目录

        Returns:
            训练历史
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        n_epochs = self.config.training.epochs
        history = {'train_loss': [], 'test_acc': [], 'test_loss': []}
        best_acc = 0.0
        best_epoch = 0

        logger.info(f"Starting training for {n_epochs} epochs")
        logger.info(f"Pre-train epochs: {self.pre_train_epochs}")
        logger.info(f"KL weight: {self.kl_weight}, Target domain: {self.target_domain}")

        for epoch in range(n_epochs):
            # 阶段转换
            if epoch == self.pre_train_epochs:
                logger.info(f"Epoch {epoch}: 初始化类中心...")
                self.model.eval()
                self.model.init_centers(train_loader)
                logger.info(f"类中心初始化完成: {self.model.c.data.cpu()}")
                self._apply_reset_strategy()

            # 训练
            train_metrics = self.train_epoch(train_loader, test_loader, epoch)
            test_metrics = self.evaluate(test_loader)

            history['train_loss'].append(train_metrics['total_loss'])
            history['test_acc'].append(test_metrics['accuracy'])
            history['test_loss'].append(test_metrics['loss'])

            # 记录最佳
            if test_metrics['accuracy'] > best_acc:
                best_acc = test_metrics['accuracy']
                best_epoch = epoch
                self._save_checkpoint(output_path / 'best_model.pth', epoch)

            # 日志
            stage = "Pretrain" if epoch < self.pre_train_epochs else "Domain Adapt"
            logger.info(
                f"{stage} Epoch {epoch}: "
                f"train_loss={train_metrics['total_loss']:.4f}, "
                f"ce_loss={train_metrics['ce_loss']:.4f}, "
                f"kl_loss={train_metrics['kl_loss']:.4f}, "
                f"test_acc={test_metrics['accuracy']:.4f}, "
                f"best_acc={best_acc:.4f} (epoch {best_epoch})"
            )

        logger.info(f"Training completed. Best accuracy: {best_acc:.4f} at epoch {best_epoch}")
        return history

    def _save_checkpoint(self, path: Path, epoch: int):
        """
        保存检查点

        Args:
            path: 保存路径
            epoch: 当前epoch
        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'class_centers': self.model.c.data,
        }, path)
