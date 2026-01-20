# -*- coding: utf-8 -*-
"""
CADT 域适应训练器模块

提供 CADT-Transformer 模型的两阶段训练逻辑。
"""

import os
import logging
from typing import Dict, List, Optional, Tuple
from itertools import cycle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from src.models.dl_cadt_models import CADTTransformerModel
from src.models.dl_cadt_config import CADTConfig
from src.models.dl_dataset import collate_fn

logger = logging.getLogger(__name__)


class CADTTrainer:
    """
    CADT 域适应训练器

    实现两阶段训练流程：
    1. 预训练阶段：分类损失 + 域对抗损失
    2. 正式训练阶段：分类损失 + 原型聚类损失 + 域对抗损失
    """

    def __init__(self, config: CADTConfig):
        """
        初始化

        Args:
            config: CADT 配置对象
        """
        self.config = config
        self.device = torch.device(config.device)

        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)

        # 模型和优化器
        self.model: Optional[CADTTransformerModel] = None
        self.optimizers: Dict[str, AdamW] = {}

        # 混合精度训练
        self.use_amp = config.use_amp and torch.cuda.is_available()
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

        # 日志
        logger.info(f'CADTTrainer 初始化完成')
        logger.info(f'目标域: {config.target_domain}')
        logger.info(f'设备: {self.device}')

    def _create_model(self) -> CADTTransformerModel:
        """创建 CADT 模型"""
        model = CADTTransformerModel(self.config)
        model = model.to(self.device)

        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f'模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}')

        return model

    def _create_optimizers(self, model: CADTTransformerModel) -> Dict[str, AdamW]:
        """
        创建分离的优化器

        Args:
            model: CADT 模型

        Returns:
            优化器字典
        """
        optimizers = {
            'encoder': AdamW(
                model.encoder.parameters(),
                lr=self.config.encoder_lr,
                weight_decay=self.config.weight_decay,
            ),
            'classifier': AdamW(
                model.classifier.parameters(),
                lr=self.config.classifier_lr,
                weight_decay=self.config.weight_decay,
            ),
            'discriminator': AdamW(
                model.discriminator.parameters(),
                lr=self.config.discriminator_lr,
                weight_decay=self.config.weight_decay,
            ),
            'discriminator2': AdamW(
                model.discriminator2.parameters(),
                lr=self.config.discriminator_lr,
                weight_decay=self.config.weight_decay,
            ),
        }
        return optimizers

    def _create_data_loaders(
        self,
        source_dataset,
        target_dataset,
        val_dataset=None,
    ) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        """
        创建数据加载器

        Args:
            source_dataset: 源域数据集
            target_dataset: 目标域数据集
            val_dataset: 验证数据集（可选）

        Returns:
            (source_loader, target_loader, val_loader)
        """
        loader_kwargs = {
            'batch_size': self.config.batch_size,
            'collate_fn': collate_fn,
            'num_workers': self.config.num_workers,
            'pin_memory': self.config.pin_memory and torch.cuda.is_available(),
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

    def _train_epoch_pretrain(
        self,
        source_loader: DataLoader,
        target_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        预训练阶段的 epoch（支持 AMP）

        Args:
            source_loader: 源域数据加载器
            target_loader: 目标域数据加载器

        Returns:
            损失字典
        """
        self.model.train()

        total_losses = {'ce_loss': 0.0, 'domain_loss': 0.0, 'total_loss': 0.0}
        num_batches = 0

        target_iter = cycle(iter(target_loader))

        for source_batch in source_loader:
            target_batch = next(target_iter)

            # 清零梯度
            for opt in self.optimizers.values():
                opt.zero_grad(set_to_none=True)

            # 混合精度前向传播
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    losses = self.model.train_step(
                        source_batch, target_batch,
                        device=self.device,
                        change_center=True,
                        kl_w=0.0,
                        dis_w=self.config.pre_train_dis_weight,  # 使用预训练阶段专用权重
                    )

                # 混合精度反向传播
                self.scaler.scale(losses['total_loss']).backward()

                # 梯度裁剪
                if self.config.grad_clip > 0:
                    self.scaler.unscale_(self.optimizers['encoder'])
                    self.scaler.unscale_(self.optimizers['classifier'])
                    self.scaler.unscale_(self.optimizers['discriminator2'])
                    nn.utils.clip_grad_norm_(self.model.encoder.parameters(), self.config.grad_clip)
                    nn.utils.clip_grad_norm_(self.model.classifier.parameters(), self.config.grad_clip)
                    nn.utils.clip_grad_norm_(self.model.discriminator2.parameters(), self.config.grad_clip)

                # 更新参数
                for opt_name in ['encoder', 'classifier', 'discriminator2']:
                    self.scaler.step(self.optimizers[opt_name])
                self.scaler.update()
            else:
                # 标准训练
                losses = self.model.train_step(
                    source_batch, target_batch,
                    device=self.device,
                    change_center=True,
                    kl_w=0.0,
                    dis_w=self.config.cadt_dis_weight,
                )

                losses['total_loss'].backward()

                if self.config.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.encoder.parameters(), self.config.grad_clip)
                    nn.utils.clip_grad_norm_(self.model.classifier.parameters(), self.config.grad_clip)
                    nn.utils.clip_grad_norm_(self.model.discriminator2.parameters(), self.config.grad_clip)

                for opt_name in ['encoder', 'classifier', 'discriminator2']:
                    self.optimizers[opt_name].step()

            # 累计损失
            for key in total_losses:
                if key in losses:
                    val = losses[key] if isinstance(losses[key], float) else losses[key].item()
                    total_losses[key] += val
            num_batches += 1

        # 计算平均损失
        for key in total_losses:
            total_losses[key] /= num_batches

        return total_losses

    def _train_epoch_main(
        self,
        source_loader: DataLoader,
        target_loader: DataLoader,
        kl_w: float,
        dis_w: float,
    ) -> Dict[str, float]:
        """
        正式训练阶段的 epoch（支持 AMP）

        Args:
            source_loader: 源域数据加载器
            target_loader: 目标域数据加载器
            kl_w: 原型聚类损失权重
            dis_w: 域对抗损失权重

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

            # === 第一阶段：更新 encoder, classifier, discriminator2 ===
            # 清零梯度（不包括 discriminator，因为它在第二阶段单独更新）
            self.optimizers['encoder'].zero_grad(set_to_none=True)
            self.optimizers['classifier'].zero_grad(set_to_none=True)
            self.optimizers['discriminator2'].zero_grad(set_to_none=True)

            # 混合精度前向传播
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    losses = self.model.train_step(
                        source_batch, target_batch,
                        device=self.device,
                        change_center=False,
                        kl_w=kl_w,
                        dis_w=dis_w,
                    )

                # 第一阶段反向传播
                self.scaler.scale(losses['total_loss']).backward()

                # 梯度裁剪
                if self.config.grad_clip > 0:
                    self.scaler.unscale_(self.optimizers['encoder'])
                    self.scaler.unscale_(self.optimizers['classifier'])
                    self.scaler.unscale_(self.optimizers['discriminator2'])
                    nn.utils.clip_grad_norm_(self.model.encoder.parameters(), self.config.grad_clip)
                    nn.utils.clip_grad_norm_(self.model.classifier.parameters(), self.config.grad_clip)
                    nn.utils.clip_grad_norm_(self.model.discriminator2.parameters(), self.config.grad_clip)

                # 更新 encoder, classifier, discriminator2
                self.scaler.step(self.optimizers['encoder'])
                self.scaler.step(self.optimizers['classifier'])
                self.scaler.step(self.optimizers['discriminator2'])

                # === 第二阶段：单独更新 discriminator（与原始 CADT 一致）===
                if losses.get('need_discriminator_step', False):
                    self.optimizers['discriminator'].zero_grad(set_to_none=True)
                    dis_loss_real = losses['dis_loss_for_discriminator']
                    self.scaler.scale(dis_loss_real).backward()
                    if self.config.grad_clip > 0:
                        self.scaler.unscale_(self.optimizers['discriminator'])
                        nn.utils.clip_grad_norm_(self.model.discriminator.parameters(), self.config.grad_clip)
                    self.scaler.step(self.optimizers['discriminator'])

                self.scaler.update()
            else:
                # 标准训练
                losses = self.model.train_step(
                    source_batch, target_batch,
                    device=self.device,
                    change_center=False,
                    kl_w=kl_w,
                    dis_w=dis_w,
                )

                # 第一阶段反向传播
                losses['total_loss'].backward()

                if self.config.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.encoder.parameters(), self.config.grad_clip)
                    nn.utils.clip_grad_norm_(self.model.classifier.parameters(), self.config.grad_clip)
                    nn.utils.clip_grad_norm_(self.model.discriminator2.parameters(), self.config.grad_clip)

                self.optimizers['encoder'].step()
                self.optimizers['classifier'].step()
                self.optimizers['discriminator2'].step()

                # === 第二阶段：单独更新 discriminator ===
                if losses.get('need_discriminator_step', False):
                    self.optimizers['discriminator'].zero_grad(set_to_none=True)
                    dis_loss_real = losses['dis_loss_for_discriminator']
                    dis_loss_real.backward()
                    if self.config.grad_clip > 0:
                        nn.utils.clip_grad_norm_(self.model.discriminator.parameters(), self.config.grad_clip)
                    self.optimizers['discriminator'].step()

            # 累计损失
            for key in total_losses:
                if key in losses:
                    val = losses[key] if isinstance(losses[key], float) else losses[key].item()
                    total_losses[key] += val
            num_batches += 1

        # 计算平均损失
        for key in total_losses:
            total_losses[key] /= num_batches

        return total_losses

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        验证模型

        Args:
            val_loader: 验证数据加载器

        Returns:
            验证指标字典
        """
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                loss, correct, total = self.model.compute_classification_loss(batch, self.device)

                total_loss += loss * total
                total_correct += correct
                total_samples += total

                # 获取预测结果
                segments = batch['segments'].to(self.device)
                segment_mask = batch['segment_mask'].to(self.device)
                task_mask = batch['task_mask'].to(self.device)
                segment_lengths = batch['segment_lengths'].to(self.device)
                labels = batch['label']

                output = self.model(segments, segment_mask, task_mask, segment_lengths)
                preds = output['prediction'].argmax(dim=1).cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        metrics = {
            'loss': total_loss / total_samples,
            'accuracy': total_correct / total_samples,
            'f1': f1_score(all_labels, all_preds, average='macro'),
        }

        return metrics

    def train(
        self,
        source_dataset,
        target_dataset,
        val_dataset=None,
    ) -> Dict[str, float]:
        """
        两阶段训练流程

        Args:
            source_dataset: 源域数据集
            target_dataset: 目标域数据集
            val_dataset: 验证数据集（可选，默认使用目标域）

        Returns:
            最佳验证指标
        """
        # 创建模型和优化器
        self.model = self._create_model()
        self.optimizers = self._create_optimizers(self.model)

        # 创建数据加载器
        if val_dataset is None:
            val_dataset = target_dataset
        source_loader, target_loader, val_loader = self._create_data_loaders(
            source_dataset, target_dataset, val_dataset
        )

        best_metrics = None
        best_model_state = None

        # ========== 阶段1：预训练 ==========
        logger.info('=' * 60)
        logger.info('阶段1：预训练（分类 + 域对抗）')
        logger.info('=' * 60)

        pbar = tqdm(range(self.config.pre_train_epochs), desc='预训练')
        for epoch in pbar:
            losses = self._train_epoch_pretrain(source_loader, target_loader)
            val_metrics = self.validate(val_loader)

            # 记录历史
            self.history['train_loss'].append(losses['total_loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['ce_loss'].append(losses['ce_loss'])
            self.history['kl_loss'].append(0.0)
            self.history['domain_loss'].append(losses['domain_loss'])

            pbar.set_postfix({
                'loss': f"{losses['total_loss']:.3f}",
                'val_acc': f"{val_metrics['accuracy']:.3f}",
            })

        # ========== 中间步骤：初始化原型中心 ==========
        logger.info('=' * 60)
        logger.info('初始化类别原型中心')
        logger.info('=' * 60)

        self.model.init_center_c(source_loader, self.device)
        logger.info(f'原型中心形状: {self.model.c.shape}')

        # 根据配置执行重置
        if self.config.reset_mode == 'full':
            # 完全重置：重建网络和优化器（原始 CADT 行为）
            self.model.reset()
            self.optimizers = self._create_optimizers(self.model)
            logger.info('网络和优化器已完全重置')
        elif self.config.reset_mode == 'optimizer':
            # 仅重置优化器（推荐：保留 Transformer 预训练权重）
            self.optimizers = self._create_optimizers(self.model)
            logger.info('优化器已重置（保留网络权重）')
        else:
            # 不重置，但应用学习率衰减以稳定训练
            logger.info('跳过重置（reset_mode=none）')
            phase2_lr_decay = 0.1  # 学习率衰减为原来的 10%
            for opt_name, opt in self.optimizers.items():
                for param_group in opt.param_groups:
                    old_lr = param_group['lr']
                    param_group['lr'] = old_lr * phase2_lr_decay
                    logger.info(f'{opt_name} 学习率: {old_lr:.2e} -> {param_group["lr"]:.2e}')

        # ========== 阶段2：正式训练 ==========
        logger.info('=' * 60)
        logger.info('阶段2：正式训练（分类 + 原型聚类 + 域对抗）')
        logger.info('=' * 60)

        main_epochs = self.config.epochs - self.config.pre_train_epochs
        kl_warmup_epochs = min(20, main_epochs // 4)  # KL Loss 预热 epoch 数
        pbar = tqdm(range(main_epochs), desc='正式训练')

        for epoch in pbar:
            # KL Loss 渐进式增加（前 kl_warmup_epochs 个 epoch 线性增加）
            if epoch < kl_warmup_epochs:
                kl_w = self.config.cadt_kl_weight * (epoch + 1) / kl_warmup_epochs
            else:
                kl_w = self.config.cadt_kl_weight

            losses = self._train_epoch_main(
                source_loader, target_loader,
                kl_w=kl_w,
                dis_w=self.config.cadt_dis_weight,
            )
            val_metrics = self.validate(val_loader)

            # 记录历史
            self.history['train_loss'].append(losses['total_loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['ce_loss'].append(losses['ce_loss'])
            self.history['kl_loss'].append(losses['kl_loss'])
            self.history['dis_loss'].append(losses.get('dis_loss', 0.0))
            self.history['domain_loss'].append(losses['domain_loss'])

            pbar.set_postfix({
                'loss': f"{losses['total_loss']:.3f}",
                'kl': f"{losses['kl_loss']:.3f}",
                'val_acc': f"{val_metrics['accuracy']:.3f}",
                'val_f1': f"{val_metrics['f1']:.3f}",
            })

            # 保存最佳模型
            if best_metrics is None or val_metrics['accuracy'] > best_metrics['accuracy']:
                best_metrics = val_metrics.copy()
                best_metrics['epoch'] = self.config.pre_train_epochs + epoch + 1
                best_model_state = self.model.state_dict().copy()

        # 恢复最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        # 保存模型
        self._save_model(best_metrics)

        # 绘制训练曲线
        self._plot_training_curves()

        logger.info('=' * 60)
        logger.info(f'训练完成！最佳验证准确率: {best_metrics["accuracy"]:.4f} @ Epoch {best_metrics["epoch"]}')
        logger.info('=' * 60)

        return best_metrics

    def _save_model(self, metrics: Dict[str, float]):
        """保存模型"""
        model_path = os.path.join(
            self.config.output_dir,
            f'cadt_model_{self.config.target_domain}.pt'
        )
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'metrics': metrics,
            'history': self.history,
        }, model_path)
        logger.info(f'模型已保存: {model_path}')

    def _plot_training_curves(self):
        """绘制训练曲线"""
        if not HAS_MATPLOTLIB:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'CADT 训练曲线 (目标域: {self.config.target_domain})', fontsize=14)

        epochs = range(1, len(self.history['train_loss']) + 1)

        # Loss
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss')
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss')
        axes[0, 0].axvline(x=self.config.pre_train_epochs, color='g', linestyle='--', label='阶段切换')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('损失曲线')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Accuracy
        axes[0, 1].plot(epochs, self.history['val_accuracy'], 'g-', label='Val Accuracy')
        axes[0, 1].axvline(x=self.config.pre_train_epochs, color='r', linestyle='--', label='阶段切换')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('准确率曲线')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # CADT Losses
        axes[1, 0].plot(epochs, self.history['ce_loss'], 'b-', label='CE Loss')
        axes[1, 0].plot(epochs, self.history['kl_loss'], 'r-', label='KL Loss')
        axes[1, 0].plot(epochs, self.history['domain_loss'], 'g-', label='Domain Loss')
        axes[1, 0].axvline(x=self.config.pre_train_epochs, color='k', linestyle='--', label='阶段切换')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('CADT 损失分解')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # F1 Score
        axes[1, 1].plot(epochs, self.history['val_f1'], 'm-', label='Val F1')
        axes[1, 1].axvline(x=self.config.pre_train_epochs, color='r', linestyle='--', label='阶段切换')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].set_title('F1 曲线')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = os.path.join(
            self.config.output_dir,
            f'cadt_training_curves_{self.config.target_domain}.png'
        )
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        logger.info(f'训练曲线已保存: {save_path}')

    def evaluate(self, dataset) -> Dict[str, float]:
        """
        评估模型

        Args:
            dataset: 评估数据集

        Returns:
            评估指标字典
        """
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )
        return self.validate(loader)

    def load_model(self, model_path: str):
        """加载模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = self._create_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f'模型已加载: {model_path}')
