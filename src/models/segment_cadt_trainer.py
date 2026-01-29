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
from typing import Dict, List
import numpy as np

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score

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
            评估指标字典（包含 accuracy, f1, precision, recall, loss）
        """
        self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_labels = []

        for batch in data_loader:
            features = batch['features'].to(self.device)
            labels = batch['labels'].to(self.device)

            loss, batch_correct, batch_total = self.model.compute_loss(features, labels)
            total_loss += loss * batch_total

            # 获取预测结果用于计算 F1
            f_invariant, _ = self.model.encode(features)
            pred_logits = self.model.classifier(f_invariant)
            pred_labels = pred_logits.argmax(dim=1)

            all_preds.extend(pred_labels.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        total = len(all_preds)

        return {
            'loss': total_loss / total if total > 0 else 0.0,
            'accuracy': (all_preds == all_labels).mean() if total > 0 else 0.0,
            'f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0) if total > 0 else 0.0,
            'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0) if total > 0 else 0.0,
            'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0) if total > 0 else 0.0,
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
        history = {
            'train_loss': [], 'ce_loss': [], 'kl_loss': [],
            'test_loss': [], 'test_acc': [], 'test_f1': [],
        }
        best_acc = 0.0
        best_epoch = 0

        # 打印训练开始信息
        logger.info("=" * 70)
        logger.info(f"开始 CADT 域适应训练 | 目标域: {self.target_domain}")
        logger.info("=" * 70)
        logger.info(f"总轮数: {n_epochs} | 预训练: {self.pre_train_epochs} | 域适应: {n_epochs - self.pre_train_epochs}")
        logger.info(f"KL权重: {self.kl_weight} | 域判别权重: {self.dis_weight} | Reset模式: {self.reset_mode}")
        logger.info("")

        for epoch in range(n_epochs):
            # 阶段转换
            if epoch == self.pre_train_epochs:
                logger.info("-" * 70)
                logger.info(f"[阶段切换] Epoch {epoch}: 初始化类中心并应用重置策略")
                self.model.eval()
                self.model.init_centers(train_loader)
                logger.info(f"类中心: {self.model.c.data.cpu().numpy()}")
                self._apply_reset_strategy()
                logger.info("")

            # 训练
            train_metrics = self.train_epoch(train_loader, test_loader, epoch)
            test_metrics = self.evaluate(test_loader)

            # 记录历史
            history['train_loss'].append(train_metrics['total_loss'])
            history['ce_loss'].append(train_metrics['ce_loss'])
            history['kl_loss'].append(train_metrics['kl_loss'])
            history['test_loss'].append(test_metrics['loss'])
            history['test_acc'].append(test_metrics['accuracy'])
            history['test_f1'].append(test_metrics['f1'])

            # 记录最佳
            if test_metrics['accuracy'] > best_acc:
                best_acc = test_metrics['accuracy']
                best_epoch = epoch
                self._save_checkpoint(output_path / 'best_model.pth', epoch)

            # 美观日志输出
            stage = "预训练" if epoch < self.pre_train_epochs else "域适应"
            stage_color = "\033[93m" if epoch < self.pre_train_epochs else "\033[96m"  # 黄色/青色
            reset = "\033[0m"

            # 简洁格式（每轮）
            if (epoch + 1) % 5 == 0 or epoch < 5 or epoch >= n_epochs - 5:
                logger.info(
                    f"{stage_color}[{stage}]{reset} Epoch {epoch:3d} | "
                    f"loss={train_metrics['total_loss']:.4f} | "
                    f"ce={train_metrics['ce_loss']:.4f} | kl={train_metrics['kl_loss']:.4f} | "
                    f"test_acc={test_metrics['accuracy']:.4f} | test_f1={test_metrics['f1']:.4f}"
                )

        logger.info("")
        logger.info("=" * 70)
        logger.info(f"训练完成! 最佳准确率: {best_acc:.4f} (Epoch {best_epoch})")
        logger.info("=" * 70)
        return history

    def plot_training_curves(self, history: Dict[str, list], save_path: str):
        """
        绘制训练曲线

        Args:
            history: 训练历史字典
            save_path: 保存路径
        """
        try:
            import matplotlib.pyplot as plt
            matplotlib_logger = logging.getLogger('matplotlib')
            matplotlib_logger.setLevel(logging.WARNING)
        except ImportError:
            logger.warning("matplotlib 未安装，跳过绘图")
            return

        n_epochs = len(history['train_loss'])
        pre_train_boundary = self.pre_train_epochs

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'CADT Training History - Target: {self.target_domain}', fontsize=14)

        # 1. 训练损失
        ax = axes[0, 0]
        ax.plot(history['train_loss'], label='Total Loss', linewidth=1.5)
        ax.plot(history['ce_loss'], label='CE Loss', linewidth=1.5, alpha=0.7)
        ax.plot(history['kl_loss'], label='KL Loss', linewidth=1.5, alpha=0.7)
        ax.axvline(x=pre_train_boundary, color='r', linestyle='--', alpha=0.5, label='Phase Switch')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. 测试准确率和F1
        ax = axes[0, 1]
        ax.plot(history['test_acc'], label='Accuracy', linewidth=1.5, color='green')
        ax.plot(history['test_f1'], label='F1 Score', linewidth=1.5, color='orange', alpha=0.8)
        ax.axvline(x=pre_train_boundary, color='r', linestyle='--', alpha=0.5, label='Phase Switch')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_title('Test Metrics')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. 测试损失
        ax = axes[1, 0]
        ax.plot(history['test_loss'], linewidth=1.5, color='red')
        ax.axvline(x=pre_train_boundary, color='r', linestyle='--', alpha=0.5, label='Phase Switch')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Test Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. 详细指标（最后50轮放大）
        ax = axes[1, 1]
        start_idx = max(0, n_epochs - 50)
        ax.plot(range(start_idx, n_epochs), history['test_acc'][start_idx:],
                label='Accuracy', linewidth=1.5, color='green')
        ax.plot(range(start_idx, n_epochs), history['test_f1'][start_idx:],
                label='F1 Score', linewidth=1.5, color='orange', alpha=0.8)
        if pre_train_boundary >= start_idx:
            ax.axvline(x=pre_train_boundary, color='r', linestyle='--', alpha=0.5, label='Phase Switch')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_title(f'Test Metrics (Last {n_epochs - start_idx} Epochs)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"训练曲线已保存: {save_path}")

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
