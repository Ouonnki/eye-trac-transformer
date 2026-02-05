# -*- coding: utf-8 -*-
"""
深度学习训练器模块

提供用于层级Transformer网络的训练、验证和评估功能。
支持训练曲线可视化和增强的指标显示。
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

# 初始化 colorama（Windows 兼容）
init(autoreset=True)

# matplotlib 可选导入（用于训练曲线绘制）
try:
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from src.models.dl_models import HierarchicalTransformerNetwork
from src.models.segment_model import SegmentEncoder
from src.models.dl_dataset import HierarchicalGazeDataset, collate_fn, SequenceConfig
from src.config import UnifiedConfig

logger = logging.getLogger(__name__)


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
        self.best_epoch = 0  # 记录最佳epoch

    def __call__(self, score: float, epoch: int = 0) -> bool:
        """
        检查是否应该早停

        Args:
            score: 当前分数
            epoch: 当前epoch

        Returns:
            是否应该早停
        """
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

    直接使用 UnifiedConfig 进行配置。
    """

    def __init__(self, config: UnifiedConfig, seq_config: SequenceConfig = None):
        """
        初始化

        Args:
            config: 统一配置对象
            seq_config: 序列配置对象（如果为None，则从config.sequence自动创建）
        """
        if seq_config is None:
            seq_config = config.to_seq_config()

        self.config = config
        self.seq_config = seq_config
        self.device = torch.device(config.device.device)

        # 创建输出目录
        os.makedirs(config.experiment.output_dir, exist_ok=True)

        # 检查GPU数量
        self.n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if self.n_gpus > 1 and config.device.use_multi_gpu:
            logger.info(f'Using {self.n_gpus} GPUs for training')
        elif self.n_gpus == 1:
            logger.info(f'Using single GPU: {torch.cuda.get_device_name(0)}')
        else:
            logger.info('Using CPU for training')

        # 混合精度训练
        self.use_amp = config.device.use_amp and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        if self.use_amp:
            logger.info('Using Automatic Mixed Precision (AMP)')

        # 模型和优化器（在train时初始化）
        self.model = None
        self.optimizer = None
        self.scheduler = None

        # 蒸馏教师模型（可选）
        self.teacher_model = None

        # 训练历史（根据任务类型初始化）
        self._init_history()

    def _init_history(self) -> None:
        """根据任务类型初始化训练历史"""
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

    def _compute_class_weights(self, dataset) -> torch.Tensor:
        """计算类别权重（频率倒数）用于处理类别不平衡"""
        labels = [d['category'] - 1 for d in dataset.data]  # 0-indexed
        class_counts = np.bincount(labels, minlength=self.config.task.num_classes)
        weights = 1.0 / (class_counts + 1e-6)
        weights = weights / weights.sum() * self.config.task.num_classes
        logger.info(f'类别权重: {weights}')
        return torch.tensor(weights, dtype=torch.float32, device=self.device)

    def _create_model(self) -> nn.Module:
        """创建模型（使用 from_config 方法）"""
        # 确定输出类别数
        num_classes = self.config.task.num_classes if self.config.task.type == 'classification' else 1

        # 使用 from_config 创建模型
        model = HierarchicalTransformerNetwork.from_config(
            config=self.config,
            seq_config=self.seq_config,
            num_classes=num_classes,
        )
        model = model.to(self.device)

        if self.config.device.use_gradient_checkpointing:
            logger.info('Using Gradient Checkpointing to save memory')

        # 多GPU并行
        if self.n_gpus > 1 and self.config.device.use_multi_gpu:
            model = nn.DataParallel(model)
            logger.info(f'Model wrapped with DataParallel on {self.n_gpus} GPUs')

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

    def _load_teacher_model(self) -> None:
        """加载蒸馏教师模型（片段模型）"""
        if not self.config.distill.enable:
            return

        if not self.config.distill.teacher_model_path or not self.config.distill.teacher_config_path:
            raise ValueError('蒸馏开启但未提供 teacher_model_path 或 teacher_config_path')

        logger.info(
            f'蒸馏开启：加载教师模型 {self.config.distill.teacher_model_path} '
            f'与配置 {self.config.distill.teacher_config_path}'
        )
        teacher_config = UnifiedConfig.from_json(self.config.distill.teacher_config_path)
        num_classes = teacher_config.task.num_classes if teacher_config.task.type == 'classification' else 1

        teacher_model = SegmentEncoder.from_config(
            config=teacher_config,
            seq_config=self.seq_config,
            num_classes=num_classes,
            use_task_embedding=getattr(teacher_config.model, 'use_task_embedding', False),
        )
        teacher_model = teacher_model.to(self.device)
        teacher_model.eval()

        checkpoint = torch.load(self.config.distill.teacher_model_path, map_location=self.device, weights_only=False)
        teacher_model.load_state_dict(checkpoint['model_state_dict'])

        logger.info('教师模型加载完成，蒸馏已启用')

        self.teacher_model = teacher_model

    def plot_training_curves(self, save_path: Optional[str] = None, fold: int = 0) -> Optional[str]:
        """
        绘制训练曲线（2×2 子图布局）

        包含：
        - Loss 曲线 (train/val)
        - 分类：Accuracy/F1 曲线；回归：R²/MAE 曲线
        - Learning Rate 曲线

        Args:
            save_path: 保存路径，None 则使用默认路径
            fold: 当前折数

        Returns:
            保存的图片路径，如果失败则返回 None
        """
        if not HAS_MATPLOTLIB:
            logger.warning('matplotlib 未安装，无法生成训练曲线图')
            return None

        if not self.history['train_loss']:
            logger.warning('训练历史为空，无法生成图表')
            return None

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        # 创建 2×2 子图
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        task_type_label = '分类' if self.config.task.type == 'classification' else '回归'
        fig.suptitle(f'训练曲线 (Fold {fold + 1}) - {task_type_label}任务', fontsize=14, fontweight='bold')

        epochs = range(1, len(self.history['train_loss']) + 1)

        # 子图1: Loss 曲线
        ax1 = axes[0, 0]
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        loss_label = 'Loss (CE)' if self.config.task.type == 'classification' else 'Loss (MSE)'
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
            ax2.plot(epochs, self.history['val_accuracy'], 'g-', label='Val Accuracy', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Accuracy 曲线')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # 标记最佳 Accuracy
            best_acc_epoch = np.argmax(self.history['val_accuracy']) + 1
            best_acc = max(self.history['val_accuracy'])
            ax2.axvline(x=best_acc_epoch, color='b', linestyle='--', alpha=0.7)
            ax2.scatter([best_acc_epoch], [best_acc], color='b', s=100, zorder=5)
            ax2.annotate(f'Best: {best_acc:.4f}', xy=(best_acc_epoch, best_acc),
                         xytext=(10, -10), textcoords='offset points', fontsize=9)

            # 子图3: F1 曲线
            ax3 = axes[1, 0]
            ax3.plot(epochs, self.history['val_f1'], 'm-', label='Val F1 (macro)', linewidth=2)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('F1 Score')
            ax3.set_title('F1 曲线')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # 标记最佳 F1
            best_f1_epoch = np.argmax(self.history['val_f1']) + 1
            best_f1 = max(self.history['val_f1'])
            ax3.axvline(x=best_f1_epoch, color='c', linestyle='--', alpha=0.7)
            ax3.scatter([best_f1_epoch], [best_f1], color='c', s=100, zorder=5)
            ax3.annotate(f'Best: {best_f1:.4f}', xy=(best_f1_epoch, best_f1),
                         xytext=(10, 10), textcoords='offset points', fontsize=9)
        else:
            # 子图2: R² 曲线
            ax2 = axes[0, 1]
            ax2.plot(epochs, self.history['val_r2'], 'g-', label='Val R2', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('R2')
            ax2.set_title('R2 曲线')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # 标记最佳 R²
            best_r2_epoch = np.argmax(self.history['val_r2']) + 1
            best_r2 = max(self.history['val_r2'])
            ax2.axvline(x=best_r2_epoch, color='b', linestyle='--', alpha=0.7)
            ax2.scatter([best_r2_epoch], [best_r2], color='b', s=100, zorder=5)
            ax2.annotate(f'Best: {best_r2:.4f}', xy=(best_r2_epoch, best_r2),
                         xytext=(10, -10), textcoords='offset points', fontsize=9)

            # 子图3: MAE 曲线
            ax3 = axes[1, 0]
            ax3.plot(epochs, self.history['val_mae'], 'm-', label='Val MAE', linewidth=2)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('MAE')
            ax3.set_title('MAE 曲线')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # 标记最佳 MAE
            best_mae_epoch = np.argmin(self.history['val_mae']) + 1
            best_mae = min(self.history['val_mae'])
            ax3.axvline(x=best_mae_epoch, color='c', linestyle='--', alpha=0.7)
            ax3.scatter([best_mae_epoch], [best_mae], color='c', s=100, zorder=5)
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

    def _format_metric_change(self, current: float, initial: float, higher_is_better: bool = False) -> str:
        """
        格式化指标变化（带颜色和箭头）

        Args:
            current: 当前值
            initial: 初始值
            higher_is_better: 越大是否越好

        Returns:
            格式化的变化字符串
        """
        if initial == 0:
            change_pct = 0.0
        else:
            change_pct = ((current - initial) / abs(initial)) * 100

        if higher_is_better:
            arrow = Fore.GREEN + '↑' if change_pct > 0 else Fore.RED + '↓'
            color = Fore.GREEN if change_pct > 0 else Fore.RED
        else:
            arrow = Fore.GREEN + '↓' if change_pct < 0 else Fore.RED + '↑'
            color = Fore.GREEN if change_pct < 0 else Fore.RED

        change_str = f"{color}{arrow} {abs(change_pct):.1f}%{Style.RESET_ALL}"
        return change_str

    def _print_epoch_summary(
        self,
        epoch: int,
        initial_train_loss: float,
        initial_val_loss: float,
        initial_val_metric: float,
        best_val_loss: float,
        best_epoch: int,
    ) -> None:
        """
        打印阶段性汇总（美化的表格格式）

        Args:
            epoch: 当前 epoch
            initial_train_loss: 初始训练损失
            initial_val_loss: 初始验证损失
            initial_val_metric: 初始验证指标（分类=accuracy，回归=R²）
            best_val_loss: 最佳验证损失
            best_epoch: 最佳 epoch
        """
        current_train_loss = self.history['train_loss'][-1]
        current_val_loss = self.history['val_loss'][-1]
        current_lr = self.history['learning_rate'][-1]

        # 打印表头
        print(f'\n{Fore.CYAN}{'═' * 70}{Style.RESET_ALL}')
        print(f'{Fore.CYAN}║{Style.RESET_ALL}  {Fore.YELLOW}Epoch {epoch:3d} 训练汇总{Style.RESET_ALL}  '
              f'{Fore.CYAN}║{Style.RESET_ALL}')
        print(f'{Fore.CYAN}{'═' * 70}{Style.RESET_ALL}')

        # 训练损失
        train_change = self._format_metric_change(current_train_loss, initial_train_loss, higher_is_better=False)
        print(f'  训练损失: {initial_train_loss:.4f} → {current_train_loss:.4f} ({train_change})')

        # 验证损失
        val_change = self._format_metric_change(current_val_loss, initial_val_loss, higher_is_better=False)
        print(f'  验证损失: {initial_val_loss:.4f} → {current_val_loss:.4f} ({val_change})')

        # 任务指标
        if self.config.task.type == 'classification':
            current_val_acc = self.history['val_accuracy'][-1]
            acc_change = self._format_metric_change(current_val_acc, initial_val_metric, higher_is_better=True)
            print(f'  验证精度: {initial_val_metric:.4f} → {current_val_acc:.4f} ({acc_change})')

            current_val_f1 = self.history['val_f1'][-1]
            print(f'  F1 分数:  {Fore.CYAN}{current_val_f1:.4f}{Style.RESET_ALL}')
        else:
            current_val_r2 = self.history['val_r2'][-1]
            r2_change = self._format_metric_change(current_val_r2, initial_val_metric, higher_is_better=True)
            print(f'  验证 R²:  {initial_val_metric:.4f} → {current_val_r2:.4f} ({r2_change})')

            current_val_mae = self.history['val_mae'][-1]
            print(f'  验证 MAE: {Fore.CYAN}{current_val_mae:.4f}{Style.RESET_ALL}')

        # 学习率
        print(f'  学习率:   {Fore.MAGENTA}{current_lr:.2e}{Style.RESET_ALL}')

        # 最佳模型信息
        if best_val_loss == current_val_loss:
            best_mark = f' {Fore.GREEN}★ 当前最佳{Style.RESET_ALL}'
        else:
            best_mark = ''
        print(f'  最佳模型: {Fore.GREEN}{best_val_loss:.4f}{Style.RESET_ALL} @ Epoch {best_epoch}{best_mark}')

        print(f'{Fore.CYAN}{'═' * 70}{Style.RESET_ALL}\n')

    def _print_training_start(self, fold: int, total_epochs: int, train_samples: int, val_samples: int) -> None:
        """打印训练开始信息"""
        task_label = f'{Fore.YELLOW}分类{Style.RESET_ALL}' if self.config.task.type == 'classification' else f'{Fore.YELLOW}回归{Style.RESET_ALL}'
        print(f'\n{Fore.CYAN}╔{'═' * 68}╗{Style.RESET_ALL}')
        print(f'{Fore.CYAN}║{Style.RESET_ALL}  {Fore.WHITE}开始训练 Fold {fold + 1} - {task_label}任务{Style.RESET_ALL}  '
              f'{Fore.CYAN}║{Style.RESET_ALL}')
        print(f'{Fore.CYAN}╠{'═' * 68}╣{Style.RESET_ALL}')
        print(f'{Fore.CYAN}║{Style.RESET_ALL}  训练样本: {train_samples:4d}  '
              f'验证样本: {val_samples:4d}  '
              f'总轮数: {total_epochs:3d}  {Fore.CYAN}║{Style.RESET_ALL}')
        print(f'{Fore.CYAN}╚{'═' * 68}╝{Style.RESET_ALL}\n')

    def _print_early_stop(self, epoch: int, patience: int, best_epoch: int, best_score: float) -> None:
        """打印早停信息"""
        print(f'\n{Fore.YELLOW}{'─' * 50}{Style.RESET_ALL}')
        print(f'  {Fore.YELLOW}⚠ 早停触发 @ Epoch {epoch}{Style.RESET_ALL} (patience={patience})')
        print(f'  {Fore.GREEN}✓ 最佳模型 @ Epoch {best_epoch} | val_loss={best_score:.4f}{Style.RESET_ALL}')
        print(f'{Fore.YELLOW}{'─' * 50}{Style.RESET_ALL}\n')

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
            task_conditions = batch['task_conditions'].to(self.device, non_blocking=True) if 'task_conditions' in batch else None

            optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零

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
                    loss = criterion(outputs['prediction'], labels)
                    if self.config.distill.enable and self.teacher_model is not None:
                        kd_loss = self._compute_kd_loss(
                            outputs['prediction'],
                            segments,
                            segment_lengths,
                            segment_mask,
                            task_conditions,
                        )
                        loss = (1 - self.config.distill.alpha) * loss + self.config.distill.alpha * kd_loss

                # 混合精度反向传播
                self.scaler.scale(loss).backward()

                # 梯度裁剪
                if self.config.training.grad_clip > 0:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.training.grad_clip)

                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                # 标准前向传播
                outputs = model(
                    segments=segments,
                    segment_mask=segment_mask,
                    task_mask=task_mask,
                    segment_lengths=segment_lengths,
                    task_conditions=task_conditions,
                )
                loss = criterion(outputs['prediction'], labels)
                if self.config.distill.enable and self.teacher_model is not None:
                    kd_loss = self._compute_kd_loss(
                        outputs['prediction'],
                        segments,
                        segment_lengths,
                        segment_mask,
                        task_conditions,
                    )
                    loss = (1 - self.config.distill.alpha) * loss + self.config.distill.alpha * kd_loss

                # 反向传播
                loss.backward()

                # 梯度裁剪
                if self.config.training.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.training.grad_clip)

                optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def _compute_kd_loss(
        self,
        student_logits: torch.Tensor,
        segments: torch.Tensor,
        segment_lengths: torch.Tensor,
        segment_mask: torch.Tensor,
        task_conditions: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """计算蒸馏损失（片段教师 -> 被试级学生）"""
        # 仅分类任务支持蒸馏
        if self.config.task.type != 'classification':
            return torch.tensor(0.0, device=student_logits.device)

        if self.teacher_model is None:
            return torch.tensor(0.0, device=student_logits.device)

        batch_size, max_tasks, max_segments, max_seq_len, feat_dim = segments.shape

        flat_segments = segments.view(-1, max_seq_len, feat_dim)
        flat_lengths = segment_lengths.view(-1)
        flat_mask = segment_mask.view(-1)

        valid = flat_mask & (flat_lengths > 0)
        if not torch.any(valid):
            return torch.tensor(0.0, device=student_logits.device)

        seg_inputs = flat_segments[valid]
        seg_lengths = flat_lengths[valid]

        teacher_task_conditions = None
        if task_conditions is not None:
            expanded = task_conditions.unsqueeze(2).expand(-1, -1, max_segments, -1)
            flat_tc = expanded.reshape(batch_size * max_tasks * max_segments, -1)
            flat_tc = flat_tc[valid]
            teacher_task_conditions = {
                'grid_scale': flat_tc[:, 0],
                'continuous_thinking': flat_tc[:, 1],
                'click_disappear': flat_tc[:, 2],
                'has_distractor': flat_tc[:, 3],
                'has_task_distractor': flat_tc[:, 4],
            }

        with torch.no_grad():
            teacher_logits = self.teacher_model(seg_inputs, seg_lengths, task_conditions=teacher_task_conditions)

        t = self.config.distill.temperature
        teacher_probs = torch.softmax(teacher_logits / t, dim=-1)

        # 聚合到被试级（按平均）
        # 生成 valid 索引的被试 id
        seg_indices = torch.nonzero(valid, as_tuple=False).squeeze(1)
        subj_ids = (seg_indices // (max_tasks * max_segments)).long()

        num_classes = teacher_probs.size(-1)
        agg = torch.zeros((batch_size, num_classes), device=teacher_probs.device)
        counts = torch.zeros((batch_size, 1), device=teacher_probs.device)

        agg.index_add_(0, subj_ids, teacher_probs)
        counts.index_add_(0, subj_ids, torch.ones_like(subj_ids, dtype=agg.dtype).unsqueeze(1))

        teacher_subj_probs = agg / counts.clamp_min(1.0)

        # KLDivLoss expects log-probabilities for input
        log_student = torch.log_softmax(student_logits / t, dim=-1)
        kd_loss = nn.KLDivLoss(reduction='batchmean')(log_student, teacher_subj_probs) * (t ** 2)
        return kd_loss

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
                task_conditions = batch['task_conditions'].to(self.device, non_blocking=True) if 'task_conditions' in batch else None

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
                        loss = criterion(outputs['prediction'], labels)
                else:
                    outputs = model(
                        segments=segments,
                        segment_mask=segment_mask,
                        task_mask=task_mask,
                        segment_lengths=segment_lengths,
                        task_conditions=task_conditions,
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

        if self.config.task.type == 'classification':
            # 分类任务指标
            preds = np.argmax(all_predictions, axis=1)
            metrics = {
                'loss': total_loss / num_batches,
                'accuracy': accuracy_score(all_labels, preds),
                'f1': f1_score(all_labels, preds, average='macro'),
            }
        else:
            # 回归任务指标
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

        # 加载教师模型（如启用蒸馏）
        if self.config.distill.enable:
            self._load_teacher_model()

        # 创建数据加载器（支持多进程和锁页内存）
        loader_kwargs = {
            'batch_size': self.config.training.batch_size,
            'collate_fn': collate_fn,
            'num_workers': self.config.device.num_workers,
            'pin_memory': self.config.device.pin_memory and torch.cuda.is_available(),
            'persistent_workers': self.config.device.num_workers > 0,
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

        logger.info(f'DataLoader: batch_size={self.config.training.batch_size}, '
                    f'num_workers={self.config.device.num_workers}, '
                    f'pin_memory={loader_kwargs["pin_memory"]}')

        # 损失函数（根据任务类型选择）
        if self.config.task.type == 'classification':
            if self.config.task.use_class_weights:
                class_weights = self._compute_class_weights(train_dataset)
                criterion = nn.CrossEntropyLoss(
                    weight=class_weights,
                    label_smoothing=self.config.training.label_smoothing,
                )
            else:
                criterion = nn.CrossEntropyLoss(
                    label_smoothing=self.config.training.label_smoothing,
                )
            logger.info(f'使用分类任务，类别数: {self.config.task.num_classes}')
        else:
            criterion = nn.MSELoss()
            logger.info('使用回归任务')

        # 早停
        early_stopping = EarlyStopping(patience=self.config.training.patience, mode='min')

        # 最佳模型
        best_metrics = None
        best_model_state = None

        # 重置历史（根据任务类型）
        self._init_history()

        # 打印训练开始信息
        self._print_training_start(fold, self.config.training.epochs, len(train_dataset), len(val_dataset))

        # 训练循环
        pbar = tqdm(
            range(self.config.training.epochs),
            desc=f'{Fore.CYAN}训练中{Style.RESET_ALL}',
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
        )

        # 记录初始指标用于阶段性汇总
        initial_train_loss = None
        initial_val_loss = None
        initial_val_metric = None  # 分类用accuracy，回归用r2

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

            # 记录历史（根据任务类型）
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['learning_rate'].append(current_lr)

            if self.config.task.type == 'classification':
                self.history['val_accuracy'].append(val_metrics['accuracy'])
                self.history['val_f1'].append(val_metrics['f1'])
            else:
                self.history['val_r2'].append(val_metrics['r2'])
                self.history['val_mae'].append(val_metrics['mae'])

            # 记录初始值
            if epoch == 0:
                initial_train_loss = train_loss
                initial_val_loss = val_metrics['loss']
                initial_val_metric = val_metrics.get('accuracy', val_metrics.get('r2'))

            # 更新进度条（根据任务类型显示不同指标）
            if self.config.task.type == 'classification':
                postfix_dict = {
                    'loss': f'{Fore.BLUE}{train_loss:.3f}{Style.RESET_ALL}',
                    'v_loss': f'{Fore.RED}{val_metrics["loss"]:.3f}{Style.RESET_ALL}',
                    'Acc': f'{Fore.GREEN}{val_metrics["accuracy"]:.3f}{Style.RESET_ALL}',
                    'F1': f'{Fore.CYAN}{val_metrics["f1"]:.3f}{Style.RESET_ALL}',
                }
            else:
                postfix_dict = {
                    'loss': f'{Fore.BLUE}{train_loss:.3f}{Style.RESET_ALL}',
                    'v_loss': f'{Fore.RED}{val_metrics["loss"]:.3f}{Style.RESET_ALL}',
                    'R2': f'{Fore.GREEN}{val_metrics["r2"]:.3f}{Style.RESET_ALL}',
                    'MAE': f'{Fore.CYAN}{val_metrics["mae"]:.2f}{Style.RESET_ALL}',
                }
            pbar.set_postfix(postfix_dict)

            # 保存最佳模型（处理DataParallel包装）
            if best_metrics is None or val_metrics['loss'] < best_metrics['loss']:
                best_metrics = val_metrics.copy()
                best_metrics['epoch'] = epoch + 1
                # 获取原始模型状态（去除DataParallel包装）
                if isinstance(self.model, nn.DataParallel):
                    best_model_state = self.model.module.state_dict().copy()
                else:
                    best_model_state = self.model.state_dict().copy()

            # 阶段性汇总（每 summary_interval 个 epoch）
            if (epoch + 1) % self.config.output.summary_interval == 0 and epoch > 0:
                self._print_epoch_summary(
                    epoch=epoch + 1,
                    initial_train_loss=initial_train_loss,
                    initial_val_loss=initial_val_loss,
                    initial_val_metric=initial_val_metric,
                    best_val_loss=early_stopping.best_score if early_stopping.best_score else val_metrics['loss'],
                    best_epoch=early_stopping.best_epoch + 1,
                )

            # 早停检查
            if early_stopping(val_metrics['loss'], epoch):
                self._print_early_stop(
                    epoch=epoch + 1,
                    patience=self.config.training.patience,
                    best_epoch=early_stopping.best_epoch + 1,
                    best_score=early_stopping.best_score,
                )
                logger.info(f'Early stopping at epoch {epoch+1}')
                break

        # 恢复最佳模型（处理DataParallel）
        if best_model_state is not None:
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(best_model_state)
            else:
                self.model.load_state_dict(best_model_state)

        # 打印训练完成信息
        print(f'\n{Fore.GREEN}╔{'═' * 68}╗{Style.RESET_ALL}')
        print(f'{Fore.GREEN}║{Style.RESET_ALL}  {Fore.WHITE}Fold {fold + 1} 训练完成{Style.RESET_ALL}  '
              f'{Fore.GREEN}║{Style.RESET_ALL}')
        print(f'{Fore.GREEN}╠{'═' * 68}╣{Style.RESET_ALL}')

        if self.config.task.type == 'classification':
            print(f'{Fore.GREEN}║{Style.RESET_ALL}  最佳验证精度: {Fore.YELLOW}{best_metrics.get("accuracy", 0):.4f}{Style.RESET_ALL}  '
                  f'最佳 F1: {Fore.YELLOW}{best_metrics.get("f1", 0):.4f}{Style.RESET_ALL}  '
                  f'{Fore.GREEN}║{Style.RESET_ALL}')
        else:
            print(f'{Fore.GREEN}║{Style.RESET_ALL}  最佳 R²: {Fore.YELLOW}{best_metrics.get("r2", 0):.4f}{Style.RESET_ALL}  '
                  f'最佳 MAE: {Fore.YELLOW}{best_metrics.get("mae", 0):.4f}{Style.RESET_ALL}  '
                  f'{Fore.GREEN}║{Style.RESET_ALL}')

        print(f'{Fore.GREEN}║{Style.RESET_ALL}  最佳验证损失: {Fore.YELLOW}{best_metrics.get("loss", 0):.4f}{Style.RESET_ALL}  '
              f'@ Epoch {Fore.YELLOW}{best_metrics.get("epoch", 0)}{Style.RESET_ALL}  '
              f'{Fore.GREEN}║{Style.RESET_ALL}')
        print(f'{Fore.GREEN}╚{'═' * 68}╝{Style.RESET_ALL}\n')

        # 保存模型（保存不带DataParallel的状态）
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
                'history': self.history,  # 保存训练历史
            }, model_path)
            logger.info(f'Model saved to {model_path}')

        # 保存训练曲线图
        if self.config.output.save_figures:
            self.plot_training_curves(fold=fold)

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
            batch_size=self.config.training.batch_size,
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
                task_conditions = batch['task_conditions'].to(self.device) if 'task_conditions' in batch else None

                outputs = self.model(
                    segments=segments,
                    segment_mask=segment_mask,
                    task_mask=task_mask,
                    segment_lengths=segment_lengths,
                    task_conditions=task_conditions,
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
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model = self._create_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f'Model loaded from {model_path}')
