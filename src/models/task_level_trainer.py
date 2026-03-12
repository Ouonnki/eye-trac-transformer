# -*- coding: utf-8 -*-
"""
任务级模型训练器
"""

import logging
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from src.models.task_level_model import TaskLevelEncoder
from src.models.losses import (
    FocalLoss,
    WeightedFocalLoss,
    OrdinalLoss,
    build_ordinal_targets,
    decode_ordinal_logits,
    ordinal_probs_to_class_probs,
)

logger = logging.getLogger(__name__)


class TaskLevelTrainer:
    """任务级模型训练器"""

    def __init__(
        self,
        model: TaskLevelEncoder,
        device: torch.device,
        num_classes: int = 3,
        head_type: str = "classification",
        class_weights: Optional[torch.Tensor] = None,
        lr: float = 1e-4,
        weight_decay: float = 0.001,
        grad_clip: float = 1.0,
        use_focal_loss: bool = False,
        focal_loss_alpha: Optional[List[float]] = None,
        focal_loss_gamma: float = 2.0,
        label_smoothing: float = 0.0,
        ordinal_thresholds: Optional[List[float]] = None,
        ordinal_pos_weight: Optional[torch.Tensor] = None,
    ):
        self.model = model
        self.device = device
        self.grad_clip = grad_clip
        self.num_classes = num_classes
        self.head_type = head_type

        if self.head_type not in {"classification", "ordinal"}:
            raise ValueError(f"不支持的 head_type: {self.head_type}")

        if self.head_type == "ordinal":
            if self.num_classes < 2:
                raise ValueError("ordinal 模式要求 num_classes >= 2")
            if ordinal_thresholds is None:
                ordinal_thresholds = [0.5] * (self.num_classes - 1)
            if len(ordinal_thresholds) != self.num_classes - 1:
                raise ValueError("ordinal_thresholds 长度必须为 num_classes - 1")
            self.ordinal_thresholds = ordinal_thresholds
        else:
            self.ordinal_thresholds = None
        
        # 损失函数
        if self.head_type == "ordinal":
            self.criterion = OrdinalLoss(
                pos_weight=ordinal_pos_weight,
                reduction='mean',
            )
            if ordinal_pos_weight is not None:
                logger.info(
                    f"使用 OrdinalLoss (thresholds={self.ordinal_thresholds}, pos_weight={ordinal_pos_weight.tolist()})"
                )
            else:
                logger.info(f"使用 OrdinalLoss (thresholds={self.ordinal_thresholds})")
        elif use_focal_loss:
            # 使用 Focal Loss（注意：FocalLoss 不支持 label_smoothing）
            if focal_loss_alpha is not None:
                alpha_tensor = torch.tensor(focal_loss_alpha, dtype=torch.float32)
            else:
                alpha_tensor = None
            
            self.criterion = FocalLoss(
                alpha=alpha_tensor,
                gamma=focal_loss_gamma,
                reduction='mean'
            )
            logger.info(f"使用 Focal Loss (gamma={focal_loss_gamma}, alpha={focal_loss_alpha})")
            if label_smoothing > 0:
                logger.warning(f"Focal Loss 不支持 label_smoothing (设置为 {label_smoothing})，已忽略")
        elif class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights.to(device),
                label_smoothing=label_smoothing
            )
            logger.info(f"使用带权重的 CrossEntropyLoss (label_smoothing={label_smoothing})")
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            logger.info(f"使用标准 CrossEntropyLoss (label_smoothing={label_smoothing})")
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )

    def _ordinal_thresholds_tensor(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.tensor(
            self.ordinal_thresholds,
            dtype=logits.dtype,
            device=logits.device,
        )

    def train_epoch(self, dataloader: DataLoader, epoch: int, total_epochs: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)
        for batch in pbar:
            self.optimizer.zero_grad()
            
            segments = batch['segments'].to(self.device)
            segment_mask = batch['segment_mask'].to(self.device)
            task_conditions = batch['task_conditions'].to(self.device)
            segment_seq_mask = batch.get('segment_seq_mask')
            if segment_seq_mask is not None:
                segment_seq_mask = segment_seq_mask.to(self.device)
            labels = batch['labels'].to(self.device)
            
            logits = self.model(
                segments,
                segment_mask,
                task_conditions,
                segment_seq_mask
            )
            if self.head_type == "ordinal":
                ordinal_targets = build_ordinal_targets(labels, self.num_classes)
                loss = self.criterion(logits, ordinal_targets)
                pred = decode_ordinal_logits(
                    logits,
                    thresholds=self._ordinal_thresholds_tensor(logits),
                )
            else:
                loss = self.criterion(logits, labels)
                pred = logits.argmax(dim=1)
            
            loss.backward()
            
            # 梯度裁剪
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 更新进度条
            acc = (np.array(all_preds) == np.array(all_labels)).mean()
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.4f}'})
        
        avg_loss = total_loss / len(dataloader)
        accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
        
        # 同时计算Weighted F1和Macro F1
        f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1': f1_weighted,           # 保持兼容
            'f1_weighted': f1_weighted,  # 显式命名
            'f1_macro': f1_macro,        # 新增
        }

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, desc: str = "Eval") -> Dict[str, float]:
        """评估"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        for batch in tqdm(dataloader, desc=desc, leave=False):
            segments = batch['segments'].to(self.device)
            segment_mask = batch['segment_mask'].to(self.device)
            task_conditions = batch['task_conditions'].to(self.device)
            segment_seq_mask = batch.get('segment_seq_mask')
            if segment_seq_mask is not None:
                segment_seq_mask = segment_seq_mask.to(self.device)
            labels = batch['labels'].to(self.device)
            
            logits = self.model(
                segments,
                segment_mask,
                task_conditions,
                segment_seq_mask
            )
            if self.head_type == "ordinal":
                ordinal_targets = build_ordinal_targets(labels, self.num_classes)
                loss = self.criterion(logits, ordinal_targets)
                pred = decode_ordinal_logits(
                    logits,
                    thresholds=self._ordinal_thresholds_tensor(logits),
                )
                cum_probs = torch.sigmoid(logits)
                probs = ordinal_probs_to_class_probs(cum_probs)
            else:
                loss = self.criterion(logits, labels)
                pred = logits.argmax(dim=1)
                probs = torch.softmax(logits, dim=1)
            
            total_loss += loss.item()
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
        
        # 同时计算Weighted F1和Macro F1
        f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1': f1_weighted,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs,
        }
    
    def save_checkpoint(self, path: str, epoch: int, best_metric: float):
        """保存检查点"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': best_metric,
        }, path)
        logger.info(f"检查点已保存: {path}")
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info(f"检查点已加载: {path}")
        return checkpoint.get('epoch', 0), checkpoint.get('best_metric', 0.0)
