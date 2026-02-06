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

from src.models.task_level_model import TaskLevelEncoder

logger = logging.getLogger(__name__)


class TaskLevelTrainer:
    """任务级模型训练器"""

    def __init__(
        self,
        model: TaskLevelEncoder,
        device: torch.device,
        num_classes: int = 3,
        class_weights: Optional[torch.Tensor] = None,
        lr: float = 1e-4,
        weight_decay: float = 0.001,
        grad_clip: float = 1.0,
    ):
        self.model = model
        self.device = device
        self.grad_clip = grad_clip
        
        # 损失函数
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in tqdm(dataloader, desc="Training"):
            self.optimizer.zero_grad()
            
            segments = batch['segments'].to(self.device)
            segment_mask = batch['segment_mask'].to(self.device)
            task_conditions = batch['task_conditions'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            logits = self.model(segments, segment_mask, task_conditions)
            loss = self.criterion(logits, labels)
            
            loss.backward()
            
            # 梯度裁剪
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """评估"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        for batch in dataloader:
            segments = batch['segments'].to(self.device)
            segment_mask = batch['segment_mask'].to(self.device)
            task_conditions = batch['task_conditions'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            logits = self.model(segments, segment_mask, task_conditions)
            loss = self.criterion(logits, labels)
            
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'predictions': all_preds,
            'labels': all_labels,
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
