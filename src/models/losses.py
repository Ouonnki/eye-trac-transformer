# -*- coding: utf-8 -*-
"""
损失函数模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Reference:
        Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
        Focal loss for dense object detection. ICCV.
    
    Args:
        alpha: 类别权重，可以是标量或每个类别的权重列表
        gamma: 聚焦参数，gamma越大，对易分类样本的权重降低越多
        reduction: 损失的归约方式 ('mean', 'sum', 'none')
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: 模型输出的logits，形状为 (N, C) 或 (N, C, ...)
            targets: 目标类别索引，形状为 (N,) 或 (N, ...)
        
        Returns:
            计算的 focal loss
        """
        # 计算交叉熵损失 (不reduce)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 计算 pt (模型对正确类别的预测概率)
        pt = torch.exp(-ce_loss)
        
        # 计算 focal weight: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma
        
        # 应用类别权重 alpha
        if self.alpha is not None:
            # 将 alpha 移动到正确的设备
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            
            # 获取每个样本对应的 alpha 值
            if self.alpha.dim() == 1:
                # alpha 是类别权重列表
                at = self.alpha[targets]
            else:
                # alpha 是标量
                at = self.alpha
            focal_weight = at * focal_weight
        
        # 计算最终的 focal loss
        loss = focal_weight * ce_loss
        
        # 应用归约
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class WeightedFocalLoss(nn.Module):
    """
    带类别权重的 Focal Loss (兼容 CrossEntropyLoss 的 weight 参数)
    
    与标准 FocalLoss 的区别是，此版本允许传入 weight 参数来兼容
    原有的类别权重设置。
    """
    
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        if weight is not None:
            self.register_buffer('weight', weight)
        else:
            self.weight = None
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 使用带有 weight 的交叉熵计算
        ce_loss = F.cross_entropy(
            inputs, targets, 
            weight=self.weight,
            reduction='none'
        )
        
        # 计算 pt
        # 注意：这里需要计算无权重的概率，否则 focal weight 计算会受影响
        with torch.no_grad():
            logits_softmax = F.softmax(inputs, dim=1)
            # 获取正确类别的概率
            pt = logits_softmax.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # 计算 focal weight
        focal_weight = (1 - pt) ** self.gamma
        
        # 计算最终损失
        loss = focal_weight * ce_loss
        
        # 应用归约
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
