# -*- coding: utf-8 -*-
"""
损失函数模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def build_ordinal_targets(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    将类别标签转换为序数学习的累计二值目标。

    例：num_classes=3 时
    - y=0 -> [0, 0]
    - y=1 -> [1, 0]
    - y=2 -> [1, 1]
    """
    if num_classes < 2:
        raise ValueError("num_classes 必须 >= 2")
    if labels.dim() != 1:
        raise ValueError("labels 必须为一维张量")

    thresholds = torch.arange(num_classes - 1, device=labels.device).unsqueeze(0)
    targets = (labels.unsqueeze(1) > thresholds).to(torch.float32)
    return targets


def decode_ordinal_logits(
    logits: torch.Tensor,
    thresholds: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    将序数 logit 解码为离散类别标签。
    """
    if logits.dim() != 2:
        raise ValueError("ordinal logits 必须为二维张量 (B, K-1)")

    probs = torch.sigmoid(logits)
    if thresholds is None:
        thresholds = torch.full(
            (logits.size(1),),
            0.5,
            dtype=probs.dtype,
            device=probs.device,
        )
    else:
        thresholds = thresholds.to(device=probs.device, dtype=probs.dtype)
        if thresholds.dim() != 1 or thresholds.numel() != logits.size(1):
            raise ValueError("thresholds 维度必须与 logits 的第二维一致")

    preds = (probs > thresholds.unsqueeze(0)).sum(dim=1).to(torch.long)
    return preds


def ordinal_probs_to_class_probs(cum_probs: torch.Tensor) -> torch.Tensor:
    """
    将累计概率 P(y>k) 转换为每个类别的概率分布 P(y=c)。
    """
    if cum_probs.dim() != 2:
        raise ValueError("cum_probs 必须为二维张量 (B, K-1)")

    # 累计概率理论上应满足非增约束，这里做一次投影以避免数值异常。
    monotonic = cum_probs.clamp(0.0, 1.0).clone()
    for i in range(1, monotonic.size(1)):
        monotonic[:, i] = torch.minimum(monotonic[:, i - 1], monotonic[:, i])

    if monotonic.size(1) == 0:
        return torch.ones((monotonic.size(0), 1), dtype=monotonic.dtype, device=monotonic.device)

    class_probs = []
    class_probs.append(1.0 - monotonic[:, 0])
    for i in range(1, monotonic.size(1)):
        class_probs.append(monotonic[:, i - 1] - monotonic[:, i])
    class_probs.append(monotonic[:, -1])

    probs = torch.stack(class_probs, dim=1).clamp(min=0.0)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return probs


class OrdinalLoss(nn.Module):
    """
    序数学习损失（CORAL 风格 BCE）。
    """

    def __init__(self, pos_weight: Optional[torch.Tensor] = None, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        if pos_weight is not None:
            self.register_buffer('pos_weight', pos_weight)
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape != targets.shape:
            raise ValueError("OrdinalLoss 的 logits 与 targets 形状必须一致")
        if logits.dim() != 2:
            raise ValueError("OrdinalLoss 输入必须是二维张量 (B, K-1)")

        pos_weight = self.pos_weight
        if pos_weight is not None and pos_weight.device != logits.device:
            pos_weight = pos_weight.to(logits.device)

        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets.to(dtype=logits.dtype),
            pos_weight=pos_weight,
            reduction=self.reduction,
        )
        return loss


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
