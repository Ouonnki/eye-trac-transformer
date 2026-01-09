# -*- coding: utf-8 -*-
"""
域迁移组件模块

提供 CADT 域迁移方法的核心组件，用于分类模式下的跨域泛化。
参考：D:\Ouonnki\cadt\cadt\model\CADT.py
"""

import torch
import torch.nn as nn
from torch.autograd import Function
from typing import Optional


class GradientReversalFunction(Function):
    """
    梯度反转函数

    前向传播：恒等变换
    反向传播：梯度乘以 -alpha
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    """
    梯度反转层

    用于对抗训练，使编码器学习域不变特征。
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.alpha)

    def set_alpha(self, alpha: float):
        """设置梯度反转强度"""
        self.alpha = alpha


class ClassCenterBank(nn.Module):
    """
    类中心库

    为每个注意力等级类别维护一个可学习的中心向量。
    对应 CADT 的 self.c（类中心矩阵）。

    用途：
    - 分类模式下，3个中心对应注意力等级 0/1/2（低/中/高）
    - 通过对齐源域样本到类中心，增强类间分离
    """

    def __init__(self, num_classes: int = 3, feature_dim: int = 128):
        """
        初始化

        Args:
            num_classes: 类别数（分类模式下为3）
            feature_dim: 特征维度（应与 subject_repr 维度一致）
        """
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        # 可学习的类中心，初始化为零（后续通过 init_centers 设置）
        self.centers = nn.Parameter(torch.zeros(num_classes, feature_dim))
        self._initialized = False

    def init_centers(self, centers: torch.Tensor):
        """
        初始化类中心

        在预训练阶段结束后，使用源域样本的平均特征初始化。

        Args:
            centers: (num_classes, feature_dim) 每个类别的平均特征
        """
        with torch.no_grad():
            self.centers.copy_(centers)
        self._initialized = True

    def get_center(self, class_ids: torch.Tensor) -> torch.Tensor:
        """
        获取指定类别的中心

        Args:
            class_ids: (batch,) 类别ID

        Returns:
            (batch, feature_dim) 对应的中心向量
        """
        return self.centers[class_ids]

    def compute_distance(
        self,
        features: torch.Tensor,
        class_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算特征到对应类中心的距离

        Args:
            features: (batch, feature_dim) 样本特征
            class_ids: (batch,) 类别ID

        Returns:
            (batch, 1) 距离值（归一化）
        """
        centers = self.get_center(class_ids)  # (batch, feature_dim)
        distances = torch.norm(features - centers.detach(), dim=-1, keepdim=True)
        distances = distances / self.feature_dim  # 归一化
        return distances

    def compute_min_distance(self, features: torch.Tensor) -> torch.Tensor:
        """
        计算特征到所有中心的最小距离（用于目标域）

        Args:
            features: (batch, feature_dim) 样本特征

        Returns:
            (batch, 1) 最小距离值（归一化）
        """
        # features: (batch, dim) -> (batch, 1, dim)
        # centers: (num_classes, dim) -> (1, num_classes, dim)
        features_exp = features.unsqueeze(1)
        centers_exp = self.centers.unsqueeze(0).detach()

        # 计算到每个中心的距离
        all_distances = torch.norm(features_exp - centers_exp, dim=-1)  # (batch, num_classes)

        # 取最小值
        min_distances = torch.min(all_distances, dim=-1, keepdim=True).values  # (batch, 1)
        min_distances = min_distances / self.feature_dim

        return min_distances


class DomainDiscriminator(nn.Module):
    """
    标准域判别器

    判别输入特征来自源域(0)还是目标域(1)。
    对应 CADT 的 discriminator2。

    通过 GRL 实现对抗训练：
    - 判别器学习区分域
    - 编码器通过反向梯度学习混淆判别器
    """

    def __init__(self, input_dim: int = 128, hidden_dim: int = 64):
        """
        初始化

        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        self.grl = GradientReversalLayer()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            features: (batch, input_dim) 样本特征

        Returns:
            (batch, 1) 域判别概率（0=源域，1=目标域）
        """
        reversed_features = self.grl(features)
        return self.model(reversed_features)

    def set_alpha(self, alpha: float):
        """设置梯度反转强度"""
        self.grl.set_alpha(alpha)


class DistanceAwareDomainDiscriminator(nn.Module):
    """
    距离感知域判别器

    输入包含特征和到类中心的距离。
    对应 CADT 的 discriminator。

    特点：
    - 不仅考虑特征本身，还考虑特征与类中心的距离
    - 保留类间结构信息
    """

    def __init__(self, feature_dim: int = 128, hidden_dim: int = 64):
        """
        初始化

        Args:
            feature_dim: 特征维度
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        self.grl = GradientReversalLayer()
        # 输入维度: feature_dim + 1 (距离)
        self.model = nn.Sequential(
            nn.Linear(feature_dim + 1, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        features: torch.Tensor,
        distances: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            features: (batch, feature_dim) 样本特征
            distances: (batch, 1) 到类中心的距离

        Returns:
            (batch, 1) 域判别概率
        """
        combined = torch.cat([features, distances], dim=-1)
        reversed_combined = self.grl(combined)
        return self.model(reversed_combined)

    def set_alpha(self, alpha: float):
        """设置梯度反转强度"""
        self.grl.set_alpha(alpha)


class Classifier(nn.Module):
    """
    简单分类器

    用于分类模式下的预测。
    对应 CADT 的 Classifier。
    """

    def __init__(self, input_dim: int, num_classes: int):
        """
        初始化

        Args:
            input_dim: 输入特征维度
            num_classes: 类别数
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            features: (batch, input_dim) 样本特征

        Returns:
            (batch, num_classes) 分类logits
        """
        return self.linear(features)
