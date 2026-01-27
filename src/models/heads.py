# -*- coding: utf-8 -*-
"""
预测头模块

包含用于最终预测的头部网络。
"""

import torch
import torch.nn as nn


class PredictionHead(nn.Module):
    """
    统一预测头

    支持分类和回归两种模式。

    结构:
    - Linear: input_dim → hidden_dim
    - GELU: 激活函数
    - Dropout: 正则化
    - Linear: hidden_dim → output_dim
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
    ):
        """
        初始化

        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度（1=回归，>1=分类）
            dropout: Dropout 比例
        """
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: (batch, input_dim) 输入特征

        Returns:
            (batch, output_dim) 预测结果
        """
        return self.model(x)
