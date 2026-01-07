# -*- coding: utf-8 -*-
"""
注意力机制模块

提供用于层级聚合的注意力机制。
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """
    注意力聚合模块

    学习每个元素的重要性权重，进行加权聚合。

    支持：
    - 变长序列（通过掩码）
    - 返回注意力权重（用于可解释性分析）

    公式：
    score = W2 * tanh(W1 * h)
    alpha = softmax(score, mask)
    output = sum(alpha * h)
    """

    def __init__(
        self,
        input_dim: int,
        attention_dim: int = 32,
        dropout: float = 0.1,
    ):
        """
        初始化

        Args:
            input_dim: 输入特征维度
            attention_dim: 注意力隐藏层维度
            dropout: Dropout比例
        """
        super().__init__()

        self.W1 = nn.Linear(input_dim, attention_dim, bias=True)
        self.W2 = nn.Linear(attention_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: (batch, seq_len, input_dim) 输入序列
            mask: (batch, seq_len) 有效位置掩码（True表示有效）

        Returns:
            output: (batch, input_dim) 聚合后的表示
            attention_weights: (batch, seq_len) 注意力权重
        """
        # 计算注意力分数
        # (batch, seq_len, attention_dim)
        hidden = torch.tanh(self.W1(x))
        hidden = self.dropout(hidden)
        # (batch, seq_len, 1) -> (batch, seq_len)
        scores = self.W2(hidden).squeeze(-1)

        # 应用掩码
        if mask is not None:
            # 将无效位置的分数设为负无穷
            scores = scores.masked_fill(~mask, float('-inf'))

        # Softmax得到注意力权重
        attention_weights = F.softmax(scores, dim=-1)

        # 处理全为-inf的情况（整行都被掩码）
        attention_weights = attention_weights.masked_fill(
            torch.isnan(attention_weights), 0.0
        )

        # 加权聚合
        # (batch, seq_len, 1) * (batch, seq_len, input_dim) -> sum -> (batch, input_dim)
        output = torch.sum(attention_weights.unsqueeze(-1) * x, dim=1)

        return output, attention_weights


class MultiHeadAttentionPooling(nn.Module):
    """
    多头注意力聚合模块

    使用多个注意力头进行聚合，然后拼接/平均。
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 4,
        attention_dim: int = 32,
        dropout: float = 0.1,
        output_mode: str = 'concat',  # 'concat' 或 'mean'
    ):
        """
        初始化

        Args:
            input_dim: 输入特征维度
            num_heads: 注意力头数
            attention_dim: 每个头的注意力隐藏层维度
            dropout: Dropout比例
            output_mode: 输出模式，'concat'拼接或'mean'平均
        """
        super().__init__()

        self.num_heads = num_heads
        self.output_mode = output_mode

        # 每个头独立的注意力
        self.heads = nn.ModuleList([
            AttentionPooling(input_dim, attention_dim, dropout)
            for _ in range(num_heads)
        ])

        if output_mode == 'concat':
            self.output_proj = nn.Linear(input_dim * num_heads, input_dim)
        else:
            self.output_proj = None

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: (batch, seq_len, input_dim) 输入序列
            mask: (batch, seq_len) 有效位置掩码

        Returns:
            output: (batch, input_dim) 聚合后的表示
            attention_weights: (batch, num_heads, seq_len) 多头注意力权重
        """
        outputs = []
        all_weights = []

        for head in self.heads:
            out, weights = head(x, mask)
            outputs.append(out)
            all_weights.append(weights)

        # 拼接注意力权重
        attention_weights = torch.stack(all_weights, dim=1)  # (batch, num_heads, seq_len)

        if self.output_mode == 'concat':
            # 拼接并投影
            output = torch.cat(outputs, dim=-1)  # (batch, input_dim * num_heads)
            output = self.output_proj(output)  # (batch, input_dim)
        else:
            # 平均
            output = torch.stack(outputs, dim=0).mean(dim=0)  # (batch, input_dim)

        return output, attention_weights


class PositionalEncoding(nn.Module):
    """
    位置编码模块

    使用正弦/余弦位置编码，支持变长序列。
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
    ):
        """
        初始化

        Args:
            d_model: 模型维度
            max_len: 最大序列长度
            dropout: Dropout比例
        """
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: (batch, seq_len, d_model) 输入序列

        Returns:
            (batch, seq_len, d_model) 添加位置编码后的序列
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    可学习的位置编码模块
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
    ):
        """
        初始化

        Args:
            d_model: 模型维度
            max_len: 最大序列长度
            dropout: Dropout比例
        """
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: (batch, seq_len, d_model) 输入序列

        Returns:
            (batch, seq_len, d_model) 添加位置编码后的序列
        """
        batch_size, seq_len, _ = x.size()
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.pe(positions)
        return self.dropout(x)
