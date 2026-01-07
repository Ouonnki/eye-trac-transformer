# -*- coding: utf-8 -*-
"""
深度学习模型架构模块

提供用于眼动注意力预测的层级Transformer网络。
"""

from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn

from src.models.attention import AttentionPooling, PositionalEncoding


class GazeTransformerEncoder(nn.Module):
    """
    眼动序列Transformer编码器

    将单个片段的眼动序列编码为固定维度的表示。

    结构：
    - 输入嵌入: Linear(input_dim → d_model)
    - 位置编码: 正弦位置编码
    - [CLS] token: 可学习的分类token
    - Transformer Encoder: 多层自注意力
    - 输出: [CLS] token的表示
    """

    def __init__(
        self,
        input_dim: int = 7,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 100,
    ):
        """
        初始化

        Args:
            input_dim: 输入特征维度（x, y, dt, velocity, acceleration, direction, direction_change）
            d_model: 模型维度
            nhead: 注意力头数
            num_layers: Transformer层数
            dim_feedforward: 前馈网络维度
            dropout: Dropout比例
            max_seq_len: 最大序列长度
        """
        super().__init__()

        self.d_model = d_model

        # 输入投影
        self.input_proj = nn.Linear(input_dim, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len + 1, dropout=dropout)

        # [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # LayerNorm
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: (batch, seq_len, input_dim) 眼动序列
            mask: (batch, seq_len) 有效位置掩码（True表示有效）

        Returns:
            output: (batch, d_model) 片段表示
            attention_weights: None（Transformer内部注意力权重，暂不返回）
        """
        batch_size = x.size(0)

        # 输入投影
        x = self.input_proj(x)  # (batch, seq_len, d_model)

        # 添加[CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d_model)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, seq_len+1, d_model)

        # 位置编码
        x = self.pos_encoder(x)

        # 构建attention mask
        if mask is not None:
            # 为[CLS]位置添加True
            cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=mask.device)
            full_mask = torch.cat([cls_mask, mask], dim=1)  # (batch, seq_len+1)
            # Transformer使用的是key_padding_mask，True表示忽略
            key_padding_mask = ~full_mask
        else:
            key_padding_mask = None

        # Transformer编码
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)

        # 取[CLS] token的输出
        output = self.norm(x[:, 0, :])  # (batch, d_model)

        return output, None


class TaskTransformerEncoder(nn.Module):
    """
    任务级Transformer编码器

    编码被试的任务序列，捕获任务间的依赖关系（如疲劳效应、学习效应）。
    """

    def __init__(
        self,
        input_dim: int = 64,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_tasks: int = 30,
    ):
        """
        初始化

        Args:
            input_dim: 输入特征维度（任务表示维度）
            d_model: 模型维度
            nhead: 注意力头数
            num_layers: Transformer层数
            dim_feedforward: 前馈网络维度
            dropout: Dropout比例
            max_tasks: 最大任务数
        """
        super().__init__()

        self.d_model = d_model

        # 输入投影（如果维度不同）
        if input_dim != d_model:
            self.input_proj = nn.Linear(input_dim, d_model)
        else:
            self.input_proj = nn.Identity()

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_tasks, dropout=dropout)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # LayerNorm
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            x: (batch, num_tasks, input_dim) 任务序列
            mask: (batch, num_tasks) 有效任务掩码（True表示有效）

        Returns:
            output: (batch, num_tasks, d_model) 编码后的任务序列
        """
        # 输入投影
        x = self.input_proj(x)  # (batch, num_tasks, d_model)

        # 位置编码
        x = self.pos_encoder(x)

        # 构建attention mask
        if mask is not None:
            key_padding_mask = ~mask
        else:
            key_padding_mask = None

        # Transformer编码
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)

        return x


class HierarchicalTransformerNetwork(nn.Module):
    """
    层级Transformer网络

    完整的眼动注意力预测模型。

    结构：
    1. GazeTransformerEncoder: 片段眼动序列 → 片段表示
    2. TaskAggregator: 片段表示 → 任务表示
    3. TaskTransformerEncoder: 任务序列 → 编码任务间依赖
    4. SubjectAggregator: 任务表示 → 被试表示
    5. PredictionHead: 被试表示 → 分数预测
    """

    def __init__(
        self,
        input_dim: int = 7,
        segment_d_model: int = 64,
        segment_nhead: int = 4,
        segment_num_layers: int = 4,
        task_d_model: int = 128,
        task_nhead: int = 4,
        task_num_layers: int = 2,
        attention_dim: int = 32,
        dropout: float = 0.1,
        max_seq_len: int = 100,
        max_tasks: int = 30,
        max_segments: int = 30,
    ):
        """
        初始化

        Args:
            input_dim: 输入特征维度（7维增强眼动特征）
            segment_d_model: 片段编码器维度
            segment_nhead: 片段编码器注意力头数
            segment_num_layers: 片段编码器层数
            task_d_model: 任务编码器维度
            task_nhead: 任务编码器注意力头数
            task_num_layers: 任务编码器层数
            attention_dim: 聚合注意力维度
            dropout: Dropout比例
            max_seq_len: 最大序列长度
            max_tasks: 最大任务数
            max_segments: 最大片段数
        """
        super().__init__()

        self.max_tasks = max_tasks
        self.max_segments = max_segments

        # 片段编码器
        self.segment_encoder = GazeTransformerEncoder(
            input_dim=input_dim,
            d_model=segment_d_model,
            nhead=segment_nhead,
            num_layers=segment_num_layers,
            dim_feedforward=segment_d_model * 4,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )

        # 任务聚合器（从片段到任务）
        self.task_aggregator = AttentionPooling(
            input_dim=segment_d_model,
            attention_dim=attention_dim,
            dropout=dropout,
        )

        # 任务序列编码器
        self.task_encoder = TaskTransformerEncoder(
            input_dim=segment_d_model,
            d_model=task_d_model,
            nhead=task_nhead,
            num_layers=task_num_layers,
            dim_feedforward=task_d_model * 4,
            dropout=dropout,
            max_tasks=max_tasks,
        )

        # 被试聚合器（从任务到被试）
        self.subject_aggregator = AttentionPooling(
            input_dim=task_d_model,
            attention_dim=attention_dim,
            dropout=dropout,
        )

        # 预测头
        self.prediction_head = nn.Sequential(
            nn.Linear(task_d_model, task_d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(task_d_model // 2, 1),
        )

    def forward(
        self,
        segments: torch.Tensor,
        segment_mask: torch.Tensor,
        task_mask: torch.Tensor,
        segment_lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            segments: (batch, max_tasks, max_segments, max_seq_len, input_dim) 眼动序列
            segment_mask: (batch, max_tasks, max_segments) 有效片段掩码
            task_mask: (batch, max_tasks) 有效任务掩码
            segment_lengths: (batch, max_tasks, max_segments) 每个片段的实际长度（可选）

        Returns:
            字典包含：
            - prediction: (batch,) 预测分数
            - segment_attention: (batch, max_tasks, max_segments) 片段注意力权重
            - task_attention: (batch, max_tasks) 任务注意力权重
        """
        batch_size = segments.size(0)
        device = segments.device

        # 1. 编码所有片段
        # 将(batch, tasks, segments, seq, feat)展平为(batch*tasks*segments, seq, feat)
        flat_segments = segments.view(-1, segments.size(-2), segments.size(-1))

        # 构建序列掩码
        if segment_lengths is not None:
            # 根据长度构建掩码
            flat_lengths = segment_lengths.view(-1)
            max_len = segments.size(-2)
            seq_mask = torch.arange(max_len, device=device).unsqueeze(0) < flat_lengths.unsqueeze(1)
        else:
            seq_mask = None

        # 编码
        segment_reprs, _ = self.segment_encoder(flat_segments, seq_mask)  # (batch*tasks*segments, d_model)

        # 重塑为(batch, tasks, segments, d_model)
        segment_reprs = segment_reprs.view(batch_size, self.max_tasks, self.max_segments, -1)

        # 2. 聚合片段到任务
        # 对每个任务分别聚合片段
        task_reprs_list = []
        segment_attns_list = []

        for t in range(self.max_tasks):
            task_segment_reprs = segment_reprs[:, t, :, :]  # (batch, segments, d_model)
            task_segment_mask = segment_mask[:, t, :]  # (batch, segments)

            task_repr, seg_attn = self.task_aggregator(task_segment_reprs, task_segment_mask)
            task_reprs_list.append(task_repr)
            segment_attns_list.append(seg_attn)

        task_reprs = torch.stack(task_reprs_list, dim=1)  # (batch, tasks, d_model)
        segment_attentions = torch.stack(segment_attns_list, dim=1)  # (batch, tasks, segments)

        # 3. 编码任务序列
        task_encoded = self.task_encoder(task_reprs, task_mask)  # (batch, tasks, task_d_model)

        # 4. 聚合任务到被试
        subject_repr, task_attention = self.subject_aggregator(task_encoded, task_mask)
        # subject_repr: (batch, task_d_model)
        # task_attention: (batch, tasks)

        # 5. 预测分数
        prediction = self.prediction_head(subject_repr).squeeze(-1)  # (batch,)

        return {
            'prediction': prediction,
            'segment_attention': segment_attentions,
            'task_attention': task_attention,
        }


class SimplifiedTransformerNetwork(nn.Module):
    """
    简化的Transformer网络

    不使用层级结构，直接将所有片段展平后编码。
    用于对比实验。
    """

    def __init__(
        self,
        input_dim: int = 7,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_total_segments: int = 900,  # 30 tasks * 30 segments
    ):
        super().__init__()

        self.segment_encoder = GazeTransformerEncoder(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.aggregator = AttentionPooling(
            input_dim=d_model,
            attention_dim=d_model // 2,
            dropout=dropout,
        )

        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        segments: torch.Tensor,
        segment_mask: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            segments: (batch, max_tasks, max_segments, max_seq_len, input_dim)
            segment_mask: (batch, max_tasks, max_segments)
        """
        batch_size = segments.size(0)

        # 展平片段
        flat_segments = segments.view(batch_size, -1, segments.size(-2), segments.size(-1))
        flat_mask = segment_mask.view(batch_size, -1)

        # 编码每个片段
        num_segments = flat_segments.size(1)
        segment_reprs = []

        for i in range(num_segments):
            repr, _ = self.segment_encoder(flat_segments[:, i, :, :])
            segment_reprs.append(repr)

        segment_reprs = torch.stack(segment_reprs, dim=1)  # (batch, total_segments, d_model)

        # 聚合
        subject_repr, attention = self.aggregator(segment_reprs, flat_mask)

        # 预测
        prediction = self.prediction_head(subject_repr).squeeze(-1)

        return {
            'prediction': prediction,
            'attention': attention,
        }
