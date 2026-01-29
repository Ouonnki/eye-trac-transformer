# -*- coding: utf-8 -*-
"""
编码器模块

包含用于层级 Transformer 的编码器组件。
"""

from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from src.config import ModelConfig
from src.models.attention import AttentionPooling, PositionalEncoding
from src.models.dl_dataset import SequenceConfig


class GazeTransformerEncoder(nn.Module):
    """
    眼动序列 Transformer 编码器

    将单个片段的眼动序列编码为固定维度的表示。

    结构：
    - 输入嵌入: Linear(input_dim → d_model)
    - 位置编码: 正弦位置编码
    - [CLS] token: 可学习的分类 token
    - Transformer Encoder: 多层自注意力
    - 输出: [CLS] token 的表示
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_seq_len: int,
        use_gradient_checkpointing: bool = False,
        use_task_embedding: bool = False,
        task_embedding_dim: int = 16,
    ):
        """
        初始化

        Args:
            input_dim: 输入特征维度
            d_model: 模型维度
            nhead: 注意力头数
            num_layers: Transformer 层数
            dim_feedforward: 前馈网络维度
            dropout: Dropout 比例
            max_seq_len: 最大序列长度
            use_gradient_checkpointing: 是否使用梯度检查点节省显存
            use_task_embedding: 是否使用任务嵌入
            task_embedding_dim: 任务嵌入维度
        """
        super().__init__()

        self.d_model = d_model
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # 输入投影
        self.input_proj = nn.Linear(input_dim, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len + 1, dropout=dropout)

        # [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # 任务嵌入（可选）
        self.use_task_embedding = use_task_embedding
        if use_task_embedding:
            from src.models.task_embedding import TaskEmbedding
            self.task_embedding = TaskEmbedding(d_model=d_model, embedding_dim=task_embedding_dim)

        # Transformer 编码器层（分开存储以支持梯度检查点）
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                activation='gelu',
            )
            for _ in range(num_layers)
        ])

        # LayerNorm
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        task_conditions: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: (batch, seq_len, input_dim) 眼动序列
            mask: (batch, seq_len) 有效位置掩码（True 表示有效）
            task_conditions: 任务条件字典，包含:
                - grid_scale: (batch,)
                - continuous_thinking: (batch,)
                - click_disappear: (batch,)
                - has_distractor: (batch,)
                - has_task_distractor: (batch,)

        Returns:
            output: (batch, d_model) 片段表示
            attention_weights: None
        """
        batch_size = x.size(0)

        # 输入投影
        x = self.input_proj(x)  # (batch, seq_len, d_model)

        # 添加 [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d_model)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, seq_len+1, d_model)

        # 添加任务嵌入（在位置编码之前）
        if self.use_task_embedding and task_conditions is not None:
            task_emb = self.task_embedding(
                grid_scale=task_conditions['grid_scale'],
                continuous_thinking=task_conditions['continuous_thinking'],
                click_disappear=task_conditions['click_disappear'],
                has_distractor=task_conditions['has_distractor'],
                has_task_distractor=task_conditions['has_task_distractor'],
            )  # (batch, d_model)

            # 广播到所有位置（包括 [CLS] token）
            x = x + task_emb.unsqueeze(1)  # (batch, seq_len+1, d_model)

        # 位置编码
        x = self.pos_encoder(x)

        # 构建 attention mask
        if mask is not None:
            # 为 [CLS] 位置添加 True
            cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=mask.device)
            full_mask = torch.cat([cls_mask, mask], dim=1)  # (batch, seq_len+1)
            # Transformer 使用的是 key_padding_mask，True 表示忽略
            key_padding_mask = ~full_mask
        else:
            key_padding_mask = None

        # Transformer 编码（支持梯度检查点）
        for layer in self.encoder_layers:
            if self.use_gradient_checkpointing and self.training:
                # 使用梯度检查点节省显存
                x = checkpoint(layer, x, None, key_padding_mask, use_reentrant=False)
            else:
                x = layer(x, src_key_padding_mask=key_padding_mask)

        # 取 [CLS] token 的输出
        output = self.norm(x[:, 0, :])  # (batch, d_model)

        return output, None


class TaskTransformerEncoder(nn.Module):
    """
    任务级 Transformer 编码器

    编码被试的任务序列，捕获任务间的依赖关系（如疲劳效应、学习效应）。
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_tasks: int,
    ):
        """
        初始化

        Args:
            input_dim: 输入特征维度（任务表示维度）
            d_model: 模型维度
            nhead: 注意力头数
            num_layers: Transformer 层数
            dim_feedforward: 前馈网络维度
            dropout: Dropout 比例
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

        # Transformer 编码器
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
            mask: (batch, num_tasks) 有效任务掩码（True 表示有效）

        Returns:
            output: (batch, num_tasks, d_model) 编码后的任务序列
        """
        # 输入投影
        x = self.input_proj(x)  # (batch, num_tasks, d_model)

        # 位置编码
        x = self.pos_encoder(x)

        # 构建 attention mask
        if mask is not None:
            key_padding_mask = ~mask
        else:
            key_padding_mask = None

        # Transformer 编码
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)

        return x


class HierarchicalEncoder(nn.Module):
    """
    层级编码器（仅特征提取部分）

    统一的编码逻辑，供 HierarchicalTransformerNetwork 和 CADT 共用。

    结构：
    1. GazeTransformerEncoder: 片段 → 片段表示
    2. AttentionPooling: 片段 → 任务表示
    3. TaskTransformerEncoder: 任务序列 → 编码任务
    4. AttentionPooling: 任务 → 被试表示
    """

    def __init__(
        self,
        model_config: ModelConfig,
        seq_config: SequenceConfig,
        use_gradient_checkpointing: bool = False,
    ):
        """
        初始化

        Args:
            model_config: 模型架构配置
            seq_config: 序列配置
            use_gradient_checkpointing: 是否使用梯度检查点
        """
        super().__init__()

        self.max_tasks = seq_config.max_tasks
        self.max_segments = seq_config.max_segments

        # 片段编码器
        self.segment_encoder = GazeTransformerEncoder(
            input_dim=model_config.input_dim,
            d_model=model_config.segment_d_model,
            nhead=model_config.segment_nhead,
            num_layers=model_config.segment_num_layers,
            dim_feedforward=model_config.segment_d_model * 4,
            dropout=model_config.dropout,
            max_seq_len=seq_config.max_seq_len,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

        # 任务聚合器（从片段到任务）
        self.task_aggregator = AttentionPooling(
            input_dim=model_config.segment_d_model,
            attention_dim=model_config.attention_dim,
            dropout=model_config.dropout,
        )

        # 任务序列编码器
        self.task_encoder = TaskTransformerEncoder(
            input_dim=model_config.segment_d_model,
            d_model=model_config.task_d_model,
            nhead=model_config.task_nhead,
            num_layers=model_config.task_num_layers,
            dim_feedforward=model_config.task_d_model * 4,
            dropout=model_config.dropout,
            max_tasks=seq_config.max_tasks,
        )

        # 被试聚合器（从任务到被试）
        self.subject_aggregator = AttentionPooling(
            input_dim=model_config.task_d_model,
            attention_dim=model_config.attention_dim,
            dropout=model_config.dropout,
        )

    def forward(
        self,
        segments: torch.Tensor,
        segment_mask: torch.Tensor,
        task_mask: torch.Tensor,
        segment_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播

        Args:
            segments: (batch, max_tasks, max_segments, max_seq_len, input_dim) 眼动序列
            segment_mask: (batch, max_tasks, max_segments) 有效片段掩码
            task_mask: (batch, max_tasks) 有效任务掩码
            segment_lengths: (batch, max_tasks, max_segments) 每个片段的实际长度

        Returns:
            subject_repr: (batch, task_d_model) 被试表示
            extras: 包含 segment_attention, task_attention 的字典
        """
        batch_size = segments.size(0)
        device = segments.device

        # 1. 编码所有片段
        # 将 (batch, tasks, segments, seq, feat) 展平为 (batch*tasks*segments, seq, feat)
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

        # 重塑为 (batch, tasks, segments, d_model)
        segment_reprs = segment_reprs.view(batch_size, self.max_tasks, self.max_segments, -1)

        # 2. 向量化聚合片段到任务
        # 将 (batch, tasks, segments, d_model) 重塑为 (batch*tasks, segments, d_model)
        segment_reprs_flat = segment_reprs.view(batch_size * self.max_tasks, self.max_segments, -1)
        segment_mask_flat = segment_mask.view(batch_size * self.max_tasks, self.max_segments)

        # 批量调用 task_aggregator
        task_reprs_flat, segment_attns_flat = self.task_aggregator(segment_reprs_flat, segment_mask_flat)
        # task_reprs_flat: (batch*tasks, d_model)
        # segment_attns_flat: (batch*tasks, segments)

        # 重塑回 (batch, tasks, d_model) 和 (batch, tasks, segments)
        task_reprs = task_reprs_flat.view(batch_size, self.max_tasks, -1)
        segment_attentions = segment_attns_flat.view(batch_size, self.max_tasks, self.max_segments)

        # 3. 编码任务序列
        task_encoded = self.task_encoder(task_reprs, task_mask)  # (batch, tasks, task_d_model)

        # 4. 聚合任务到被试
        subject_repr, task_attention = self.subject_aggregator(task_encoded, task_mask)
        # subject_repr: (batch, task_d_model)
        # task_attention: (batch, tasks)

        extras = {
            'segment_attention': segment_attentions,
            'task_attention': task_attention,
        }

        return subject_repr, extras
