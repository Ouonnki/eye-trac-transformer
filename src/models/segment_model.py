# -*- coding: utf-8 -*-
"""
片段级编码器模块

提供简单的片段级编码器，用于方案A的实现。
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.config import ModelConfig
from src.models.base import BaseModel
from src.models.attention import PositionalEncoding
from src.models.dl_dataset import SequenceConfig
from src.models.encoders import GazeTransformerEncoder
from src.models.heads import PredictionHead

logger = logging.getLogger(__name__)


class SegmentEncoder(BaseModel):
    """
    片段级编码器

    将单个片段的眼动序列编码为预测结果。

    结构：
    1. GazeTransformerEncoder: 片段序列 → 片段表示
    2. PredictionHead: 片段表示 → 预测

    与层级模型不同，这里直接输出片段级预测，
    被试级预测由推理时的投票/平均聚合得到。
    """

    def __init__(
        self,
        model_config: ModelConfig,
        seq_config: SequenceConfig,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        num_classes: int = 1,
        use_gradient_checkpointing: bool = False,
    ):
        """
        初始化

        Args:
            model_config: 模型配置
            seq_config: 序列配置
            d_model: 模型维度
            nhead: 注意力头数
            num_layers: Transformer层数
            dim_feedforward: 前馈网络维度
            dropout: Dropout比例
            num_classes: 输出类别数（1=回归，>1=分类）
            use_gradient_checkpointing: 是否使用梯度检查点
        """
        super().__init__()

        self.d_model = d_model
        self.num_classes = num_classes

        # 片段编码器（复用现有的 GazeTransformerEncoder）
        self.segment_encoder = GazeTransformerEncoder(
            input_dim=model_config.input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=seq_config.max_seq_len,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

        # 预测头
        self.prediction_head = PredictionHead(
            input_dim=d_model,
            hidden_dim=d_model // 2,
            output_dim=num_classes,
            dropout=dropout,
        )

    def forward(
        self,
        features: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            features: (batch, seq_len, input_dim) 眼动序列
            lengths: (batch,) 每个序列的实际长度

        Returns:
            (batch, num_classes) 预测结果
        """
        # 构建序列掩码
        if lengths is not None:
            batch_size = features.size(0)
            max_len = features.size(1)
            mask = torch.arange(max_len, device=features.device).unsqueeze(0) < lengths.unsqueeze(1)
        else:
            mask = None

        # 编码片段
        segment_repr, _ = self.segment_encoder(features, mask)  # (batch, d_model)

        # 预测
        output = self.prediction_head(segment_repr)  # (batch, num_classes)

        return output

    @classmethod
    def from_config(
        cls,
        config,
        seq_config: SequenceConfig,
        **kwargs,
    ) -> 'SegmentEncoder':
        """
        从配置创建模型

        Args:
            config: UnifiedConfig
            seq_config: SequenceConfig
            **kwargs: 额外参数（如 num_classes）

        Returns:
            SegmentEncoder 实例
        """
        num_classes = kwargs.get('num_classes', 1)

        return cls(
            model_config=config.model,
            seq_config=seq_config,
            d_model=config.model.segment_d_model,
            nhead=config.model.segment_nhead,
            num_layers=config.model.segment_num_layers,
            dim_feedforward=config.model.segment_d_model * 4,
            dropout=config.model.dropout,
            num_classes=num_classes,
            use_gradient_checkpointing=config.device.use_gradient_checkpointing,
        )
