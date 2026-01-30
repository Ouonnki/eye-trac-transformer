# -*- coding: utf-8 -*-
"""
深度学习模型架构模块

提供用于眼动注意力预测的层级 Transformer 网络。
"""

from typing import Optional, Dict

import torch
import torch.nn as nn

from src.config import UnifiedConfig
from src.models.base import BaseModel
from src.models.encoders import HierarchicalEncoder
from src.models.heads import PredictionHead
from src.models.dl_dataset import SequenceConfig

# 为了向后兼容，导出编码器类
from src.models.encoders import GazeTransformerEncoder, TaskTransformerEncoder


class HierarchicalTransformerNetwork(BaseModel):
    """
    层级 Transformer 网络

    完整的眼动注意力预测模型。

    结构：
    1. HierarchicalEncoder: 片段 → 任务 → 被试表示
    2. PredictionHead: 被试表示 → 预测

    使用 from_config() 从配置创建实例。
    """

    def __init__(
        self,
        model_config: 'ModelConfig',
        seq_config: SequenceConfig,
        device_config: 'DeviceConfig',
        num_classes: int = 1,
    ):
        """
        初始化

        Args:
            model_config: 模型架构配置
            seq_config: 序列配置
            device_config: 设备配置
            num_classes: 输出类别数，1 表示回归，>1 表示分类
        """
        super().__init__()

        self.num_classes = num_classes
        self.max_tasks = seq_config.max_tasks
        self.max_segments = seq_config.max_segments

        # 编码器（复用）
        self.encoder = HierarchicalEncoder(
            model_config=model_config,
            seq_config=seq_config,
            use_gradient_checkpointing=device_config.use_gradient_checkpointing,
            use_task_embedding=model_config.use_task_embedding,
            task_embedding_dim=model_config.task_embedding_dim,
        )

        # 预测头
        output_dim = num_classes if num_classes > 1 else 1
        self.prediction_head = PredictionHead(
            input_dim=model_config.task_d_model,
            hidden_dim=model_config.task_d_model // 2,
            output_dim=output_dim,
            dropout=model_config.dropout,
        )

    @classmethod
    def from_config(
        cls,
        config: UnifiedConfig,
        seq_config: SequenceConfig,
        **kwargs,
    ) -> 'HierarchicalTransformerNetwork':
        """
        从配置创建模型

        Args:
            config: 统一配置对象
            seq_config: 序列配置对象
            **kwargs: 额外参数（如 num_classes）

        Returns:
            模型实例
        """
        # 从 kwargs 获取 num_classes，默认根据 task 类型决定
        num_classes = kwargs.get('num_classes', None)
        if num_classes is None:
            num_classes = config.task.num_classes

        return cls(
            model_config=config.model,
            seq_config=seq_config,
            device_config=config.device,
            num_classes=num_classes,
        )

    def forward(
        self,
        segments: torch.Tensor,
        segment_mask: torch.Tensor,
        task_mask: torch.Tensor,
        segment_lengths: Optional[torch.Tensor] = None,
        task_conditions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            segments: (batch, max_tasks, max_segments, max_seq_len, input_dim) 眼动序列
            segment_mask: (batch, max_tasks, max_segments) 有效片段掩码
            task_mask: (batch, max_tasks) 有效任务掩码
            segment_lengths: (batch, max_tasks, max_segments) 每个片段的实际长度
            task_conditions: (batch, max_tasks, 5) 任务条件

        Returns:
            字典包含：
            - prediction: (batch,) 或 (batch, num_classes) 预测结果
            - subject_repr: (batch, task_d_model) 被试级特征表示
            - segment_attention: (batch, max_tasks, max_segments) 片段注意力权重
            - task_attention: (batch, max_tasks) 任务注意力权重
        """
        # 调用编码器
        subject_repr, extras = self.encoder(
            segments=segments,
            segment_mask=segment_mask,
            task_mask=task_mask,
            segment_lengths=segment_lengths,
            task_conditions=task_conditions,
        )

        # 预测
        prediction = self.prediction_head(subject_repr)

        # 回归任务: 压缩最后一维
        if self.num_classes == 1:
            prediction = prediction.squeeze(-1)

        return {
            'prediction': prediction,
            'subject_repr': subject_repr,
            'segment_attention': extras['segment_attention'],
            'task_attention': extras['task_attention'],
        }
