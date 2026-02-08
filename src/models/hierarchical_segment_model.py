# -*- coding: utf-8 -*-
"""
层级-片段联合模型

结合层级 Transformer 网络和片段级预测的联合模型。
结构：
1. 共享片段编码器 (GazeTransformerEncoder): 编码眼动序列为片段表示
2. 片段级预测分支: 片段表示 → 片段预测头 → 片段级标签预测
3. 层级预测分支: 片段表示 → 任务聚合 → 任务编码 → 被试聚合 → 被试级预测

损失函数为两个预测损失的加权和。
"""

from typing import Optional, Dict

import torch
import torch.nn as nn

from src.config import UnifiedConfig
from src.models.base import BaseModel
from src.models.encoders import GazeTransformerEncoder, TaskTransformerEncoder
from src.models.attention import AttentionPooling
from src.models.task_embedding import TaskEmbedding
from src.models.heads import PredictionHead
from src.models.dl_dataset import SequenceConfig


class HierarchicalSegmentTransformerNetwork(BaseModel):
    """
    结合层级 Transformer 网络和片段级预测的联合模型

    结构：
    1. 共享片段编码器 (GazeTransformerEncoder): 编码眼动序列为片段表示
    2. 片段级预测分支: 片段表示 → 片段预测头 → 片段级标签预测
    3. 层级预测分支: 片段表示 → 任务聚合 → 任务编码 → 被试聚合 → 被试级预测

    损失函数为两个预测损失的加权和。
    """

    def __init__(
        self,
        model_config: 'ModelConfig',
        seq_config: SequenceConfig,
        device_config: 'DeviceConfig',
        num_classes: int = 1,
        segment_loss_weight: float = 0.5,
        hierarchical_loss_weight: float = 0.5,
    ):
        """
        初始化

        Args:
            model_config: 模型架构配置
            seq_config: 序列配置
            device_config: 设备配置
            num_classes: 输出类别数，1 表示回归，>1 表示分类
            segment_loss_weight: 片段级损失权重
            hierarchical_loss_weight: 层级（被试级）损失权重
        """
        super().__init__()

        self.num_classes = num_classes
        self.segment_loss_weight = segment_loss_weight
        self.hierarchical_loss_weight = hierarchical_loss_weight
        self.max_tasks = seq_config.max_tasks
        self.max_segments = seq_config.max_segments

        # 共享的片段编码器
        self.segment_encoder = GazeTransformerEncoder(
            input_dim=seq_config.input_dim,
            d_model=model_config.segment_d_model,
            nhead=model_config.segment_nhead,
            num_layers=model_config.segment_num_layers,
            dim_feedforward=model_config.segment_d_model * 4,
            dropout=model_config.dropout,
            max_seq_len=seq_config.max_seq_len,
            use_gradient_checkpointing=device_config.use_gradient_checkpointing,
        )

        # 片段级预测头
        self.segment_prediction_head = PredictionHead(
            input_dim=model_config.segment_d_model,
            hidden_dim=model_config.segment_d_model // 2,
            output_dim=num_classes if num_classes > 1 else 1,
            dropout=model_config.dropout,
        )

        # 任务聚合器（从片段到任务）
        self.task_aggregator = AttentionPooling(
            input_dim=model_config.segment_d_model,
            attention_dim=model_config.attention_dim,
            dropout=model_config.dropout,
        )

        # 任务嵌入模块
        self.use_task_embedding = getattr(model_config, 'use_task_embedding', False)
        if self.use_task_embedding:
            self.task_embedding = TaskEmbedding(
                task_embedding_dim=getattr(model_config, 'task_embedding_dim', 2),
            )
            self.task_emb_output_dim = self.task_embedding.output_dim
            task_encoder_input_dim = model_config.segment_d_model + self.task_emb_output_dim
        else:
            self.task_embedding = None
            self.task_emb_output_dim = 0
            task_encoder_input_dim = model_config.segment_d_model

        # 任务序列编码器
        self.task_encoder = TaskTransformerEncoder(
            input_dim=task_encoder_input_dim,
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

        # 被试级预测头
        self.subject_prediction_head = PredictionHead(
            input_dim=model_config.task_d_model,
            hidden_dim=model_config.task_d_model // 2,
            output_dim=num_classes if num_classes > 1 else 1,
            dropout=model_config.dropout,
        )

    @classmethod
    def from_config(
        cls,
        config: UnifiedConfig,
        seq_config: SequenceConfig = None,
        **kwargs,
    ) -> 'HierarchicalSegmentTransformerNetwork':
        """
        从配置创建模型

        Args:
            config: 统一配置对象
            seq_config: 序列配置对象（如果为None，则从config.sequence自动创建）
            **kwargs: 额外参数（如 num_classes, segment_loss_weight, hierarchical_loss_weight）

        Returns:
            模型实例
        """
        if seq_config is None:
            seq_config = config.to_seq_config()

        # 从 kwargs 获取参数
        num_classes = kwargs.get('num_classes', None)
        if num_classes is None:
            num_classes = config.task.num_classes

        segment_loss_weight = kwargs.get('segment_loss_weight', 0.5)
        hierarchical_loss_weight = kwargs.get('hierarchical_loss_weight', 0.5)

        return cls(
            model_config=config.model,
            seq_config=seq_config,
            device_config=config.device,
            num_classes=num_classes,
            segment_loss_weight=segment_loss_weight,
            hierarchical_loss_weight=hierarchical_loss_weight,
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
            - subject_prediction: (batch,) 或 (batch, num_classes) 被试级预测
            - segment_predictions: (batch, max_tasks, max_segments) 片段级预测
            - subject_repr: (batch, task_d_model) 被试级特征表示
            - segment_reprs: (batch, max_tasks, max_segments, segment_d_model) 片段特征
            - segment_attention: (batch, max_tasks, max_segments) 片段注意力权重
            - task_attention: (batch, max_tasks) 任务注意力权重
        """
        batch_size = segments.size(0)
        device = segments.device

        # ========== 1. 共享片段编码器 ==========
        # 将 (batch, tasks, segments, seq, feat) 展平为 (batch*tasks*segments, seq, feat)
        flat_segments = segments.view(-1, segments.size(-2), segments.size(-1))

        # 构建序列掩码
        if segment_lengths is not None:
            flat_lengths = segment_lengths.view(-1)
            max_len = segments.size(-2)
            seq_mask = torch.arange(max_len, device=device).unsqueeze(0) < flat_lengths.unsqueeze(1)
        else:
            seq_mask = None

        # 编码片段: (batch*tasks*segments, segment_d_model)
        segment_reprs_flat, _ = self.segment_encoder(flat_segments, seq_mask)

        # 重塑为 (batch, tasks, segments, segment_d_model)
        segment_reprs = segment_reprs_flat.view(
            batch_size, self.max_tasks, self.max_segments, -1
        )

        # ========== 2. 片段级预测分支 ==========
        # 片段级预测: (batch*tasks*segments, output_dim)
        segment_predictions_flat = self.segment_prediction_head(segment_reprs_flat)

        # 重塑为 (batch, max_tasks, max_segments, num_classes) 或 (batch, max_tasks, max_segments)
        if self.num_classes == 1:
            segment_predictions = segment_predictions_flat.squeeze(-1).view(
                batch_size, self.max_tasks, self.max_segments
            )
        else:
            segment_predictions = segment_predictions_flat.view(
                batch_size, self.max_tasks, self.max_segments, self.num_classes
            )

        # ========== 3. 层级预测分支 ==========
        # 将 (batch, tasks, segments, d_model) 重塑为 (batch*tasks, segments, d_model)
        segment_reprs_flat_for_agg = segment_reprs.view(
            batch_size * self.max_tasks, self.max_segments, -1
        )
        segment_mask_flat = segment_mask.view(batch_size * self.max_tasks, self.max_segments)

        # 任务聚合
        task_reprs_flat, segment_attns_flat = self.task_aggregator(
            segment_reprs_flat_for_agg, segment_mask_flat
        )

        # 重塑回 (batch, tasks, d_model) 和 (batch, tasks, segments)
        task_reprs = task_reprs_flat.view(batch_size, self.max_tasks, -1)
        segment_attentions = segment_attns_flat.view(batch_size, self.max_tasks, self.max_segments)

        # 任务嵌入
        if self.use_task_embedding and self.task_embedding is not None:
            if task_conditions is not None:
                task_embs = []
                for t_idx in range(self.max_tasks):
                    tc = task_conditions[:, t_idx, :]
                    te = self.task_embedding(
                        grid_scale=tc[:, 0].long(),
                        continuous_thinking=tc[:, 1].long(),
                        click_disappear=tc[:, 2].long(),
                        has_distractor=tc[:, 3].long(),
                        has_task_distractor=tc[:, 4].long(),
                    )
                    task_embs.append(te)
                task_emb_seq = torch.stack(task_embs, dim=1)

                if task_mask is not None:
                    mask_expanded = task_mask.unsqueeze(-1).float()
                    task_emb_seq = task_emb_seq * mask_expanded
            else:
                task_emb_seq = torch.zeros(
                    batch_size, self.max_tasks, self.task_emb_output_dim,
                    device=device
                )

            task_reprs = torch.cat([task_reprs, task_emb_seq], dim=-1)

        # 任务编码
        task_encoded = self.task_encoder(task_reprs, task_mask)

        # 被试聚合
        subject_repr, task_attention = self.subject_aggregator(task_encoded, task_mask)

        # 被试级预测
        subject_prediction = self.subject_prediction_head(subject_repr)
        if self.num_classes == 1:
            subject_prediction = subject_prediction.squeeze(-1)

        return {
            'subject_prediction': subject_prediction,
            'segment_predictions': segment_predictions,
            'subject_repr': subject_repr,
            'segment_reprs': segment_reprs,
            'segment_attention': segment_attentions,
            'task_attention': task_attention,
        }

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        subject_targets: torch.Tensor,
        segment_targets: Optional[torch.Tensor] = None,
        segment_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算联合损失

        Args:
            outputs: forward 方法的输出字典
            subject_targets: (batch,) 被试级目标标签
            segment_targets: (batch, max_tasks, max_segments) 片段级目标标签（可选）
            segment_mask: (batch, max_tasks, max_segments) 有效片段掩码

        Returns:
            字典包含：
            - total_loss: 加权总损失
            - subject_loss: 被试级损失
            - segment_loss: 片段级损失
        """
        if self.num_classes == 1:
            # 回归任务
            subject_loss = nn.functional.mse_loss(
                outputs['subject_prediction'], subject_targets
            )
        else:
            # 分类任务
            subject_loss = nn.functional.cross_entropy(
                outputs['subject_prediction'], subject_targets.long()
            )

        # 片段级损失（如果提供了片段目标）
        if segment_targets is not None and segment_mask is not None:
            if self.num_classes == 1:
                # 回归任务：使用 MSE，只对有效片段计算
                segment_preds = outputs['segment_predictions']
                diff = (segment_preds - segment_targets) ** 2
                # 应用掩码
                masked_diff = diff * segment_mask.float()
                # 计算平均（只考虑有效片段）
                segment_loss = masked_diff.sum() / (segment_mask.sum() + 1e-8)
            else:
                # 分类任务：使用 ignore_index 忽略无效标签（-100）
                segment_preds = outputs['segment_predictions']
                batch_size, max_tasks, max_segments, num_classes = segment_preds.shape
                # 展平
                segment_preds_flat = segment_preds.view(-1, num_classes)
                segment_targets_flat = segment_targets.view(-1).long()

                # 计算损失（使用 ignore_index 自动忽略 -100 标签）
                segment_loss = nn.functional.cross_entropy(
                    segment_preds_flat, segment_targets_flat, 
                    ignore_index=-100,  # 忽略无效片段
                    reduction='mean'    # 自动平均有效样本
                )
        else:
            # 如果没有片段目标，片段损失为0
            segment_loss = torch.tensor(0.0, device=subject_targets.device)

        # 加权总损失
        total_loss = (
            self.hierarchical_loss_weight * subject_loss +
            self.segment_loss_weight * segment_loss
        )

        return {
            'total_loss': total_loss,
            'subject_loss': subject_loss,
            'segment_loss': segment_loss,
        }
