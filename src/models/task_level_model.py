# -*- coding: utf-8 -*-
"""
任务级眼动识别模型

结构：
1. 片段编码器 (GazeTransformerEncoder): (25, 300, 7) -> (25, 96)
2. 片段聚合 (AttentionPooling): (25, 96) -> (96)
3. 任务嵌入 (TaskEmbedding): (5,) -> (10)
4. 拼接 + 预测头: (106) -> 3
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoders import GazeTransformerEncoder, GazeCnnEncoder, GazeRnnEncoder
from src.models.attention import AttentionPooling
from src.models.task_embedding import TaskEmbedding, MLPTaskEmbedding
from src.models.heads import PredictionHead

logger = logging.getLogger(__name__)


class TaskLevelEncoder(nn.Module):
    """
    任务级编码器
    
    输入: 
        - segments: (B, 25, 300, 7) 眼动序列
        - segment_mask: (B, 25) 有效片段掩码
        - task_conditions: (B, 5) 任务条件
    输出:
        - classification 模式: (B, 3) 任务级分类 logits
        - ordinal 模式: (B, 2) 序数累计 logits（当类别数为3）
    """

    def __init__(
        self,
        input_dim: int = 7,
        max_seq_len: int = 300,
        max_segments: int = 25,
        segment_d_model: int = 96,
        segment_nhead: int = 4,
        segment_num_layers: int = 4,
        segment_encoder_type: str = "transformer",
        segment_rnn_hidden_size: Optional[int] = None,
        segment_rnn_layers: int = 1,
        segment_rnn_dropout: float = 0.0,
        segment_cnn_channels: Optional[list] = None,
        segment_cnn_kernel_sizes: Optional[list] = None,
        attention_dim: int = 32,
        task_embedding_dim: int = 2,
        use_task_embedding: bool = True,
        task_embedding_type: str = "independent",
        use_conditional_pooling: bool = False,
        dropout: float = 0.5,
        num_classes: int = 3,
        head_type: str = "classification",
        use_gradient_checkpointing: bool = False,
        use_output_norm: bool = False,
        task_embedding_normalize: bool = False,
        use_conditional_cutpoints: bool = False,
    ):
        super().__init__()

        self.max_segments = max_segments
        self.segment_d_model = segment_d_model
        self.use_task_embedding = use_task_embedding
        self.task_embedding_type = task_embedding_type
        self.use_conditional_pooling = use_conditional_pooling
        self.num_classes = num_classes
        self.head_type = head_type
        if self.head_type not in {"classification", "ordinal"}:
            raise ValueError(f"不支持的 head_type: {self.head_type}")
        if self.head_type == "ordinal" and self.num_classes < 2:
            raise ValueError("ordinal 模式要求 num_classes >= 2")
        self.use_output_norm = use_output_norm
        self.task_embedding_normalize = task_embedding_normalize
        self.use_conditional_cutpoints = use_conditional_cutpoints
        if self.use_conditional_cutpoints and self.head_type != "ordinal":
            raise ValueError("use_conditional_cutpoints 仅在 ordinal 模式下支持")
        if self.use_conditional_cutpoints and not use_task_embedding:
            raise ValueError("use_conditional_cutpoints 需要 use_task_embedding=true")
        
        # 1. 片段编码器
        if not segment_encoder_type:
            raise ValueError("segment_encoder_type 不能为空")
        self.segment_encoder_type = segment_encoder_type.lower()

        if self.segment_encoder_type == "transformer":
            self.segment_encoder = GazeTransformerEncoder(
                input_dim=input_dim,
                d_model=segment_d_model,
                nhead=segment_nhead,
                num_layers=segment_num_layers,
                dim_feedforward=segment_d_model * 4,
                dropout=dropout,
                max_seq_len=max_seq_len,
                use_gradient_checkpointing=use_gradient_checkpointing,
            )
        elif self.segment_encoder_type == "cnn1d":
            self.segment_encoder = GazeCnnEncoder(
                input_dim=input_dim,
                channels=segment_cnn_channels or [],
                kernel_sizes=segment_cnn_kernel_sizes or [],
                dropout=dropout,
                output_dim=segment_d_model,
            )
        elif self.segment_encoder_type in {"rnn", "lstm", "gru", "bilstm"}:
            rnn_type = "lstm" if self.segment_encoder_type == "bilstm" else self.segment_encoder_type
            hidden_size = segment_rnn_hidden_size or segment_d_model
            self.segment_encoder = GazeRnnEncoder(
                rnn_type=rnn_type,
                input_dim=input_dim,
                hidden_size=hidden_size,
                num_layers=segment_rnn_layers,
                dropout=segment_rnn_dropout,
                bidirectional=self.segment_encoder_type == "bilstm",
                output_dim=segment_d_model,
                use_output_norm=use_output_norm,
            )
        else:
            raise ValueError(f"不支持的片段编码器类型: {self.segment_encoder_type}")
        
        # 2. 片段->任务聚合 (延后创建，需要先确定 cond_dim)
        # 3. 任务嵌入
        if self.use_task_embedding:
            if task_embedding_type == "mlp":
                self.task_embedding = MLPTaskEmbedding(
                    input_dim=5,
                    hidden_dim=max(task_embedding_dim * 2, 32),
                    output_dim=task_embedding_dim,
                )
                self.task_emb_output_dim = task_embedding_dim
            else:
                self.task_embedding = TaskEmbedding(
                    task_embedding_dim=task_embedding_dim,
                    normalize=task_embedding_normalize,
                )
                self.task_emb_output_dim = self.task_embedding.output_dim  # = 5 * dim
        else:
            self.task_embedding = None
            self.task_emb_output_dim = 0

        # 片段聚合器（可条件化）
        cond_dim = self.task_emb_output_dim if (use_task_embedding and use_conditional_pooling) else None
        self.segment_aggregator = AttentionPooling(
            input_dim=segment_d_model,
            attention_dim=attention_dim,
            dropout=dropout,
            cond_dim=cond_dim,
        )
        
        # 4. 预测头
        if self.use_conditional_cutpoints:
            # 条件化 cutpoints：骨干输出标量分数 z，任务条件输出阈值 θ
            self.score_head = nn.Linear(segment_d_model, 1)
            self.cutpoint_net = nn.Linear(self.task_emb_output_dim, num_classes - 1)
            self.prediction_head = None
        else:
            # (96 + 10 = 106 -> 3, 或 96 -> 3 当不使用任务嵌入时)
            head_input_dim = segment_d_model + self.task_emb_output_dim
            head_output_dim = num_classes if self.head_type == "classification" else (num_classes - 1)
            self.prediction_head = PredictionHead(
                input_dim=head_input_dim,
                hidden_dim=head_input_dim // 2,  # 53
                output_dim=head_output_dim,
                dropout=dropout,
            )
        
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        segments: torch.Tensor,
        segment_mask: Optional[torch.Tensor] = None,
        task_conditions: Optional[torch.Tensor] = None,
        segment_seq_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            segments: (B, S, 300, 7) 眼动序列（S为每个任务片段数）
            segment_mask: (B, S) 有效片段掩码
            task_conditions: (B, 5) 任务条件 [grid_scale, continuous_thinking(0=1-N,1=1-99), click_disappear, has_distractor, has_task_distractor]
            segment_seq_mask: (B, S, 300) 片段内有效时步掩码
        
        Returns:
            logits:
                - classification: (B, C)
                - ordinal: (B, C-1)
        """
        B = segments.size(0)
        
        # 1. 编码所有片段
        # (B, S, 300, 7) -> (B*S, 300, 7)
        num_segments = segments.size(1)
        flat_segments = segments.view(B * num_segments, -1, segments.size(-1))
        if segment_seq_mask is not None:
            flat_seq_mask = segment_seq_mask.view(B * num_segments, -1)
        else:
            flat_seq_mask = None
        
        # 编码 -> (B*S, 96)
        segment_reprs, _ = self.segment_encoder(flat_segments, mask=flat_seq_mask)
        
        # 重塑 -> (B, S, 96)
        segment_reprs = segment_reprs.view(B, num_segments, self.segment_d_model)

        # 2. 计算任务嵌入（提前到聚合之前，用于条件化注意力）
        task_emb = None
        if self.use_task_embedding and task_conditions is not None:
            if self.task_embedding_type == "mlp":
                task_emb = self.task_embedding(task_conditions)
            else:
                task_emb = self.task_embedding(
                    grid_scale=task_conditions[:, 0].long(),
                    continuous_thinking=task_conditions[:, 1].long(),
                    click_disappear=task_conditions[:, 2].long(),
                    has_distractor=task_conditions[:, 3].long(),
                    has_task_distractor=task_conditions[:, 4].long(),
                )

        # 3. 聚合片段 -> (B, 96)，可选条件化注意力
        if self.use_conditional_pooling and task_emb is not None:
            task_repr, _ = self.segment_aggregator(segment_reprs, segment_mask, condition=task_emb)
        else:
            task_repr, _ = self.segment_aggregator(segment_reprs, segment_mask)

        # 4. 预测 -> (B, C) 或 (B, C-1)
        if self.use_conditional_cutpoints:
            # 条件化 cutpoints：z 只依赖眼动，θ 只依赖任务条件
            z = self.score_head(task_repr)                      # (B, 1) 场景无关分数
            theta = self.cutpoint_net(task_emb)                 # (B, K-1)
            theta0 = theta[:, 0:1]
            theta1 = theta0 + F.softplus(theta[:, 1:2])
            theta = torch.cat([theta0, theta1], dim=1)          # 保证 θ0 < θ1
            logits = z - theta                                  # (B, K-1)
        else:
            if task_emb is not None:
                fused = torch.cat([task_repr, task_emb], dim=-1)
            else:
                fused = task_repr
            logits = self.prediction_head(fused)
        
        return logits
