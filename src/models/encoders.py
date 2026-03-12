# -*- coding: utf-8 -*-
"""
编码器模块

包含用于层级 Transformer 的编码器组件。
"""

from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch.nn.utils.rnn import pack_padded_sequence

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
        """
        super().__init__()

        self.d_model = d_model
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # 输入投影
        self.input_proj = nn.Linear(input_dim, d_model)

        # [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len + 1, dropout=dropout)

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: (batch, seq_len, input_dim) 眼动序列
            mask: (batch, seq_len) 有效位置掩码（True 表示有效）

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


def _mask_to_lengths(mask: Optional[torch.Tensor], batch_size: int, seq_len: int) -> torch.Tensor:
    if mask is None:
        return torch.full((batch_size,), seq_len, dtype=torch.long)
    lengths = mask.long().sum(dim=1)
    lengths = torch.clamp(lengths, min=1)
    return lengths


class GazeCnnEncoder(nn.Module):
    """
    1D-CNN 片段编码器

    输入: (batch, seq_len, input_dim)
    输出: (batch, output_dim)
    """

    def __init__(
        self,
        input_dim: int,
        channels: List[int],
        kernel_sizes: List[int],
        dropout: float,
        output_dim: int,
    ):
        super().__init__()
        if not channels or not kernel_sizes:
            raise ValueError("CNN 编码器配置缺少 channels 或 kernel_sizes")
        if len(channels) != len(kernel_sizes):
            raise ValueError("CNN 编码器 channels 与 kernel_sizes 长度必须一致")

        layers: List[nn.Module] = []
        in_channels = input_dim
        for out_channels, kernel in zip(channels, kernel_sizes):
            padding = kernel // 2
            layers.append(nn.Conv1d(in_channels, out_channels, kernel, padding=padding))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_channels = out_channels
        self.conv = nn.Sequential(*layers)

        if in_channels != output_dim:
            self.proj = nn.Linear(in_channels, output_dim)
        else:
            self.proj = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        feats = self.conv(x)   # (batch, channels, seq_len)

        if mask is not None:
            mask_f = mask.float().unsqueeze(1)
            feats = feats * mask_f
            lengths = mask_f.sum(dim=2).clamp(min=1.0)
            pooled = feats.sum(dim=2) / lengths
        else:
            pooled = feats.mean(dim=2)

        output = self.proj(pooled)
        return output, None


class GazeRnnEncoder(nn.Module):
    """
    RNN/LSTM/GRU 片段编码器
    """

    def __init__(
        self,
        rnn_type: str,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        output_dim: int,
    ):
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("RNN 编码器 hidden_size 必须大于 0")
        if num_layers <= 0:
            raise ValueError("RNN 编码器 num_layers 必须大于 0")

        rnn_dropout = dropout if num_layers > 1 else 0.0
        rnn_type = rnn_type.lower()
        if rnn_type == "rnn":
            self.rnn = nn.RNN(
                input_dim,
                hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
                bidirectional=bidirectional,
            )
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_dim,
                hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
                bidirectional=bidirectional,
            )
        elif rnn_type == "gru":
            self.rnn = nn.GRU(
                input_dim,
                hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
                bidirectional=bidirectional,
            )
        else:
            raise ValueError(f"不支持的 RNN 类型: {rnn_type}")

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        in_dim = hidden_size * self.num_directions
        if in_dim != output_dim:
            self.proj = nn.Linear(in_dim, output_dim)
        else:
            self.proj = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lengths = _mask_to_lengths(mask, x.size(0), x.size(1)).to(x.device)
        packed = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        outputs, state = self.rnn(packed)

        if isinstance(state, tuple):
            h_n = state[0]
        else:
            h_n = state

        h_n = h_n.view(self.num_layers, self.num_directions, x.size(0), self.hidden_size)
        last = h_n[-1]
        if self.num_directions == 2:
            reprs = torch.cat([last[0], last[1]], dim=-1)
        else:
            reprs = last[0]

        output = self.proj(reprs)
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
    3. [可选] TaskTransformerEncoder: 任务序列 → 编码任务
    4. AttentionPooling: 任务 → 被试表示
    """

    def __init__(
        self,
        model_config: ModelConfig,
        seq_config: SequenceConfig,
        use_gradient_checkpointing: bool = False,
        use_task_embedding: bool = False,
        task_embedding_dim: int = 2,
    ):
        """
        初始化

        Args:
            model_config: 模型架构配置
            seq_config: 序列配置
            use_gradient_checkpointing: 是否使用梯度检查点
            use_task_embedding: 是否使用任务嵌入（在任务编码器前）
            task_embedding_dim: 任务嵌入维度（所有五个条件统一）
        """
        super().__init__()

        self.max_tasks = seq_config.max_tasks
        self.max_segments = seq_config.max_segments
        self.use_task_embedding = use_task_embedding

        # 片段编码器
        self.segment_encoder = GazeTransformerEncoder(
            input_dim=seq_config.input_dim,
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

        # 任务嵌入模块（在任务编码器前）
        if use_task_embedding:
            from src.models.task_embedding import TaskEmbedding
            self.task_embedding = TaskEmbedding(
                task_embedding_dim=task_embedding_dim,
            )
            self.task_emb_output_dim = self.task_embedding.output_dim
            # 任务编码器输入维度 = 任务表示维度 + 任务嵌入维度
            task_encoder_input_dim = model_config.segment_d_model + self.task_emb_output_dim
        else:
            self.task_embedding = None
            self.task_emb_output_dim = 0
            task_encoder_input_dim = model_config.segment_d_model

        # 任务序列编码器（总是使用）
        self.task_encoder = TaskTransformerEncoder(
            input_dim=task_encoder_input_dim,
            d_model=model_config.task_d_model,
            nhead=model_config.task_nhead,
            num_layers=model_config.task_num_layers,
            dim_feedforward=model_config.task_d_model * 4,
            dropout=model_config.dropout,
            max_tasks=seq_config.max_tasks,
        )
        aggregator_input_dim = model_config.task_d_model

        # 被试聚合器（从任务到被试）
        self.subject_aggregator = AttentionPooling(
            input_dim=aggregator_input_dim,
            attention_dim=model_config.attention_dim,
            dropout=model_config.dropout,
        )

    def forward(
        self,
        segments: torch.Tensor,
        segment_mask: torch.Tensor,
        task_mask: torch.Tensor,
        segment_lengths: Optional[torch.Tensor] = None,
        task_conditions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播

        Args:
            segments: (batch, max_tasks, max_segments, max_seq_len, input_dim) 眼动序列
            segment_mask: (batch, max_tasks, max_segments) 有效片段掩码
            task_mask: (batch, max_tasks) 有效任务掩码
            segment_lengths: (batch, max_tasks, max_segments) 每个片段的实际长度
            task_conditions: (batch, max_tasks, 5) 任务条件，每个任务一个5维向量

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

        # 3. 任务嵌入：在任务编码器前，将任务条件嵌入与任务表示拼接
        if self.use_task_embedding and self.task_embedding is not None:
            if task_conditions is not None:
                # 对每个任务进行嵌入: (B, T, 5) -> (B, T, task_emb_output_dim)
                task_embs = []
                for t_idx in range(self.max_tasks):
                    tc = task_conditions[:, t_idx, :]  # (batch, 5)
                    te = self.task_embedding(
                        grid_scale=tc[:, 0].long(),
                        continuous_thinking=tc[:, 1].long(),
                        click_disappear=tc[:, 2].long(),
                        has_distractor=tc[:, 3].long(),
                        has_task_distractor=tc[:, 4].long(),
                    )  # (batch, task_emb_output_dim)
                    task_embs.append(te)
                task_emb_seq = torch.stack(task_embs, dim=1)  # (batch, T, task_emb_output_dim)
                
                # 应用任务掩码
                if task_mask is not None:
                    mask_expanded = task_mask.unsqueeze(-1).float()  # (batch, T, 1)
                    task_emb_seq = task_emb_seq * mask_expanded
            else:
                task_emb_seq = torch.zeros(
                    batch_size, self.max_tasks, self.task_emb_output_dim,
                    device=device
                )
            
            # 拼接任务表示和任务嵌入
            task_reprs = torch.cat([task_reprs, task_emb_seq], dim=-1)  # (batch, T, segment_d_model + task_emb_output_dim)

        # 4. 编码任务序列
        task_encoded = self.task_encoder(task_reprs, task_mask)  # (batch, tasks, task_d_model)

        # 4. 聚合任务到被试
        subject_repr, task_attention = self.subject_aggregator(task_encoded, task_mask)
        # subject_repr: (batch, task_d_model 或 segment_d_model)
        # task_attention: (batch, tasks)

        extras = {
            'segment_attention': segment_attentions,
            'task_attention': task_attention,
        }

        return subject_repr, extras
