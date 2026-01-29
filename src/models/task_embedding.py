# -*- coding: utf-8 -*-
"""
任务嵌入模块

将任务条件（如视野规模、思维连续性等）编码为向量表示。
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class TaskCondition:
    """
    规范化的任务条件

    Attributes:
        grid_scale: 视野规模等级 (1=3x3, 2=4x4, 3=5x5, 4=6x6) - 连续值
        continuous_thinking: 思维是否连续 (0=连续/1-N, 1=不连续/1-99)
        click_disappear: 点击后数字是否消失 (0=否, 1=是)
        has_distractor: 是否有背景干扰 (0=否, 1=是)
        has_task_distractor: 是否有任务干扰 (0=否, 1=是)
    """
    grid_scale: int  # 1, 2, 3, 4
    continuous_thinking: int  # 0 或 1
    click_disappear: int  # 0 或 1
    has_distractor: int  # 0 或 1
    has_task_distractor: int  # 0 或 1

    @classmethod
    def from_task_config(cls, grid_size: int, number_range: Tuple[int, int],
                         click_disappear: bool, has_distractor: bool,
                         distractor_count: int) -> 'TaskCondition':
        """
        从原始 TaskConfig 创建 TaskCondition

        Args:
            grid_size: 方格数量 (9, 16, 25, 36)
            number_range: 数字范围，如 (1, 25) 或 (1, 99)
            click_disappear: 点击后数字是否消失
            has_distractor: 是否有干扰项
            distractor_count: 干扰项数量

        Returns:
            TaskCondition 实例
        """
        # 视野规模映射: 9→1, 16→2, 25→3, 36→4
        grid_to_scale = {9: 1, 16: 2, 25: 3, 36: 4}
        grid_scale = grid_to_scale.get(grid_size, 3)  # 默认为3 (5x5)

        # 思维连续性: (1, N) 表示连续，(1, 99) 表示不连续
        continuous_thinking = 0 if number_range[1] < 99 else 1

        return cls(
            grid_scale=grid_scale,
            continuous_thinking=continuous_thinking,
            click_disappear=int(click_disappear),
            has_distractor=int(has_distractor),
            has_task_distractor=int(distractor_count > 0),
        )

    def to_dict(self) -> Dict[str, int]:
        """转换为字典"""
        return {
            'grid_scale': self.grid_scale,
            'continuous_thinking': self.continuous_thinking,
            'click_disappear': self.click_disappear,
            'has_distractor': self.has_distractor,
            'has_task_distractor': self.has_task_distractor,
        }


class TaskEmbedding(nn.Module):
    """
    任务嵌入模块

    融合连续嵌入和离散嵌入，生成任务条件向量。

    设计说明:
    - 视野规模（grid_scale）: 连续嵌入，使用可学习向量 Emb * x
    - 其他4个维度: 离散嵌入，使用 nn.Embedding
    """

    def __init__(
        self,
        d_model: int,
        embedding_dim: int = 16,  # 每个离散嵌入的维度
    ):
        """
        初始化

        Args:
            d_model: 输出维度（与Transformer的d_model一致）
            embedding_dim: 每个离散嵌入的基础维度
        """
        super().__init__()

        self.d_model = d_model
        self.embedding_dim = embedding_dim

        # 连续嵌入：可学习的基础向量，与 grid_scale 相乘
        self.grid_base_emb = nn.Parameter(torch.randn(d_model))

        # 离散嵌入：4个维度，每个2类
        # 顺序: continuous_thinking, click_disappear, has_distractor, has_task_distractor
        self.discrete_emb = nn.ModuleList([
            nn.Embedding(2, embedding_dim),  # continuous_thinking
            nn.Embedding(2, embedding_dim),  # click_disappear
            nn.Embedding(2, embedding_dim),  # has_distractor
            nn.Embedding(2, embedding_dim),  # has_task_distractor
        ])

        # 离散嵌入投影到 d_model
        total_discrete_dim = embedding_dim * 4
        self.discrete_proj = nn.Linear(total_discrete_dim, d_model)

        # 融合后的投影
        self.fusion_proj = nn.Linear(d_model * 2, d_model)

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.grid_base_emb, mean=0.0, std=0.02)
        for emb in self.discrete_emb:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

    def forward(
        self,
        grid_scale: torch.Tensor,
        continuous_thinking: torch.Tensor,
        click_disappear: torch.Tensor,
        has_distractor: torch.Tensor,
        has_task_distractor: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            grid_scale: 视野规模等级 (batch,)
            continuous_thinking: 思维连续性 (batch,)
            click_disappear: 问题拆解 (batch,)
            has_distractor: 背景干扰 (batch,)
            has_task_distractor: 任务干扰 (batch,)

        Returns:
            task_emb: (batch, d_model) 任务嵌入向量
        """
        device = grid_scale.device

        # 1. 连续嵌入：grid_base_emb * grid_scale
        grid_continuous = grid_scale.unsqueeze(1).float()  # (batch, 1)
        grid_emb = self.grid_base_emb.unsqueeze(0) * grid_continuous  # (batch, d_model)

        # 2. 离散嵌入
        discrete_embs = [
            self.discrete_emb[0](continuous_thinking),
            self.discrete_emb[1](click_disappear),
            self.discrete_emb[2](has_distractor),
            self.discrete_emb[3](has_task_distractor),
        ]

        # 拼接所有离散嵌入
        discrete_emb = torch.cat(discrete_embs, dim=1)  # (batch, embedding_dim * 4)
        discrete_emb = self.discrete_proj(discrete_emb)  # (batch, d_model)

        # 3. 融合连续和离散嵌入
        combined = torch.cat([grid_emb, discrete_emb], dim=1)  # (batch, d_model * 2)
        task_emb = self.fusion_proj(combined)  # (batch, d_model)

        return task_emb
