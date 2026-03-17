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
        continuous_thinking: 数字范围类型 (0=1-N, 1=1-99)
        click_disappear: 点击后数字是否消失 (0=否, 1=是)
        has_distractor: 是否有方格干扰项 (0=否, 1=是)
        has_task_distractor: 是否有数字干扰项 (0=否, 1=是)
    """
    grid_scale: int  # 1, 2, 3, 4
    continuous_thinking: int  # 0 或 1
    click_disappear: int  # 0 或 1
    has_distractor: int  # 0 或 1
    has_task_distractor: int  # 0 或 1

    @classmethod
    def from_task_config(cls, grid_size: int, number_range: Tuple[int, int],
                         click_disappear: bool, grid_distractor_count: int,
                         number_distractor_count: int) -> 'TaskCondition':
        """
        从原始 TaskConfig 创建 TaskCondition

        Args:
            grid_size: 方格数量 (9, 16, 25, 36)
            number_range: 数字范围，如 (1, 25) 或 (1, 99)
            click_disappear: 点击后数字是否消失
            grid_distractor_count: 方格干扰项数量
            number_distractor_count: 数字干扰项数量

        Returns:
            TaskCondition 实例
        """
        # 视野规模映射: 9→1, 16→2, 25→3, 36→4
        grid_to_scale = {9: 1, 16: 2, 25: 3, 36: 4}
        grid_scale = grid_to_scale.get(grid_size, 3)  # 默认为3 (5x5)

        # 数字范围类型: (1, 99) 表示 1-99；其他表示 1-N
        continuous_thinking = 1 if number_range[1] == 99 else 0

        return cls(
            grid_scale=grid_scale,
            continuous_thinking=continuous_thinking,
            click_disappear=int(click_disappear),
            has_distractor=int(grid_distractor_count > 0),
            has_task_distractor=int(number_distractor_count > 0),
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
    - 视野规模（grid_scale）: 连续嵌入，使用可学习向量 Emb * scale
    - 其他4个维度: 离散嵌入，使用 nn.Embedding
    - 所有五个条件使用统一的嵌入维度

    输出维度 = 5 * task_embedding_dim
    """

    def __init__(
        self,
        task_embedding_dim: int = 2,
    ):
        """
        初始化

        Args:
            task_embedding_dim: 所有条件统一的嵌入维度
        """
        super().__init__()

        self.task_embedding_dim = task_embedding_dim
        # 输出维度 = 5 * task_embedding_dim（1个连续 + 4个离散）
        self.output_dim = task_embedding_dim * 5

        # 连续嵌入：可学习的基础向量，与 grid_scale 相乘
        # grid_scale 范围是 1-4，可学习向量 shape 为 (task_embedding_dim,)
        self.grid_base_emb = nn.Parameter(torch.randn(task_embedding_dim))

        # 离散嵌入：4个维度，每个2类
        # 顺序: continuous_thinking, click_disappear, has_distractor, has_task_distractor
        self.discrete_emb = nn.ModuleList([
            nn.Embedding(2, task_embedding_dim),  # continuous_thinking
            nn.Embedding(2, task_embedding_dim),  # click_disappear
            nn.Embedding(2, task_embedding_dim),  # has_distractor
            nn.Embedding(2, task_embedding_dim),  # has_task_distractor
        ])

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
            task_emb: (batch, output_dim) 任务嵌入向量
                     output_dim = 5 * task_embedding_dim
        """
        # 1. 连续嵌入：grid_base_emb * grid_scale
        # grid_scale: (batch,) -> (batch, 1)
        grid_scale_f = grid_scale.unsqueeze(1).float()  # (batch, 1)
        # base_emb: (task_embedding_dim,) -> (1, task_embedding_dim)
        # result: (batch, task_embedding_dim)
        grid_emb = self.grid_base_emb.unsqueeze(0) * grid_scale_f

        # 2. 离散嵌入
        discrete_embs = [
            self.discrete_emb[0](continuous_thinking),
            self.discrete_emb[1](click_disappear),
            self.discrete_emb[2](has_distractor),
            self.discrete_emb[3](has_task_distractor),
        ]

        # 拼接所有嵌入：连续(task_embedding_dim) + 离散(4*task_embedding_dim) = 5*task_embedding_dim
        task_emb = torch.cat([grid_emb] + discrete_embs, dim=1)

        return task_emb


class MLPTaskEmbedding(nn.Module):
    """
    MLP 任务嵌入模块

    将5个任务条件通过 MLP 映射到嵌入空间，自动学习条件间的交互效应。
    相比独立嵌入+拼接的方式，MLP 能捕捉如"大格+干扰项"的组合难度。

    输入: 5维任务条件向量（grid_scale 归一化到 [0.25,1.0]，其余为 0/1）
    输出: (batch, output_dim) 嵌入向量
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 32,
        output_dim: int = 16,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, task_conditions: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            task_conditions: (batch, 5) 任务条件
                [grid_scale(1-4), continuous_thinking(0/1),
                 click_disappear(0/1), has_distractor(0/1), has_task_distractor(0/1)]

        Returns:
            (batch, output_dim) 任务嵌入向量
        """
        x = task_conditions.float()
        # 归一化 grid_scale (1-4) 到 [0.25, 1.0]，与二值特征量纲对齐
        x = torch.cat([x[:, 0:1] / 4.0, x[:, 1:]], dim=1)
        return self.mlp(x)
