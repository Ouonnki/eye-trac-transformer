# -*- coding: utf-8 -*-
"""
模型基类模块

定义统一的模型接口，所有模型类应继承 BaseModel 并实现 from_config 类方法。
"""

from abc import ABC, abstractmethod
from typing import Any

import torch.nn as nn

from src.config import UnifiedConfig
from src.models.dl_dataset import SequenceConfig


class BaseModel(nn.Module):
    """
    模型基类

    统一模型初始化接口，所有模型类应继承此类并实现 from_config。

    所有模型都应提供:
    1. from_config() 类方法：从 UnifiedConfig 创建模型实例
    2. 标准化的前向传播接口
    """

    def __init__(self):
        super().__init__()

    @classmethod
    @abstractmethod
    def from_config(
        cls,
        config: UnifiedConfig,
        seq_config: SequenceConfig,
        **kwargs: Any,
    ) -> 'BaseModel':
        """
        从配置创建模型实例

        Args:
            config: 统一配置对象，包含所有子配置（model, device 等）
            seq_config: 序列配置对象，包含数据相关参数
            **kwargs: 额外参数（如 num_classes）

        Returns:
            模型实例
        """
        raise NotImplementedError(f"{cls.__name__} 必须实现 from_config() 方法")
