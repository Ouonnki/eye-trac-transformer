# -*- coding: utf-8 -*-
"""深度学习模型模块"""

# 注意力机制
from .attention import (
    AttentionPooling,
    MultiHeadAttentionPooling,
    PositionalEncoding,
)

# 深度学习数据集
from .dl_dataset import (
    SequenceFeatureExtractor,
    HierarchicalGazeDataset,
    LightweightGazeDataset,
)

# 深度学习模型
from .dl_models import (
    GazeTransformerEncoder,
    TaskTransformerEncoder,
    HierarchicalTransformerNetwork,
    DomainAdaptiveHierarchicalNetwork,
)

# 深度学习训练器
from .dl_trainer import (
    DeepLearningTrainer,
    TrainingConfig,
    EarlyStopping,
)

# 域迁移组件
from .domain_adaptation import (
    GradientReversalFunction,
    ClassCenterBank,
    DomainDiscriminator,
    DistanceAwareDomainDiscriminator,
)

# 域迁移训练器
from .domain_trainer import (
    DomainAdaptiveTrainer,
)

__all__ = [
    # 注意力机制
    'AttentionPooling',
    'MultiHeadAttentionPooling',
    'PositionalEncoding',
    # 数据集
    'SequenceFeatureExtractor',
    'HierarchicalGazeDataset',
    'LightweightGazeDataset',
    # 模型
    'GazeTransformerEncoder',
    'TaskTransformerEncoder',
    'HierarchicalTransformerNetwork',
    'DomainAdaptiveHierarchicalNetwork',
    # 训练器
    'DeepLearningTrainer',
    'TrainingConfig',
    'EarlyStopping',
    # 域迁移
    'GradientReversalFunction',
    'ClassCenterBank',
    'DomainDiscriminator',
    'DistanceAwareDomainDiscriminator',
    'DomainAdaptiveTrainer',
]
