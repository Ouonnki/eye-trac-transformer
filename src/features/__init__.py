# -*- coding: utf-8 -*-
"""特征工程模块（已移除传统特征工程，专注于深度学习的7维原始特征）"""

# 注意：传统特征工程模块（micro/meso/macro_features.py）已被移除
# 深度学习使用7维原始特征：(x, y, dt, velocity, acceleration, direction, direction_change)
# 这些特征由 src.models.dl_dataset.SequenceFeatureExtractor 提取

__all__ = []
