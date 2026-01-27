# -*- coding: utf-8 -*-
"""
深度学习数据集模块

提供用于Transformer模型的层级眼动数据集类。
"""

import math
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.schemas import SubjectData, TaskTrial, SearchSegment, GazePoint

logger = logging.getLogger(__name__)


@dataclass
class SequenceConfig:
    """序列配置（数据属性，由数据本身决定）"""
    max_seq_len: int = 100      # 每个片段最大眼动点数
    max_tasks: int = 30         # 每个被试最大任务数
    max_segments: int = 30      # 每个任务最大片段数
    screen_width: int = 1920    # 屏幕宽度
    screen_height: int = 1080   # 屏幕高度
    input_dim: int = 7          # 输入特征维度


class SequenceFeatureExtractor:
    """
    序列特征提取器

    从原始眼动点序列提取7维增强特征：
    - x: 归一化X坐标 [0, 1]
    - y: 归一化Y坐标 [0, 1]
    - dt: 时间差（毫秒，归一化）
    - velocity: 瞬时速度（归一化）
    - acceleration: 瞬时加速度（归一化）
    - direction: 移动方向（弧度，归一化到[-1, 1]）
    - direction_change: 方向变化量（归一化）
    """

    def __init__(self, config: SequenceConfig):
        self.config = config
        # 归一化统计量（将在训练时计算）
        self.velocity_mean = 0.0
        self.velocity_std = 1.0
        self.acceleration_mean = 0.0
        self.acceleration_std = 1.0
        self.dt_mean = 0.0
        self.dt_std = 1.0

    def extract(self, gaze_points: List[GazePoint]) -> np.ndarray:
        """
        从眼动点序列提取7维特征

        Args:
            gaze_points: 眼动点列表

        Returns:
            (seq_len, 7) 的特征数组
        """
        n = len(gaze_points)
        if n < 2:
            # 返回空序列（后续会填充）
            return np.zeros((0, 7), dtype=np.float32)

        features = np.zeros((n, 7), dtype=np.float32)

        prev_velocity = 0.0
        prev_direction = 0.0

        for i, point in enumerate(gaze_points):
            # 基础坐标归一化
            features[i, 0] = point.x / self.config.screen_width
            features[i, 1] = point.y / self.config.screen_height

            if i == 0:
                # 第一个点
                features[i, 2] = 0.0  # dt
                features[i, 3] = 0.0  # velocity
                features[i, 4] = 0.0  # acceleration
                features[i, 5] = 0.0  # direction
                features[i, 6] = 0.0  # direction_change
            else:
                prev_point = gaze_points[i - 1]

                # 时间差（毫秒）
                dt = (point.timestamp - prev_point.timestamp).total_seconds() * 1000
                dt = max(dt, 1.0)  # 避免除零
                features[i, 2] = dt

                # 位移
                dx = point.x - prev_point.x
                dy = point.y - prev_point.y
                distance = math.sqrt(dx**2 + dy**2)

                # 速度
                velocity = distance / dt
                features[i, 3] = velocity

                # 加速度
                acceleration = (velocity - prev_velocity) / dt
                features[i, 4] = acceleration

                # 方向（弧度）
                direction = math.atan2(dy, dx)  # [-π, π]
                features[i, 5] = direction / math.pi  # 归一化到[-1, 1]

                # 方向变化
                if i > 1:
                    direction_change = abs(direction - prev_direction)
                    # 处理跨越±π的情况
                    if direction_change > math.pi:
                        direction_change = 2 * math.pi - direction_change
                    features[i, 6] = direction_change / math.pi  # 归一化到[0, 1]

                prev_velocity = velocity
                prev_direction = direction

        return features

    def fit_normalization(self, all_features: List[np.ndarray]) -> None:
        """
        计算归一化统计量

        Args:
            all_features: 所有序列特征的列表
        """
        # 收集所有非零的dt、velocity、acceleration
        all_dt = []
        all_velocity = []
        all_acceleration = []

        for features in all_features:
            if len(features) > 1:
                all_dt.extend(features[1:, 2].tolist())
                all_velocity.extend(features[1:, 3].tolist())
                all_acceleration.extend(features[1:, 4].tolist())

        if all_dt:
            self.dt_mean = np.mean(all_dt)
            self.dt_std = np.std(all_dt) + 1e-8
        if all_velocity:
            self.velocity_mean = np.mean(all_velocity)
            self.velocity_std = np.std(all_velocity) + 1e-8
        if all_acceleration:
            self.acceleration_mean = np.mean(all_acceleration)
            self.acceleration_std = np.std(all_acceleration) + 1e-8

    def normalize(self, features: np.ndarray) -> np.ndarray:
        """
        应用归一化

        Args:
            features: (seq_len, 7) 的特征数组

        Returns:
            归一化后的特征数组
        """
        normalized = features.copy()
        if len(features) > 1:
            # 归一化dt、velocity、acceleration
            normalized[1:, 2] = (features[1:, 2] - self.dt_mean) / self.dt_std
            normalized[1:, 3] = (features[1:, 3] - self.velocity_mean) / self.velocity_std
            normalized[1:, 4] = (features[1:, 4] - self.acceleration_mean) / self.acceleration_std
        return normalized


class HierarchicalGazeDataset(Dataset):
    """
    层级眼动数据集

    组织结构：被试 → 任务 → 片段 → 眼动序列

    返回格式：
    - segments: (max_tasks, max_segments, max_seq_len, 7) 眼动序列
    - segment_lengths: (max_tasks, max_segments) 每个片段的实际长度
    - segment_mask: (max_tasks, max_segments) 有效片段掩码
    - task_lengths: (max_tasks,) 每个任务的有效片段数
    - task_mask: (max_tasks,) 有效任务掩码
    - label: 被试总分
    """

    def __init__(
        self,
        subjects: List[SubjectData],
        config: SequenceConfig,
        feature_extractor: Optional[SequenceFeatureExtractor] = None,
        fit_normalizer: bool = False,
    ):
        """
        初始化数据集

        Args:
            subjects: 被试数据列表
            config: 序列配置
            feature_extractor: 特征提取器（如果为None则创建新的）
            fit_normalizer: 是否在当前数据上拟合归一化器
        """
        self.subjects = subjects
        self.config = config
        self.feature_extractor = feature_extractor or SequenceFeatureExtractor(config)

        # 预提取所有特征
        self._precompute_features()

        # 拟合归一化器
        if fit_normalizer:
            self._fit_normalizer()

    def _precompute_features(self) -> None:
        """预计算所有被试的特征"""
        self.all_subject_features = []

        for subject in self.subjects:
            subject_data = {
                'subject_id': subject.subject_id,
                'label': subject.total_score,
                'tasks': []
            }

            for trial in subject.trials[:self.config.max_tasks]:
                task_data = {
                    'task_id': trial.task_id,
                    'segments': []
                }

                for segment in trial.segments[:self.config.max_segments]:
                    # 提取序列特征
                    features = self.feature_extractor.extract(segment.gaze_points)
                    task_data['segments'].append(features)

                subject_data['tasks'].append(task_data)

            self.all_subject_features.append(subject_data)

    def _fit_normalizer(self) -> None:
        """在当前数据上拟合归一化器"""
        all_features = []
        for subject_data in self.all_subject_features:
            for task_data in subject_data['tasks']:
                for features in task_data['segments']:
                    if len(features) > 0:
                        all_features.append(features)

        self.feature_extractor.fit_normalization(all_features)

        # 应用归一化
        for subject_data in self.all_subject_features:
            for task_data in subject_data['tasks']:
                for i, features in enumerate(task_data['segments']):
                    if len(features) > 0:
                        task_data['segments'][i] = self.feature_extractor.normalize(features)

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        subject_data = self.all_subject_features[idx]

        # 初始化张量
        segments = np.zeros(
            (self.config.max_tasks, self.config.max_segments,
             self.config.max_seq_len, self.config.input_dim),
            dtype=np.float32
        )
        segment_lengths = np.zeros(
            (self.config.max_tasks, self.config.max_segments),
            dtype=np.int64
        )
        segment_mask = np.zeros(
            (self.config.max_tasks, self.config.max_segments),
            dtype=np.bool_
        )
        task_lengths = np.zeros(self.config.max_tasks, dtype=np.int64)
        task_mask = np.zeros(self.config.max_tasks, dtype=np.bool_)

        # 填充数据
        num_tasks = len(subject_data['tasks'])
        for t_idx, task_data in enumerate(subject_data['tasks']):
            num_segments = len(task_data['segments'])
            task_lengths[t_idx] = num_segments
            task_mask[t_idx] = True

            for s_idx, features in enumerate(task_data['segments']):
                seq_len = min(len(features), self.config.max_seq_len)
                if seq_len > 0:
                    segments[t_idx, s_idx, :seq_len, :] = features[:seq_len]
                    segment_lengths[t_idx, s_idx] = seq_len
                    segment_mask[t_idx, s_idx] = True

        return {
            'segments': torch.from_numpy(segments),
            'segment_lengths': torch.from_numpy(segment_lengths),
            'segment_mask': torch.from_numpy(segment_mask),
            'task_lengths': torch.from_numpy(task_lengths),
            'task_mask': torch.from_numpy(task_mask),
            'label': torch.tensor(subject_data['label'], dtype=torch.float32),
            'subject_id': subject_data['subject_id'],
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    DataLoader的collate函数

    Args:
        batch: 样本列表

    Returns:
        批次数据字典
    """
    return {
        'segments': torch.stack([b['segments'] for b in batch]),
        'segment_lengths': torch.stack([b['segment_lengths'] for b in batch]),
        'segment_mask': torch.stack([b['segment_mask'] for b in batch]),
        'task_lengths': torch.stack([b['task_lengths'] for b in batch]),
        'task_mask': torch.stack([b['task_mask'] for b in batch]),
        'label': torch.stack([b['label'] for b in batch]),
        'subject_ids': [b['subject_id'] for b in batch],
    }
