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
        from src.models.task_embedding import TaskCondition

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
                    'config': trial.config,  # 保存原始任务配置
                    'task_condition': TaskCondition.from_task_config(
                        grid_size=trial.config.grid_size,
                        number_range=trial.config.number_range,
                        click_disappear=trial.config.click_disappear,
                        has_distractor=trial.config.has_distractor,
                        distractor_count=trial.config.distractor_count,
                    ),  # 保存规范化的任务条件
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
            'task_conditions': torch.from_numpy(self._get_task_conditions(subject_data)),
            'label': torch.tensor(subject_data['label'], dtype=torch.float32),
            'subject_id': subject_data['subject_id'],
        }

    def _get_task_conditions(self, subject_data: Dict) -> np.ndarray:
        """
        获取任务条件张量

        Args:
            subject_data: 被试数据字典

        Returns:
            (max_tasks, 5) 的任务条件数组
            每行格式: [grid_scale, continuous_thinking, click_disappear, has_distractor, has_task_distractor]
        """
        task_conditions = np.zeros((self.config.max_tasks, 5), dtype=np.int64)

        for t_idx, task_data in enumerate(subject_data['tasks']):
            if 'task_condition' in task_data and task_data['task_condition'] is not None:
                tc = task_data['task_condition']
                task_conditions[t_idx] = [
                    tc.grid_scale,
                    tc.continuous_thinking,
                    tc.click_disappear,
                    tc.has_distractor,
                    tc.has_task_distractor,
                ]

        return task_conditions


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    DataLoader的collate函数（被试级数据集）

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
        'task_conditions': torch.stack([b['task_conditions'] for b in batch]),
        'label': torch.stack([b['label'] for b in batch]),
        'subject_ids': [b['subject_id'] for b in batch],
    }


class SegmentGazeDataset(Dataset):
    """
    片段级眼动数据集

    将每个被试的每个片段作为独立样本。
    相比被试级数据集，样本量大幅增加（被试×任务×片段）。

    返回格式：
    - features: (max_seq_len, input_dim) 眼动序列
    - length: 实际序列长度
    - label: 被试标签（分数/类别）
    - subject_id: 被试ID（用于推理时聚合）
    - task_id: 任务ID（可选，用于添加任务嵌入）
    """

    def __init__(
        self,
        subjects: List[SubjectData],
        config: SequenceConfig,
        feature_extractor: Optional[SequenceFeatureExtractor] = None,
        fit_normalizer: bool = False,
    ):
        """
        初始化片段级数据集

        Args:
            subjects: 被试数据列表
            config: 序列配置
            feature_extractor: 特征提取器（如果为None则创建新的）
            fit_normalizer: 是否在当前数据上拟合归一化器
        """
        self.subjects = subjects
        self.config = config
        self.feature_extractor = feature_extractor or SequenceFeatureExtractor(config)

        # 预提取所有特征并展平为片段级
        self._precompute_segment_features()

        # 拟合归一化器
        if fit_normalizer:
            self._fit_normalizer()

    def _precompute_segment_features(self) -> None:
        """预计算所有被试的特征并展平为片段级"""
        self.segments = []
        self.segment_labels = []
        self.segment_subject_ids = []
        self.segment_task_ids = []

        for subject in self.subjects:
            label = subject.total_score
            subject_id = subject.subject_id

            for trial in subject.trials[:self.config.max_tasks]:
                task_id = trial.task_id

                for segment in trial.segments[:self.config.max_segments]:
                    # 提取序列特征
                    features = self.feature_extractor.extract(segment.gaze_points)

                    # 只保留有效片段（至少有2个点）
                    if len(features) >= 2:
                        self.segments.append(features)
                        self.segment_labels.append(label)
                        self.segment_subject_ids.append(subject_id)
                        self.segment_task_ids.append(task_id)

        logger.info(f'片段级数据集创建完成: {len(self.segments)} 个片段, '
                   f'{len(set(self.segment_subject_ids))} 个被试')

    def _fit_normalizer(self) -> None:
        """在当前数据上拟合归一化器"""
        all_features = [f for f in self.segments if len(f) > 0]
        if all_features:
            self.feature_extractor.fit_normalization(all_features)

            # 应用归一化
            for i, features in enumerate(self.segments):
                if len(features) > 0:
                    self.segments[i] = self.feature_extractor.normalize(features)

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        features = self.segments[idx]
        seq_len = min(len(features), self.config.max_seq_len)

        # 填充或截断到 max_seq_len
        padded = np.zeros((self.config.max_seq_len, self.config.input_dim), dtype=np.float32)
        if seq_len > 0:
            padded[:seq_len, :] = features[:seq_len]

        # 根据标签类型决定 dtype：整数用 long（分类），浮点用 float32（回归）
        label_value = self.segment_labels[idx]
        if isinstance(label_value, (int, np.integer)):
            label_dtype = torch.long
        else:
            label_dtype = torch.float32

        result = {
            'features': torch.from_numpy(padded),
            'length': seq_len,
            'label': torch.tensor(label_value, dtype=label_dtype),
            'subject_id': self.segment_subject_ids[idx],
            'task_id': self.segment_task_ids[idx],
        }

        # 添加任务条件（如果有）
        if hasattr(self, 'task_config_map'):
            task_id = self.segment_task_ids[idx]
            task_cond = self.task_config_map.get(task_id)
            if task_cond is not None:
                result['task_conditions'] = task_cond.to_dict()
            else:
                # 默认值
                result['task_conditions'] = {
                    'grid_scale': 3,
                    'continuous_thinking': 0,
                    'click_disappear': 0,
                    'has_distractor': 0,
                    'has_task_distractor': 0,
                }

        return result

    @classmethod
    def from_processed_data(
        cls,
        processed_data: List[Dict],
        config: SequenceConfig,
        normalizer_stats: Optional[Dict] = None,
    ) -> 'SegmentGazeDataset':
        """
        直接从预处理数据创建数据集（无需逆向转换）

        Args:
            processed_data: 预处理数据列表，每个元素包含:
                - subject_id: str
                - label: float
                - category: int
                - tasks: List[Dict] with 'task_id', 'segments', 'task_conditions'
            config: 序列配置
            normalizer_stats: 归一化统计量（可选）

        Returns:
            SegmentGazeDataset 实例
        """
        from src.models.task_embedding import TaskCondition

        segments = []
        segment_labels = []
        segment_subject_ids = []
        segment_task_ids = []
        # 任务条件映射：task_id -> TaskCondition
        task_config_map = {}

        for subject_dict in processed_data:
            subject_id = subject_dict['subject_id']
            # 分类任务使用 category（1/2/3 → 转换为 0/1/2），回归任务使用 label
            if 'category' in subject_dict:
                label = subject_dict['category'] - 1  # 转换为 0-based 索引
            else:
                label = subject_dict['label']

            for task_dict in subject_dict['tasks']:
                task_id = task_dict['task_id']

                # 构建任务条件映射
                if task_id not in task_config_map and 'task_conditions' in task_dict:
                    tc = task_dict['task_conditions']
                    task_config_map[task_id] = TaskCondition.from_task_config(
                        grid_size=tc['grid_size'],
                        number_range=tc['number_range'],
                        click_disappear=tc['click_disappear'],
                        has_distractor=tc['has_distractor'],
                        distractor_count=tc['distractor_count'],
                    )

                for segment_features in task_dict['segments']:
                    if len(segment_features) >= 2:  # 至少2个点
                        segments.append(segment_features)
                        segment_labels.append(label)
                        segment_subject_ids.append(subject_id)
                        segment_task_ids.append(task_id)

        # 创建数据集实例（绕过 __init__ 避免重新提取特征）
        dataset = cls.__new__(cls)
        dataset.config = config
        dataset.segments = segments
        dataset.segment_labels = segment_labels
        dataset.segment_subject_ids = segment_subject_ids
        dataset.segment_task_ids = segment_task_ids
        dataset.task_config_map = task_config_map  # 新增
        dataset.feature_extractor = SequenceFeatureExtractor(config)

        # 应用归一化统计量
        if normalizer_stats:
            dataset.feature_extractor.dt_mean = normalizer_stats['dt_mean']
            dataset.feature_extractor.dt_std = normalizer_stats['dt_std']
            dataset.feature_extractor.velocity_mean = normalizer_stats['velocity_mean']
            dataset.feature_extractor.velocity_std = normalizer_stats['velocity_std']
            dataset.feature_extractor.acceleration_mean = normalizer_stats['acceleration_mean']
            dataset.feature_extractor.acceleration_std = normalizer_stats['acceleration_std']

            # 应用归一化
            for i, features in enumerate(dataset.segments):
                if len(features) > 0:
                    dataset.segments[i] = dataset.feature_extractor.normalize(features)

        logger.info(f'片段级数据集创建完成: {len(dataset.segments)} 个片段, '
                   f'{len(set(dataset.segment_subject_ids))} 个被试')

        return dataset


def segment_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    片段级数据集的 DataLoader collate 函数

    Args:
        batch: 样本列表

    Returns:
        批次数据字典
    """
    labels = [b['label'] for b in batch]
    # 检查 label 是否为标量（0维），如果是则转换为 1 维张量
    if len(labels) > 0 and labels[0].dim() == 0:
        labels = [l.unsqueeze(0) for l in labels]

    result = {
        'features': torch.stack([b['features'] for b in batch]),
        'length': torch.tensor([b['length'] for b in batch], dtype=torch.long),
        'labels': torch.stack(labels),
        'subject_ids': [b['subject_id'] for b in batch],
        'task_ids': [b['task_id'] for b in batch],
    }

    # 处理任务条件（如果存在）
    if 'task_conditions' in batch[0]:
        task_conditions = {
            'grid_scale': torch.tensor([b['task_conditions']['grid_scale'] for b in batch]),
            'continuous_thinking': torch.tensor([b['task_conditions']['continuous_thinking'] for b in batch]),
            'click_disappear': torch.tensor([b['task_conditions']['click_disappear'] for b in batch]),
            'has_distractor': torch.tensor([b['task_conditions']['has_distractor'] for b in batch]),
            'has_task_distractor': torch.tensor([b['task_conditions']['has_task_distractor'] for b in batch]),
        }
        result['task_conditions'] = task_conditions

    return result
