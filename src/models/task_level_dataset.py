# -*- coding: utf-8 -*-
"""
任务级眼动数据集模块

每个样本 = 一个被试的一个任务
输入: 25个片段的眼动序列
输出: 任务级分类标签 (0/1/2)
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from src.models.dl_dataset import SequenceConfig, SequenceFeatureExtractor

logger = logging.getLogger(__name__)


@dataclass
class TaskLevelSequenceConfig:
    """任务级序列配置"""
    max_seq_len: int = 300      # 每个片段最大眼动点数
    max_segments: int = 25      # 每个任务最大片段数
    screen_width: int = 1920
    screen_height: int = 1080
    input_dim: int = 7


class TaskLevelGazeDataset(Dataset):
    """
    任务级眼动数据集
    
    每个样本 = 一个被试的一个任务
    返回格式：
    - segments: (N, 300, 7) 眼动序列（N为该任务片段数，不做截断）
    - segment_mask: (N,) 有效片段掩码
    - segment_seq_mask: (N, 300) 片段内有效时步掩码
    - task_conditions: (5,) 任务条件
    - label: 任务级分类标签 (0/1/2)
    """

    def __init__(
        self,
        processed_data: List[Dict],
        config: TaskLevelSequenceConfig,
        feature_extractor: Optional[SequenceFeatureExtractor] = None,
        fit_normalizer: bool = False,
    ):
        self.config = config
        self.feature_extractor = feature_extractor or SequenceFeatureExtractor(
            SequenceConfig(
                max_seq_len=config.max_seq_len,
                max_tasks=1,  # 任务级不需要任务维度
                max_segments=config.max_segments,
                screen_width=config.screen_width,
                screen_height=config.screen_height,
                input_dim=config.input_dim,
            )
        )

        # 数据增强（仅对 training_indices 中的样本生效）
        self.augmentation = None
        self.training_indices: set = set()

        # 构建任务级样本列表
        self.samples = self._build_samples(processed_data)

        if fit_normalizer:
            self._fit_normalizer()

        logger.info(f"任务级数据集创建完成: {len(self.samples)} 个样本")

    def _build_samples(self, processed_data: List[Dict]) -> List[Dict]:
        """将预处理数据展开为任务级样本"""
        samples = []
        
        for subject_data in processed_data:
            subject_id = subject_data['subject_id']
            
            for task_data in subject_data['tasks']:
                # 只保留有任务级标签的数据
                if 'task_label' not in task_data:
                    continue
                    
                samples.append({
                    'subject_id': subject_id,
                    'task_id': task_data['task_id'],
                    'segments': task_data['segments'],
                    'task_conditions': task_data.get('task_conditions', {}),
                    'label': task_data['task_label'],
                })
                
        return samples

    def _fit_normalizer(self) -> None:
        """拟合归一化器"""
        all_features = []
        for sample in self.samples:
            for seg in sample['segments']:
                if len(seg) > 0:
                    all_features.append(seg)
        
        self.feature_extractor.fit_normalization(all_features)
        
        # 应用归一化
        for sample in self.samples:
            for i, seg in enumerate(sample['segments']):
                if len(seg) > 0:
                    sample['segments'][i] = self.feature_extractor.normalize(seg)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # 使用真实片段数，避免截断
        num_segments = len(sample['segments'])
        segments = np.zeros(
            (num_segments, self.config.max_seq_len, self.config.input_dim),
            dtype=np.float32
        )
        segment_mask = np.zeros(num_segments, dtype=np.bool_)
        segment_seq_mask = np.zeros((num_segments, self.config.max_seq_len), dtype=np.bool_)

        # 填充片段数据
        for s_idx, seg_features in enumerate(sample['segments']):
            seq_len = min(len(seg_features), self.config.max_seq_len)
            if seq_len > 0:
                segments[s_idx, :seq_len] = seg_features[:seq_len]
                segment_mask[s_idx] = True
                segment_seq_mask[s_idx, :seq_len] = True

        # 数据增强（仅训练集样本）
        if self.augmentation is not None and idx in self.training_indices:
            segments, segment_mask, segment_seq_mask = self.augmentation(
                segments, segment_mask, segment_seq_mask
            )

        # 任务条件 (5维)
        tc = sample['task_conditions']
        number_range_max = tc.get('number_range', (1, 25))[1]
        continuous_thinking = 1 if number_range_max == 99 else 0
        task_conditions = np.array([
            self._get_grid_scale(tc.get('grid_size', 25)),
            continuous_thinking,
            int(tc.get('click_disappear', False)),
            int(tc.get('grid_distractor_count', 0) > 0),
            int(tc.get('number_distractor_count', 0) > 0),
        ], dtype=np.int64)
        
        return {
            'segments': torch.from_numpy(segments),
            'segment_mask': torch.from_numpy(segment_mask),
            'segment_seq_mask': torch.from_numpy(segment_seq_mask),
            'task_conditions': torch.from_numpy(task_conditions),
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'subject_id': sample['subject_id'],
            'task_id': sample['task_id'],
        }

    @staticmethod
    def _get_grid_scale(grid_size: int) -> int:
        """网格大小映射到视野规模等级"""
        grid_to_scale = {9: 1, 16: 2, 25: 3, 36: 4}
        return grid_to_scale.get(grid_size, 3)


def task_level_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    任务级数据集的collate函数
    """
    if not batch:
        return {
            'segments': torch.empty(0),
            'segment_mask': torch.empty(0, dtype=torch.bool),
            'segment_seq_mask': torch.empty(0, dtype=torch.bool),
            'task_conditions': torch.empty(0, dtype=torch.long),
            'labels': torch.empty(0, dtype=torch.long),
            'subject_ids': [],
            'task_ids': [],
        }

    batch_size = len(batch)
    max_segments = max(b['segments'].shape[0] for b in batch)
    max_seq_len = batch[0]['segments'].shape[1]
    input_dim = batch[0]['segments'].shape[2]

    segments = torch.zeros(
        (batch_size, max_segments, max_seq_len, input_dim),
        dtype=batch[0]['segments'].dtype
    )
    segment_mask = torch.zeros((batch_size, max_segments), dtype=torch.bool)
    segment_seq_mask = torch.zeros((batch_size, max_segments, max_seq_len), dtype=torch.bool)

    for i, item in enumerate(batch):
        segs = item['segments']
        n = segs.shape[0]
        if n > 0:
            segments[i, :n] = segs
            segment_mask[i, :n] = item['segment_mask']
            segment_seq_mask[i, :n] = item['segment_seq_mask']

    return {
        'segments': segments,
        'segment_mask': segment_mask,
        'segment_seq_mask': segment_seq_mask,
        'task_conditions': torch.stack([b['task_conditions'] for b in batch]),
        'labels': torch.stack([b['label'] for b in batch]),
        'subject_ids': [b['subject_id'] for b in batch],
        'task_ids': [b['task_id'] for b in batch],
    }
