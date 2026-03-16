# -*- coding: utf-8 -*-
"""
眼动数据增强模块

提供 5 种针对眼动轨迹时序数据的增强方法：
1. Gaussian Noise  - 空间坐标抖动
2. Temporal Scaling - 时间维度缩放
3. Segment Dropout  - 片段随机丢弃
4. Time Masking     - 时间遮蔽 (类 SpecAugment)
5. Feature Masking  - 特征通道遮蔽

所有增强仅在训练时启用，通过参数值 > 0 开启，= 0 时跳过。
"""

import numpy as np


class GazeAugmentation:
    """
    眼动轨迹数据增强

    对 (N, max_seq_len, 7) 格式的 segment 数据进行在线增强。
    7 维特征: [x, y, dt, velocity, acceleration, direction, direction_change]

    Args:
        config: 增强配置字典，支持以下参数:
            noise_std (float): 空间坐标高斯噪声标准差，默认 0.0（关闭）
            time_scale_range (float): 时间缩放范围，默认 0.0（关闭）
                实际缩放因子 ~ Uniform[1-range, 1+range]
            segment_drop_prob (float): 片段丢弃概率，默认 0.0（关闭）
            time_mask_max_len (int): 时间遮蔽最大长度，默认 0（关闭）
            feature_mask_prob (float): 特征通道遮蔽概率，默认 0.0（关闭）
    """

    def __init__(self, config: dict):
        self.noise_std = config.get('noise_std', 0.0)
        self.time_scale_range = config.get('time_scale_range', 0.0)
        self.segment_drop_prob = config.get('segment_drop_prob', 0.0)
        self.time_mask_max_len = config.get('time_mask_max_len', 0)
        self.feature_mask_prob = config.get('feature_mask_prob', 0.0)

    def __call__(self, segments, segment_mask, segment_seq_mask):
        """
        对一个样本的所有 segments 应用增强。

        Args:
            segments: np.ndarray (N, max_seq_len, 7) float32
            segment_mask: np.ndarray (N,) bool - 有效片段掩码
            segment_seq_mask: np.ndarray (N, max_seq_len) bool - 有效时步掩码

        Returns:
            增强后的 (segments, segment_mask, segment_seq_mask)
        """
        # 复制避免修改原始数据
        segments = segments.copy()
        segment_mask = segment_mask.copy()
        segment_seq_mask = segment_seq_mask.copy()

        # 按固定顺序依次应用增强
        if self.segment_drop_prob > 0:
            segments, segment_mask, segment_seq_mask = self._segment_dropout(
                segments, segment_mask, segment_seq_mask
            )

        if self.time_mask_max_len > 0:
            segments, segment_seq_mask = self._time_masking(
                segments, segment_mask, segment_seq_mask
            )

        if self.feature_mask_prob > 0:
            segments = self._feature_masking(segments, segment_mask)

        if self.time_scale_range > 0:
            segments = self._temporal_scaling(segments, segment_mask, segment_seq_mask)

        if self.noise_std > 0:
            segments = self._gaussian_noise(segments, segment_seq_mask)

        return segments, segment_mask, segment_seq_mask

    def _gaussian_noise(self, segments, segment_seq_mask):
        """对 x, y 坐标添加高斯噪声"""
        N, T, _ = segments.shape
        noise = np.random.randn(N, T, 2).astype(np.float32) * self.noise_std
        # 仅对有效时步添加噪声
        mask = segment_seq_mask[:, :, np.newaxis]  # (N, T, 1)
        segments[:, :, :2] += noise * mask
        return segments

    def _temporal_scaling(self, segments, segment_mask, segment_seq_mask):
        """对 dt/velocity/acceleration 进行时间缩放"""
        N = segments.shape[0]
        for i in range(N):
            if not segment_mask[i]:
                continue
            scale = np.random.uniform(
                1.0 - self.time_scale_range,
                1.0 + self.time_scale_range,
            )
            # 对有效时步的 dt(2), velocity(3), acceleration(4) 乘以缩放因子
            valid = segment_seq_mask[i]  # (T,)
            segments[i, valid, 2:5] *= scale
        return segments

    def _segment_dropout(self, segments, segment_mask, segment_seq_mask):
        """随机丢弃片段，保证至少保留 1 个"""
        valid_indices = np.where(segment_mask)[0]
        if len(valid_indices) <= 1:
            return segments, segment_mask, segment_seq_mask

        drop_mask = np.random.rand(len(valid_indices)) < self.segment_drop_prob
        # 保证至少保留 1 个
        if drop_mask.all():
            keep_idx = np.random.randint(len(valid_indices))
            drop_mask[keep_idx] = False

        for j, idx in enumerate(valid_indices):
            if drop_mask[j]:
                segment_mask[idx] = False
                segment_seq_mask[idx, :] = False
                segments[idx, :, :] = 0.0

        return segments, segment_mask, segment_seq_mask

    def _time_masking(self, segments, segment_mask, segment_seq_mask):
        """随机遮蔽连续时间窗口"""
        N, T, _ = segments.shape
        for i in range(N):
            if not segment_mask[i]:
                continue
            valid_len = int(segment_seq_mask[i].sum())
            if valid_len <= 1:
                continue

            mask_len = np.random.randint(0, min(self.time_mask_max_len, valid_len - 1) + 1)
            if mask_len == 0:
                continue

            start = np.random.randint(0, valid_len - mask_len + 1)
            segments[i, start:start + mask_len, :] = 0.0
            segment_seq_mask[i, start:start + mask_len] = False

        return segments, segment_seq_mask

    def _feature_masking(self, segments, segment_mask):
        """随机遮蔽特征通道"""
        N = segments.shape[0]
        num_features = segments.shape[2]  # 7
        for i in range(N):
            if not segment_mask[i]:
                continue
            for f in range(num_features):
                if np.random.rand() < self.feature_mask_prob:
                    segments[i, :, f] = 0.0
        return segments
