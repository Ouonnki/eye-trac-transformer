# -*- coding: utf-8 -*-
"""
中观特征提取模块 - Level 2（任务级）

基于微观特征聚合计算任务级特征。

数据来源：
- 点击事件（sheet3）：用于基础搜索时间和距离统计
- 连续眼动轨迹（sheet4）：用于路径效率、注视熵、速度等眼动特征
"""

import logging
import math
from typing import Dict, List, Optional

import numpy as np

from src.data.schemas import TaskTrial, SearchSegment, MesoFeatures

logger = logging.getLogger(__name__)


class MesoFeatureExtractor:
    """
    中观特征提取器

    基于微观特征聚合计算任务级特征：
    - 搜索时间统计：总时长、均值、标准差、最大/最小值
    - 移动距离统计：总距离、均值、标准差
    - 搜索速度统计：均值、标准差、变异系数
    - 空间特征：象限变化比例、中心距离统计
    - 眼动特征：路径效率、注视熵、方向变化、犹豫时间等
    - 任务配置：网格大小、干扰项等
    """

    def __init__(self, target_tolerance: float = 100.0):
        """
        初始化提取器

        Args:
            target_tolerance: 目标区域容差（用于接近判断）
        """
        self.target_tolerance = target_tolerance

    def extract(
        self,
        trial: TaskTrial,
        micro_features: List[Dict[str, float]],
    ) -> Dict[str, float]:
        """
        提取任务级特征

        Args:
            trial: 任务试次
            micro_features: 该任务所有片段的微观特征列表

        Returns:
            任务级特征字典
        """
        if not micro_features:
            return self._empty_features(trial)

        features = {}

        # 搜索时间统计
        search_times = [f.get('search_time_ms', f.get('duration_ms', 0)) for f in micro_features]
        features['total_search_time'] = float(sum(search_times))
        features['search_time_mean'] = float(np.mean(search_times))
        features['search_time_std'] = float(np.std(search_times))
        features['search_time_max'] = float(np.max(search_times))
        features['search_time_min'] = float(np.min(search_times))
        features['search_time_median'] = float(np.median(search_times))

        # 搜索时间变异系数（衡量稳定性）
        if features['search_time_mean'] > 0:
            features['search_time_cv'] = features['search_time_std'] / features['search_time_mean']
        else:
            features['search_time_cv'] = 0.0

        # 移动距离统计
        move_distances = [f.get('move_distance', f.get('path_length', 0)) for f in micro_features]
        features['total_move_distance'] = float(sum(move_distances))
        features['move_distance_mean'] = float(np.mean(move_distances))
        features['move_distance_std'] = float(np.std(move_distances))
        features['move_distance_max'] = float(np.max(move_distances))
        features['move_distance_min'] = float(np.min(move_distances))

        # 搜索速度统计
        search_speeds = [f.get('search_speed', f.get('velocity_mean', 0)) for f in micro_features]
        features['search_speed_mean'] = float(np.mean(search_speeds))
        features['search_speed_std'] = float(np.std(search_speeds))
        if features['search_speed_mean'] > 0:
            features['search_speed_cv'] = features['search_speed_std'] / features['search_speed_mean']
        else:
            features['search_speed_cv'] = 0.0

        # 空间特征
        if 'quadrant_change' in micro_features[0]:
            quadrant_changes = [f['quadrant_change'] for f in micro_features]
            features['quadrant_change_ratio'] = float(np.mean(quadrant_changes))
        else:
            features['quadrant_change_ratio'] = 0.0

        if 'distance_from_center' in micro_features[0]:
            center_distances = [f['distance_from_center'] for f in micro_features]
            features['center_distance_mean'] = float(np.mean(center_distances))
            features['center_distance_std'] = float(np.std(center_distances))
        else:
            features['center_distance_mean'] = 0.0
            features['center_distance_std'] = 0.0

        # 归一化距离
        if 'normalized_distance' in micro_features[0]:
            norm_dists = [f['normalized_distance'] for f in micro_features]
            features['normalized_distance_mean'] = float(np.mean(norm_dists))
        else:
            features['normalized_distance_mean'] = 0.0

        # 回访统计
        revisit_counts = [f.get('revisit_area_count', f.get('revisit_count', 0)) for f in micro_features]
        features['revisit_total'] = float(sum(revisit_counts))
        features['revisit_ratio'] = features['revisit_total'] / len(micro_features) if micro_features else 0.0

        # 任务配置特征
        features['has_distractor'] = float(trial.config.has_distractor)
        features['grid_size'] = float(trial.config.grid_size)
        features['distractor_count'] = float(trial.config.distractor_count)
        features['click_disappear'] = float(trial.config.click_disappear)

        # 片段统计
        features['segment_count'] = float(len(micro_features))
        features['click_count'] = float(len(micro_features) + 1)  # 点击数 = 片段数 + 1

        # 效率指标：完成速度（点击数/总时间）
        if features['total_search_time'] > 0:
            features['completion_rate'] = features['click_count'] / (features['total_search_time'] / 1000)  # 点击/秒
        else:
            features['completion_rate'] = 0.0

        # 搜索效率：距离/时间的标准化
        if features['total_search_time'] > 0:
            features['distance_time_ratio'] = features['total_move_distance'] / features['total_search_time']
        else:
            features['distance_time_ratio'] = 0.0

        # ===== 眼动特征聚合 =====
        # 检查是否有眼动数据
        has_gaze = any(f.get('has_gaze_data', 0) > 0 for f in micro_features)
        features['has_gaze_data'] = float(has_gaze)

        # 路径效率统计
        path_efficiencies = [f.get('path_efficiency', 1.0) for f in micro_features]
        features['path_efficiency_mean'] = float(np.mean(path_efficiencies))
        features['path_efficiency_std'] = float(np.std(path_efficiencies))
        features['path_efficiency_min'] = float(np.min(path_efficiencies))

        # 注视熵统计
        fixation_entropies = [f.get('fixation_entropy', 0.0) for f in micro_features]
        features['fixation_entropy_mean'] = float(np.mean(fixation_entropies))
        features['fixation_entropy_std'] = float(np.std(fixation_entropies))
        features['fixation_entropy_max'] = float(np.max(fixation_entropies))

        # 速度统计（眼动速度）
        velocity_means = [f.get('velocity_mean', 0.0) for f in micro_features]
        velocity_stds = [f.get('velocity_std', 0.0) for f in micro_features]
        velocity_maxs = [f.get('velocity_max', 0.0) for f in micro_features]
        features['gaze_velocity_mean'] = float(np.mean(velocity_means))
        features['gaze_velocity_std_mean'] = float(np.mean(velocity_stds))
        features['gaze_velocity_max_mean'] = float(np.mean(velocity_maxs))
        features['gaze_velocity_max_max'] = float(np.max(velocity_maxs))

        # 方向变化统计
        direction_changes = [f.get('direction_changes', 0) for f in micro_features]
        features['direction_changes_total'] = float(sum(direction_changes))
        features['direction_changes_mean'] = float(np.mean(direction_changes))
        features['direction_changes_max'] = float(np.max(direction_changes))

        # 犹豫时间统计
        hesitation_times = [f.get('hesitation_time', 0.0) for f in micro_features]
        features['hesitation_time_total'] = float(sum(hesitation_times))
        features['hesitation_time_mean'] = float(np.mean(hesitation_times))
        features['hesitation_time_max'] = float(np.max(hesitation_times))

        # 犹豫时间占比
        if features['total_search_time'] > 0:
            features['hesitation_ratio'] = features['hesitation_time_total'] / features['total_search_time']
        else:
            features['hesitation_ratio'] = 0.0

        # 眼动点数统计
        gaze_point_counts = [f.get('gaze_point_count', 0) for f in micro_features]
        features['gaze_point_total'] = float(sum(gaze_point_counts))
        features['gaze_point_mean'] = float(np.mean(gaze_point_counts))

        # 路径长度统计
        path_lengths = [f.get('path_length', 0.0) for f in micro_features]
        features['path_length_total'] = float(sum(path_lengths))
        features['path_length_mean'] = float(np.mean(path_lengths))
        features['path_length_std'] = float(np.std(path_lengths))

        # 加速度统计
        accelerations = [f.get('acceleration_mean', 0.0) for f in micro_features]
        features['acceleration_mean'] = float(np.mean(accelerations))
        features['acceleration_max'] = float(np.max(accelerations))

        # 兼容旧特征名称（为 macro 特征提取器使用）
        features['efficiency_mean'] = features['search_speed_mean']
        features['entropy_mean'] = features['fixation_entropy_mean']  # 使用实际眼动熵
        features['total_duration'] = features['total_search_time']
        features['revisit_mean'] = features['revisit_ratio']
        features['velocity_mean'] = features['gaze_velocity_mean']
        features['distraction_cost'] = 0.0

        return features

    def extract_as_dataclass(
        self,
        trial: TaskTrial,
        micro_features: List[Dict[str, float]],
    ) -> MesoFeatures:
        """
        提取任务级特征（返回数据类）
        """
        features_dict = self.extract(trial, micro_features)
        return MesoFeatures(
            efficiency_mean=features_dict['efficiency_mean'],
            efficiency_std=features_dict.get('search_speed_std', 0.0),
            efficiency_median=features_dict.get('search_time_median', 0.0),
            efficiency_max=features_dict.get('search_speed_mean', 0.0),
            efficiency_min=features_dict.get('search_speed_mean', 0.0),
            entropy_mean=0.0,
            entropy_std=0.0,
            total_duration=features_dict['total_duration'],
            duration_mean=features_dict['search_time_mean'],
            duration_std=features_dict['search_time_std'],
            hesitation_mean=features_dict['hesitation_mean'],
            hesitation_ratio=features_dict['search_time_cv'],
            revisit_total=int(features_dict['revisit_total']),
            revisit_mean=features_dict['revisit_mean'],
            velocity_mean=features_dict['velocity_mean'],
            velocity_std=features_dict['search_speed_std'],
            total_path_length=features_dict['total_move_distance'],
            total_euclidean=features_dict['total_move_distance'],
            distraction_cost=0.0,
            has_distractor=int(features_dict['has_distractor']),
            grid_size=int(features_dict['grid_size']),
            distractor_count=int(features_dict['distractor_count']),
        )

    def _empty_features(self, trial: TaskTrial) -> Dict[str, float]:
        """返回空特征字典"""
        return {
            'total_search_time': 0.0,
            'search_time_mean': 0.0,
            'search_time_std': 0.0,
            'search_time_max': 0.0,
            'search_time_min': 0.0,
            'search_time_median': 0.0,
            'search_time_cv': 0.0,
            'total_move_distance': 0.0,
            'move_distance_mean': 0.0,
            'move_distance_std': 0.0,
            'move_distance_max': 0.0,
            'move_distance_min': 0.0,
            'search_speed_mean': 0.0,
            'search_speed_std': 0.0,
            'search_speed_cv': 0.0,
            'quadrant_change_ratio': 0.0,
            'center_distance_mean': 0.0,
            'center_distance_std': 0.0,
            'normalized_distance_mean': 0.0,
            'revisit_total': 0.0,
            'revisit_ratio': 0.0,
            'has_distractor': float(trial.config.has_distractor),
            'grid_size': float(trial.config.grid_size),
            'distractor_count': float(trial.config.distractor_count),
            'click_disappear': float(trial.config.click_disappear),
            'segment_count': 0.0,
            'click_count': 0.0,
            'completion_rate': 0.0,
            'distance_time_ratio': 0.0,
            # 眼动特征
            'has_gaze_data': 0.0,
            'path_efficiency_mean': 1.0,
            'path_efficiency_std': 0.0,
            'path_efficiency_min': 1.0,
            'fixation_entropy_mean': 0.0,
            'fixation_entropy_std': 0.0,
            'fixation_entropy_max': 0.0,
            'gaze_velocity_mean': 0.0,
            'gaze_velocity_std_mean': 0.0,
            'gaze_velocity_max_mean': 0.0,
            'gaze_velocity_max_max': 0.0,
            'direction_changes_total': 0.0,
            'direction_changes_mean': 0.0,
            'direction_changes_max': 0.0,
            'hesitation_time_total': 0.0,
            'hesitation_time_mean': 0.0,
            'hesitation_time_max': 0.0,
            'hesitation_ratio': 0.0,
            'gaze_point_total': 0.0,
            'gaze_point_mean': 0.0,
            'path_length_total': 0.0,
            'path_length_mean': 0.0,
            'path_length_std': 0.0,
            'acceleration_mean': 0.0,
            'acceleration_max': 0.0,
            # 兼容旧名称
            'efficiency_mean': 0.0,
            'entropy_mean': 0.0,
            'total_duration': 0.0,
            'revisit_mean': 0.0,
            'velocity_mean': 0.0,
            'distraction_cost': 0.0,
        }


class TaskPerformanceAnalyzer:
    """
    任务表现分析器

    提供任务级别的深度分析
    """

    @staticmethod
    def analyze_performance_trend(
        meso_features_list: List[Dict[str, float]],
    ) -> Dict[str, float]:
        """
        分析任务表现趋势

        Args:
            meso_features_list: 按任务顺序排列的中观特征列表

        Returns:
            趋势分析结果
        """
        if len(meso_features_list) < 2:
            return {
                'search_time_trend': 0.0,
                'speed_trend': 0.0,
                'distance_trend': 0.0,
            }

        n = len(meso_features_list)
        x = np.arange(n)

        # 搜索时间趋势
        search_times = [f.get('search_time_mean', f.get('total_duration', 0)) for f in meso_features_list]
        time_slope = np.polyfit(x, search_times, 1)[0] if any(t > 0 for t in search_times) else 0.0

        # 速度趋势
        speeds = [f.get('search_speed_mean', f.get('velocity_mean', 0)) for f in meso_features_list]
        speed_slope = np.polyfit(x, speeds, 1)[0] if any(s > 0 for s in speeds) else 0.0

        # 距离趋势
        distances = [f.get('move_distance_mean', f.get('total_move_distance', 0)) for f in meso_features_list]
        dist_slope = np.polyfit(x, distances, 1)[0] if any(d > 0 for d in distances) else 0.0

        return {
            'search_time_trend': float(time_slope),
            'speed_trend': float(speed_slope),
            'distance_trend': float(dist_slope),
        }

    @staticmethod
    def compare_conditions(
        meso_features_list: List[Dict[str, float]],
    ) -> Dict[str, float]:
        """
        比较不同条件下的表现

        Args:
            meso_features_list: 中观特征列表

        Returns:
            条件对比结果
        """
        # 按干扰条件分组
        with_distractor = [
            f for f in meso_features_list if f.get('has_distractor', 0) > 0
        ]
        without_distractor = [
            f for f in meso_features_list if f.get('has_distractor', 0) == 0
        ]

        result = {}

        # 搜索时间对比
        if with_distractor and without_distractor:
            time_with = np.mean([f.get('search_time_mean', 0) for f in with_distractor])
            time_without = np.mean([f.get('search_time_mean', 0) for f in without_distractor])
            result['distractor_time_diff'] = float(time_with - time_without)

            speed_with = np.mean([f.get('search_speed_mean', 0) for f in with_distractor])
            speed_without = np.mean([f.get('search_speed_mean', 0) for f in without_distractor])
            result['distractor_speed_diff'] = float(speed_with - speed_without)
        else:
            result['distractor_time_diff'] = 0.0
            result['distractor_speed_diff'] = 0.0

        # 按网格大小分组
        small_grid = [f for f in meso_features_list if f.get('grid_size', 25) <= 25]
        large_grid = [f for f in meso_features_list if f.get('grid_size', 25) > 25]

        if small_grid and large_grid:
            time_small = np.mean([f.get('search_time_mean', 0) for f in small_grid])
            time_large = np.mean([f.get('search_time_mean', 0) for f in large_grid])
            result['grid_size_time_diff'] = float(time_small - time_large)
        else:
            result['grid_size_time_diff'] = 0.0

        return result
