# -*- coding: utf-8 -*-
"""
宏观特征提取模块 - Level 3（被试级）

基于中观特征聚合计算被试级特征，用于最终建模。

数据来源：
- 点击事件：用于搜索时间、移动距离和速度统计
- 连续眼动轨迹：用于路径效率、注视熵、方向变化等眼动特征
"""

import logging
from typing import Dict, List, Tuple

import numpy as np

from src.data.schemas import SubjectData, TaskTrial

logger = logging.getLogger(__name__)


class MacroFeatureExtractor:
    """
    宏观特征提取器

    基于中观特征聚合计算被试级特征：
    - distraction_sensitivity: 干扰敏感度
    - fatigue_slope: 疲劳斜率（搜索速度随任务序号的变化）
    - stability_score: 稳定性分数
    - learning_rate: 学习率
    - global_*: 全局聚合指标
    - grid_size_effect: 方格大小效应
    - 眼动特征聚合：路径效率、注视熵、方向变化、犹豫时间等
    """

    def extract(
        self,
        subject: SubjectData,
        meso_features_list: List[Dict[str, float]],
    ) -> Dict[str, float]:
        """
        提取被试级特征

        Args:
            subject: 被试数据
            meso_features_list: 该被试所有任务的中观特征列表（按任务顺序）

        Returns:
            被试级特征字典
        """
        if not meso_features_list:
            return self._empty_features()

        features = {}

        # 1. 干扰敏感度
        features['distraction_sensitivity'] = self._calc_distraction_sensitivity(
            subject.trials, meso_features_list
        )

        # 2. 疲劳斜率和截距（基于搜索速度）
        fatigue_slope, fatigue_intercept = self._calc_fatigue_coefficients(
            meso_features_list
        )
        features['fatigue_slope'] = fatigue_slope
        features['fatigue_intercept'] = fatigue_intercept

        # 3. 稳定性分数
        features['stability_score'] = self._calc_stability_score(meso_features_list)

        # 4. 学习率
        features['learning_rate'] = self._calc_learning_rate(meso_features_list)

        # 5. 全局聚合指标
        features.update(self._calc_global_aggregates(meso_features_list))

        # 6. 条件对比特征
        features.update(
            self._calc_condition_contrasts(subject.trials, meso_features_list)
        )

        # 7. 一致性指标
        features.update(self._calc_consistency_metrics(meso_features_list))

        # 8. 极值特征
        features.update(self._calc_extreme_features(meso_features_list))

        # 9. 新增：搜索效率相关特征
        features.update(self._calc_search_efficiency_features(meso_features_list))

        # 10. 眼动特征聚合
        features.update(self._calc_gaze_features(meso_features_list))

        return features

    def _calc_distraction_sensitivity(
        self,
        trials: List[TaskTrial],
        meso_features_list: List[Dict[str, float]],
    ) -> float:
        """
        干扰敏感度

        公式：(有干扰搜索时间 - 无干扰搜索时间) / 无干扰搜索时间

        解释：
        - 正值表示干扰导致搜索变慢
        - 值越大表示越敏感
        """
        with_distractor = []
        without_distractor = []

        for trial, features in zip(trials, meso_features_list):
            search_time = features.get('search_time_mean', features.get('total_duration', 0))
            if trial.config.has_distractor:
                with_distractor.append(search_time)
            else:
                without_distractor.append(search_time)

        if not with_distractor or not without_distractor:
            return 0.0

        mean_with = np.mean(with_distractor)
        mean_without = np.mean(without_distractor)

        if mean_without < 1e-6:
            return 0.0

        return float((mean_with - mean_without) / mean_without)

    def _calc_fatigue_coefficients(
        self,
        meso_features_list: List[Dict[str, float]],
    ) -> Tuple[float, float]:
        """
        疲劳系数

        使用线性回归计算搜索速度随任务序号的变化趋势

        返回：(斜率, 截距)

        解释：
        - 负斜率表示随时间速度下降（疲劳）
        - 正斜率表示随时间速度提升（学习/适应）
        """
        if len(meso_features_list) < 2:
            return 0.0, 0.0

        task_indices = np.arange(len(meso_features_list))
        speed_values = [
            f.get('search_speed_mean', f.get('velocity_mean', 0))
            for f in meso_features_list
        ]

        if all(v == 0 for v in speed_values):
            return 0.0, 0.0

        # 线性回归
        coeffs = np.polyfit(task_indices, speed_values, 1)

        return float(coeffs[0]), float(coeffs[1])

    def _calc_stability_score(
        self,
        meso_features_list: List[Dict[str, float]],
    ) -> float:
        """
        稳定性分数

        公式：1 / (1 + CV)，其中 CV = std / mean

        解释：
        - 值越接近 1 表示越稳定
        - 值越接近 0 表示波动越大
        """
        speed_values = [
            f.get('search_speed_mean', f.get('velocity_mean', 0))
            for f in meso_features_list
        ]

        mean = np.mean(speed_values)
        std = np.std(speed_values)

        if mean < 1e-6:
            return 0.0

        cv = std / mean
        return float(1 / (1 + cv))

    def _calc_learning_rate(
        self,
        meso_features_list: List[Dict[str, float]],
    ) -> float:
        """
        学习率

        公式：(后半段速度 - 前半段速度) / 前半段速度

        解释：
        - 正值表示有学习效应（速度提升）
        - 负值表示有疲劳效应（速度下降）
        """
        if len(meso_features_list) < 4:
            return 0.0

        mid = len(meso_features_list) // 2
        first_half = [
            f.get('search_speed_mean', f.get('velocity_mean', 0))
            for f in meso_features_list[:mid]
        ]
        second_half = [
            f.get('search_speed_mean', f.get('velocity_mean', 0))
            for f in meso_features_list[mid:]
        ]

        mean_first = np.mean(first_half)
        mean_second = np.mean(second_half)

        if mean_first < 1e-6:
            return 0.0

        return float((mean_second - mean_first) / mean_first)

    def _calc_global_aggregates(
        self,
        meso_features_list: List[Dict[str, float]],
    ) -> Dict[str, float]:
        """
        计算全局聚合指标
        """
        aggregates = {}

        # 要聚合的特征列表（使用新的特征名称，兼容旧名称）
        keys_to_aggregate = [
            ('search_speed_mean', 'velocity_mean'),
            ('search_time_mean', 'hesitation_mean'),
            ('total_search_time', 'total_duration'),
            ('move_distance_mean', 'path_length'),
            ('quadrant_change_ratio', 'quadrant_change_ratio'),
            ('revisit_ratio', 'revisit_mean'),
            ('completion_rate', 'completion_rate'),
        ]

        for new_key, old_key in keys_to_aggregate:
            values = [f.get(new_key, f.get(old_key, 0)) for f in meso_features_list]
            aggregates[f'global_{new_key}_mean'] = float(np.mean(values))
            aggregates[f'global_{new_key}_std'] = float(np.std(values))
            aggregates[f'global_{new_key}_median'] = float(np.median(values))

        # 总任务数
        aggregates['total_trials'] = float(len(meso_features_list))

        # 总时长（毫秒）
        aggregates['total_experiment_duration'] = float(
            sum(f.get('total_search_time', f.get('total_duration', 0)) for f in meso_features_list)
        )

        # 总点击数
        aggregates['total_clicks'] = float(
            sum(f.get('click_count', 0) for f in meso_features_list)
        )

        # 平均完成速度
        total_time = aggregates['total_experiment_duration']
        if total_time > 0:
            aggregates['overall_completion_rate'] = aggregates['total_clicks'] / (total_time / 1000)
        else:
            aggregates['overall_completion_rate'] = 0.0

        return aggregates

    def _calc_condition_contrasts(
        self,
        trials: List[TaskTrial],
        meso_features_list: List[Dict[str, float]],
    ) -> Dict[str, float]:
        """
        计算不同条件下的对比特征
        """
        contrasts = {}

        # 按方格大小分组
        small_grid = []
        large_grid = []

        for trial, features in zip(trials, meso_features_list):
            speed = features.get('search_speed_mean', features.get('velocity_mean', 0))
            if trial.config.grid_size <= 25:
                small_grid.append(speed)
            else:
                large_grid.append(speed)

        if small_grid and large_grid:
            contrasts['grid_size_effect'] = float(
                np.mean(small_grid) - np.mean(large_grid)
            )
        else:
            contrasts['grid_size_effect'] = 0.0

        # 按点击消失分组
        disappear = []
        not_disappear = []

        for trial, features in zip(trials, meso_features_list):
            speed = features.get('search_speed_mean', features.get('velocity_mean', 0))
            if trial.config.click_disappear:
                disappear.append(speed)
            else:
                not_disappear.append(speed)

        if disappear and not_disappear:
            contrasts['click_disappear_effect'] = float(
                np.mean(disappear) - np.mean(not_disappear)
            )
        else:
            contrasts['click_disappear_effect'] = 0.0

        return contrasts

    def _calc_consistency_metrics(
        self,
        meso_features_list: List[Dict[str, float]],
    ) -> Dict[str, float]:
        """
        计算一致性指标
        """
        metrics = {}

        # 搜索速度四分位距
        speed_values = [
            f.get('search_speed_mean', f.get('velocity_mean', 0))
            for f in meso_features_list
        ]
        q1, q3 = np.percentile(speed_values, [25, 75])
        metrics['speed_iqr'] = float(q3 - q1)

        # 搜索时间变异系数
        time_values = [
            f.get('search_time_mean', f.get('total_duration', 0))
            for f in meso_features_list
        ]
        time_mean = np.mean(time_values)
        time_std = np.std(time_values)
        metrics['search_time_cv'] = (
            float(time_std / time_mean) if time_mean > 0 else 0.0
        )

        # 移动距离变异系数
        distance_values = [
            f.get('move_distance_mean', 0)
            for f in meso_features_list
        ]
        dist_mean = np.mean(distance_values)
        dist_std = np.std(distance_values)
        metrics['distance_cv'] = (
            float(dist_std / dist_mean) if dist_mean > 0 else 0.0
        )

        return metrics

    def _calc_extreme_features(
        self,
        meso_features_list: List[Dict[str, float]],
    ) -> Dict[str, float]:
        """
        计算极值特征
        """
        extremes = {}

        # 最佳表现任务（速度最快）
        speed_values = [
            f.get('search_speed_mean', f.get('velocity_mean', 0))
            for f in meso_features_list
        ]
        best_idx = np.argmax(speed_values)
        worst_idx = np.argmin(speed_values)

        extremes['best_task_speed'] = float(speed_values[best_idx])
        extremes['worst_task_speed'] = float(speed_values[worst_idx])
        extremes['speed_range'] = float(
            speed_values[best_idx] - speed_values[worst_idx]
        )

        # 最长/最短搜索时间
        time_values = [
            f.get('total_search_time', f.get('total_duration', 0))
            for f in meso_features_list
        ]
        extremes['shortest_task_time'] = float(min(time_values))
        extremes['longest_task_time'] = float(max(time_values))
        extremes['time_range'] = float(max(time_values) - min(time_values))

        return extremes

    def _calc_search_efficiency_features(
        self,
        meso_features_list: List[Dict[str, float]],
    ) -> Dict[str, float]:
        """
        计算搜索效率相关特征
        """
        features = {}

        # 平均搜索速度
        speeds = [
            f.get('search_speed_mean', f.get('velocity_mean', 0))
            for f in meso_features_list
        ]
        features['avg_search_speed'] = float(np.mean(speeds))
        features['max_search_speed'] = float(np.max(speeds))
        features['min_search_speed'] = float(np.min(speeds))

        # 平均搜索时间
        times = [
            f.get('search_time_mean', 0)
            for f in meso_features_list
        ]
        features['avg_search_time'] = float(np.mean(times))

        # 平均移动距离
        distances = [
            f.get('move_distance_mean', 0)
            for f in meso_features_list
        ]
        features['avg_move_distance'] = float(np.mean(distances))

        # 象限变化平均比例
        quadrant_changes = [
            f.get('quadrant_change_ratio', 0)
            for f in meso_features_list
        ]
        features['avg_quadrant_change'] = float(np.mean(quadrant_changes))

        # 早期 vs 晚期表现对比
        if len(meso_features_list) >= 6:
            early = speeds[:len(speeds)//3]
            late = speeds[-len(speeds)//3:]
            features['early_late_speed_diff'] = float(np.mean(late) - np.mean(early))
        else:
            features['early_late_speed_diff'] = 0.0

        return features

    def _calc_gaze_features(
        self,
        meso_features_list: List[Dict[str, float]],
    ) -> Dict[str, float]:
        """
        计算眼动特征聚合
        """
        features = {}

        # 检查是否有眼动数据
        gaze_data_counts = [f.get('has_gaze_data', 0) for f in meso_features_list]
        features['gaze_data_ratio'] = float(np.mean(gaze_data_counts))

        # 路径效率聚合
        path_eff_means = [f.get('path_efficiency_mean', 1.0) for f in meso_features_list]
        features['global_path_efficiency_mean'] = float(np.mean(path_eff_means))
        features['global_path_efficiency_std'] = float(np.std(path_eff_means))
        features['global_path_efficiency_min'] = float(np.min(path_eff_means))

        # 注视熵聚合
        entropy_means = [f.get('fixation_entropy_mean', 0.0) for f in meso_features_list]
        features['global_fixation_entropy_mean'] = float(np.mean(entropy_means))
        features['global_fixation_entropy_std'] = float(np.std(entropy_means))
        features['global_fixation_entropy_max'] = float(np.max(entropy_means))

        # 眼动速度聚合
        gaze_vel_means = [f.get('gaze_velocity_mean', 0.0) for f in meso_features_list]
        features['global_gaze_velocity_mean'] = float(np.mean(gaze_vel_means))
        features['global_gaze_velocity_std'] = float(np.std(gaze_vel_means))

        # 方向变化聚合
        dir_changes = [f.get('direction_changes_mean', 0.0) for f in meso_features_list]
        features['global_direction_changes_mean'] = float(np.mean(dir_changes))
        features['global_direction_changes_std'] = float(np.std(dir_changes))
        dir_totals = [f.get('direction_changes_total', 0.0) for f in meso_features_list]
        features['total_direction_changes'] = float(sum(dir_totals))

        # 犹豫时间聚合
        hes_means = [f.get('hesitation_time_mean', 0.0) for f in meso_features_list]
        features['global_hesitation_time_mean'] = float(np.mean(hes_means))
        features['global_hesitation_time_std'] = float(np.std(hes_means))
        hes_totals = [f.get('hesitation_time_total', 0.0) for f in meso_features_list]
        features['total_hesitation_time'] = float(sum(hes_totals))

        # 犹豫比例聚合
        hes_ratios = [f.get('hesitation_ratio', 0.0) for f in meso_features_list]
        features['global_hesitation_ratio_mean'] = float(np.mean(hes_ratios))

        # 眼动点数聚合
        gaze_points = [f.get('gaze_point_total', 0.0) for f in meso_features_list]
        features['total_gaze_points'] = float(sum(gaze_points))
        features['avg_gaze_points_per_task'] = float(np.mean(gaze_points))

        # 路径长度聚合
        path_lengths = [f.get('path_length_total', 0.0) for f in meso_features_list]
        features['total_path_length'] = float(sum(path_lengths))
        features['avg_path_length_per_task'] = float(np.mean(path_lengths))

        # 加速度聚合
        accel_means = [f.get('acceleration_mean', 0.0) for f in meso_features_list]
        features['global_acceleration_mean'] = float(np.mean(accel_means))

        # 眼动效率趋势（路径效率随任务序号的变化）
        if len(meso_features_list) >= 4:
            mid = len(meso_features_list) // 2
            first_half = path_eff_means[:mid]
            second_half = path_eff_means[mid:]
            mean_first = np.mean(first_half)
            mean_second = np.mean(second_half)
            if mean_first > 1e-6:
                features['path_efficiency_learning'] = float((mean_second - mean_first) / mean_first)
            else:
                features['path_efficiency_learning'] = 0.0
        else:
            features['path_efficiency_learning'] = 0.0

        # 注视熵趋势
        if len(meso_features_list) >= 4:
            mid = len(meso_features_list) // 2
            first_half = entropy_means[:mid]
            second_half = entropy_means[mid:]
            features['entropy_trend'] = float(np.mean(second_half) - np.mean(first_half))
        else:
            features['entropy_trend'] = 0.0

        # 干扰条件下的眼动特征差异
        with_distractor_eff = []
        without_distractor_eff = []
        for f in meso_features_list:
            eff = f.get('path_efficiency_mean', 1.0)
            if f.get('has_distractor', 0) > 0:
                with_distractor_eff.append(eff)
            else:
                without_distractor_eff.append(eff)

        if with_distractor_eff and without_distractor_eff:
            features['distractor_efficiency_effect'] = float(
                np.mean(without_distractor_eff) - np.mean(with_distractor_eff)
            )
        else:
            features['distractor_efficiency_effect'] = 0.0

        return features

    def _empty_features(self) -> Dict[str, float]:
        """返回空特征字典"""
        return {
            'distraction_sensitivity': 0.0,
            'fatigue_slope': 0.0,
            'fatigue_intercept': 0.0,
            'stability_score': 0.0,
            'learning_rate': 0.0,
            'global_search_speed_mean_mean': 0.0,
            'global_search_speed_mean_std': 0.0,
            'global_search_speed_mean_median': 0.0,
            'global_search_time_mean_mean': 0.0,
            'global_search_time_mean_std': 0.0,
            'global_search_time_mean_median': 0.0,
            'global_total_search_time_mean': 0.0,
            'global_total_search_time_std': 0.0,
            'global_total_search_time_median': 0.0,
            'global_move_distance_mean_mean': 0.0,
            'global_move_distance_mean_std': 0.0,
            'global_move_distance_mean_median': 0.0,
            'global_quadrant_change_ratio_mean': 0.0,
            'global_quadrant_change_ratio_std': 0.0,
            'global_quadrant_change_ratio_median': 0.0,
            'global_revisit_ratio_mean': 0.0,
            'global_revisit_ratio_std': 0.0,
            'global_revisit_ratio_median': 0.0,
            'global_completion_rate_mean': 0.0,
            'global_completion_rate_std': 0.0,
            'global_completion_rate_median': 0.0,
            'total_trials': 0.0,
            'total_experiment_duration': 0.0,
            'total_clicks': 0.0,
            'overall_completion_rate': 0.0,
            'grid_size_effect': 0.0,
            'click_disappear_effect': 0.0,
            'speed_iqr': 0.0,
            'search_time_cv': 0.0,
            'distance_cv': 0.0,
            'best_task_speed': 0.0,
            'worst_task_speed': 0.0,
            'speed_range': 0.0,
            'shortest_task_time': 0.0,
            'longest_task_time': 0.0,
            'time_range': 0.0,
            'avg_search_speed': 0.0,
            'max_search_speed': 0.0,
            'min_search_speed': 0.0,
            'avg_search_time': 0.0,
            'avg_move_distance': 0.0,
            'avg_quadrant_change': 0.0,
            'early_late_speed_diff': 0.0,
            # 眼动特征
            'gaze_data_ratio': 0.0,
            'global_path_efficiency_mean': 1.0,
            'global_path_efficiency_std': 0.0,
            'global_path_efficiency_min': 1.0,
            'global_fixation_entropy_mean': 0.0,
            'global_fixation_entropy_std': 0.0,
            'global_fixation_entropy_max': 0.0,
            'global_gaze_velocity_mean': 0.0,
            'global_gaze_velocity_std': 0.0,
            'global_direction_changes_mean': 0.0,
            'global_direction_changes_std': 0.0,
            'total_direction_changes': 0.0,
            'global_hesitation_time_mean': 0.0,
            'global_hesitation_time_std': 0.0,
            'total_hesitation_time': 0.0,
            'global_hesitation_ratio_mean': 0.0,
            'total_gaze_points': 0.0,
            'avg_gaze_points_per_task': 0.0,
            'total_path_length': 0.0,
            'avg_path_length_per_task': 0.0,
            'global_acceleration_mean': 0.0,
            'path_efficiency_learning': 0.0,
            'entropy_trend': 0.0,
            'distractor_efficiency_effect': 0.0,
        }
