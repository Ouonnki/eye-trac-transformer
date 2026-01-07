# -*- coding: utf-8 -*-
"""
微观特征提取模块 - Level 1（片段级）

基于点击序列和连续眼动轨迹提取每个搜索片段的特征。

数据来源：
- sheet3: 点击事件（用于片段边界定义）
- sheet4: 连续眼动轨迹（用于详细眼动特征提取）
"""

import logging
import math
from typing import Dict, List, Optional

import numpy as np

from src.data.schemas import GazePoint, SearchSegment, MicroFeatures

logger = logging.getLogger(__name__)


class MicroFeatureExtractor:
    """
    微观特征提取器

    为每个搜索片段提取以下特征：

    基础特征（基于点击）：
    - search_time_ms: 搜索时间
    - move_distance: 起止点距离
    - search_speed: 搜索速度

    眼动特征（基于连续轨迹，如果有）：
    - path_length: 实际眼动路径长度
    - path_efficiency: 路径效率（欧氏距离/路径长度）
    - fixation_entropy: 注视熵（空间分布混乱度）
    - velocity_mean/std: 速度统计
    - direction_changes: 方向变化次数
    - hesitation_time: 犹豫时间（首次接近目标到点击的时间）
    - revisit_count: 回视次数
    """

    def __init__(
        self,
        screen_width: int = 1920,
        screen_height: int = 1080,
        proximity_threshold: float = 100.0,
        entropy_grid_size: int = 8,
    ):
        """
        初始化提取器

        Args:
            screen_width: 屏幕宽度
            screen_height: 屏幕高度
            proximity_threshold: 接近已点击位置的阈值（像素）
            entropy_grid_size: 熵计算网格大小
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.proximity_threshold = proximity_threshold
        self.entropy_grid_size = entropy_grid_size
        self.screen_diagonal = math.sqrt(screen_width**2 + screen_height**2)
        self.screen_center = (screen_width / 2, screen_height / 2)

    def extract(self, segment: SearchSegment) -> Dict[str, float]:
        """
        提取片段级微观特征

        Args:
            segment: 搜索片段（包含起始点击、中间眼动点和目标点击）

        Returns:
            特征字典
        """
        if len(segment.gaze_points) < 2:
            return self._empty_features()

        start_point = segment.gaze_points[0]
        end_point = segment.gaze_points[-1]

        # 基础时间和距离特征
        search_time_ms = segment.duration_ms
        euclidean_distance = self._calc_distance(start_point, end_point)
        search_speed = euclidean_distance / search_time_ms if search_time_ms > 0 else 0.0

        # 位移特征
        x_displacement = end_point.x - start_point.x
        y_displacement = end_point.y - start_point.y

        # 象限变化
        quadrant_change = self._calc_quadrant_change(start_point, end_point)

        # 目标距离屏幕中心的距离
        distance_from_center = self._calc_distance_from_center(end_point)

        # 回访已点击区域
        revisit_area_count = self._calc_revisit_count(end_point, segment.clicked_positions)

        # 归一化距离
        normalized_distance = euclidean_distance / self.screen_diagonal

        # 检测是否有真正的眼动数据（超过2个点）
        has_gaze_data = len(segment.gaze_points) > 2

        if has_gaze_data:
            # 提取眼动轨迹特征
            gaze_features = self._extract_gaze_features(segment)
        else:
            # 无眼动数据时使用默认值
            gaze_features = {
                'path_length': euclidean_distance,
                'path_efficiency': 1.0,
                'fixation_entropy': 0.0,
                'velocity_mean': search_speed,
                'velocity_std': 0.0,
                'velocity_max': search_speed,
                'direction_changes': 0,
                'hesitation_time': 0.0,
                'gaze_point_count': 2,
                'acceleration_mean': 0.0,
            }

        return {
            'search_time_ms': search_time_ms,
            'move_distance': euclidean_distance,
            'search_speed': search_speed,
            'target_number': float(segment.target_number),
            'x_displacement': x_displacement,
            'y_displacement': y_displacement,
            'quadrant_change': quadrant_change,
            'distance_from_center': distance_from_center,
            'revisit_area_count': float(revisit_area_count),
            'normalized_distance': normalized_distance,
            # 眼动特征
            'path_length': gaze_features['path_length'],
            'path_efficiency': gaze_features['path_efficiency'],
            'fixation_entropy': gaze_features['fixation_entropy'],
            'velocity_mean': gaze_features['velocity_mean'],
            'velocity_std': gaze_features['velocity_std'],
            'velocity_max': gaze_features['velocity_max'],
            'direction_changes': float(gaze_features['direction_changes']),
            'hesitation_time': gaze_features['hesitation_time'],
            'gaze_point_count': float(gaze_features['gaze_point_count']),
            'acceleration_mean': gaze_features['acceleration_mean'],
            'has_gaze_data': float(has_gaze_data),
        }

    def _extract_gaze_features(self, segment: SearchSegment) -> Dict[str, float]:
        """
        从连续眼动轨迹提取特征

        Args:
            segment: 包含眼动点的搜索片段

        Returns:
            眼动特征字典
        """
        points = segment.gaze_points
        n = len(points)

        # 路径长度（实际扫视路径）
        path_length = 0.0
        for i in range(1, n):
            path_length += self._calc_distance(points[i - 1], points[i])

        # 路径效率（欧氏距离 / 路径长度）
        euclidean = self._calc_distance(points[0], points[-1])
        path_efficiency = euclidean / path_length if path_length > 0 else 1.0

        # 计算速度序列
        velocities = []
        for i in range(1, n):
            dt = (points[i].timestamp - points[i - 1].timestamp).total_seconds() * 1000
            if dt > 0:
                dist = self._calc_distance(points[i - 1], points[i])
                velocities.append(dist / dt)

        velocity_mean = np.mean(velocities) if velocities else 0.0
        velocity_std = np.std(velocities) if velocities else 0.0
        velocity_max = max(velocities) if velocities else 0.0

        # 加速度
        accelerations = []
        for i in range(1, len(velocities)):
            dt = (points[i + 1].timestamp - points[i].timestamp).total_seconds() * 1000
            if dt > 0:
                accelerations.append((velocities[i] - velocities[i - 1]) / dt)
        acceleration_mean = np.mean(np.abs(accelerations)) if accelerations else 0.0

        # 方向变化次数
        direction_changes = self._calc_direction_changes(points)

        # 注视熵
        fixation_entropy = self._calc_fixation_entropy(points)

        # 犹豫时间（首次接近目标到最终点击的时间）
        hesitation_time = self._calc_hesitation_time(points, segment.target_position)

        return {
            'path_length': path_length,
            'path_efficiency': path_efficiency,
            'fixation_entropy': fixation_entropy,
            'velocity_mean': velocity_mean,
            'velocity_std': velocity_std,
            'velocity_max': velocity_max,
            'direction_changes': direction_changes,
            'hesitation_time': hesitation_time,
            'gaze_point_count': n,
            'acceleration_mean': acceleration_mean,
        }

    def _calc_fixation_entropy(self, points: List[GazePoint]) -> float:
        """
        计算注视熵（空间分布的混乱度）

        使用 entropy_grid_size x entropy_grid_size 网格
        """
        if len(points) < 2:
            return 0.0

        grid_size = self.entropy_grid_size
        cell_width = self.screen_width / grid_size
        cell_height = self.screen_height / grid_size

        # 统计每个网格的注视次数
        grid_counts = np.zeros((grid_size, grid_size))
        for p in points:
            col = min(int(p.x / cell_width), grid_size - 1)
            row = min(int(p.y / cell_height), grid_size - 1)
            col = max(0, col)
            row = max(0, row)
            grid_counts[row, col] += 1

        # 计算概率分布
        total = np.sum(grid_counts)
        if total == 0:
            return 0.0

        probs = grid_counts.flatten() / total
        probs = probs[probs > 0]  # 过滤零概率

        # 计算香农熵
        entropy = -np.sum(probs * np.log2(probs))

        # 归一化（最大熵为 log2(grid_size^2)）
        max_entropy = np.log2(grid_size * grid_size)
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def _calc_direction_changes(self, points: List[GazePoint]) -> int:
        """
        计算方向变化次数

        检测眼动轨迹中的方向反转
        """
        if len(points) < 3:
            return 0

        changes = 0
        for i in range(2, len(points)):
            # 计算前后两段的方向向量
            dx1 = points[i - 1].x - points[i - 2].x
            dy1 = points[i - 1].y - points[i - 2].y
            dx2 = points[i].x - points[i - 1].x
            dy2 = points[i].y - points[i - 1].y

            # 计算角度变化
            dot = dx1 * dx2 + dy1 * dy2
            mag1 = math.sqrt(dx1**2 + dy1**2)
            mag2 = math.sqrt(dx2**2 + dy2**2)

            if mag1 > 0 and mag2 > 0:
                cos_angle = dot / (mag1 * mag2)
                cos_angle = max(-1, min(1, cos_angle))
                angle = math.acos(cos_angle)

                # 超过90度视为方向变化
                if angle > math.pi / 2:
                    changes += 1

        return changes

    def _calc_hesitation_time(
        self,
        points: List[GazePoint],
        target_position: tuple,
    ) -> float:
        """
        计算犹豫时间

        从首次接近目标到最终点击的时间差
        """
        if len(points) < 2 or target_position is None:
            return 0.0

        target_x, target_y = target_position
        threshold = self.proximity_threshold

        first_approach_time = None
        for p in points:
            dist = math.sqrt((p.x - target_x)**2 + (p.y - target_y)**2)
            if dist <= threshold:
                first_approach_time = p.timestamp
                break

        if first_approach_time is None:
            return 0.0

        # 犹豫时间 = 最终点击时间 - 首次接近时间
        end_time = points[-1].timestamp
        hesitation = (end_time - first_approach_time).total_seconds() * 1000
        return max(0.0, hesitation)

    @staticmethod
    def _calc_distance(p1: GazePoint, p2: GazePoint) -> float:
        """计算两点之间的欧氏距离"""
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        return math.sqrt(dx**2 + dy**2)

    def _calc_quadrant_change(self, p1: GazePoint, p2: GazePoint) -> float:
        """
        计算象限变化

        将屏幕分为4个象限，返回是否跨象限移动
        """
        cx, cy = self.screen_center
        q1 = self._get_quadrant(p1.x, p1.y, cx, cy)
        q2 = self._get_quadrant(p2.x, p2.y, cx, cy)
        return 1.0 if q1 != q2 else 0.0

    @staticmethod
    def _get_quadrant(x: float, y: float, cx: float, cy: float) -> int:
        """获取点所在象限（1-4）"""
        if x >= cx and y < cy:
            return 1
        elif x < cx and y < cy:
            return 2
        elif x < cx and y >= cy:
            return 3
        else:
            return 4

    def _calc_distance_from_center(self, point: GazePoint) -> float:
        """计算点到屏幕中心的距离"""
        cx, cy = self.screen_center
        return math.sqrt((point.x - cx)**2 + (point.y - cy)**2)

    def _calc_revisit_count(
        self,
        current_point: GazePoint,
        clicked_positions: List,
    ) -> int:
        """
        计算当前点击是否接近已点击位置

        Returns:
            如果接近任何已点击位置返回1，否则返回0
        """
        for pos in clicked_positions:
            if isinstance(pos, tuple):
                px, py = pos
            else:
                px, py = pos.x, pos.y

            dist = math.sqrt((current_point.x - px)**2 + (current_point.y - py)**2)
            if dist <= self.proximity_threshold:
                return 1
        return 0

    def _empty_features(self) -> Dict[str, float]:
        """返回空特征字典"""
        return {
            'search_time_ms': 0.0,
            'move_distance': 0.0,
            'search_speed': 0.0,
            'target_number': 0.0,
            'x_displacement': 0.0,
            'y_displacement': 0.0,
            'quadrant_change': 0.0,
            'distance_from_center': 0.0,
            'revisit_area_count': 0.0,
            'normalized_distance': 0.0,
            # 眼动特征
            'path_length': 0.0,
            'path_efficiency': 1.0,
            'fixation_entropy': 0.0,
            'velocity_mean': 0.0,
            'velocity_std': 0.0,
            'velocity_max': 0.0,
            'direction_changes': 0.0,
            'hesitation_time': 0.0,
            'gaze_point_count': 0.0,
            'acceleration_mean': 0.0,
            'has_gaze_data': 0.0,
        }


class ClickSequenceAnalyzer:
    """
    点击序列分析器

    分析整个任务的点击序列模式
    """

    def __init__(self, screen_width: int = 1920, screen_height: int = 1080):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.extractor = MicroFeatureExtractor(screen_width, screen_height)

    def analyze(self, segments: List[SearchSegment]) -> Dict[str, float]:
        """
        分析点击序列

        Args:
            segments: 搜索片段列表

        Returns:
            任务级特征字典
        """
        if not segments:
            return self._empty_analysis()

        # 提取所有片段的微观特征
        all_features = [self.extractor.extract(seg) for seg in segments]

        # 聚合特征
        search_times = [f['search_time_ms'] for f in all_features]
        move_distances = [f['move_distance'] for f in all_features]
        search_speeds = [f['search_speed'] for f in all_features]
        quadrant_changes = [f['quadrant_change'] for f in all_features]

        return {
            'total_search_time': sum(search_times),
            'mean_search_time': np.mean(search_times),
            'std_search_time': np.std(search_times),
            'total_move_distance': sum(move_distances),
            'mean_move_distance': np.mean(move_distances),
            'std_move_distance': np.std(move_distances),
            'mean_search_speed': np.mean(search_speeds),
            'std_search_speed': np.std(search_speeds),
            'quadrant_change_ratio': np.mean(quadrant_changes),
            'search_time_cv': np.std(search_times) / np.mean(search_times) if np.mean(search_times) > 0 else 0,
            'speed_cv': np.std(search_speeds) / np.mean(search_speeds) if np.mean(search_speeds) > 0 else 0,
        }

    def _empty_analysis(self) -> Dict[str, float]:
        """返回空分析结果"""
        return {
            'total_search_time': 0.0,
            'mean_search_time': 0.0,
            'std_search_time': 0.0,
            'total_move_distance': 0.0,
            'mean_move_distance': 0.0,
            'std_move_distance': 0.0,
            'mean_search_speed': 0.0,
            'std_search_speed': 0.0,
            'quadrant_change_ratio': 0.0,
            'search_time_cv': 0.0,
            'speed_cv': 0.0,
        }
