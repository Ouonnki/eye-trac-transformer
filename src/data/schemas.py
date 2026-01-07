# -*- coding: utf-8 -*-
"""
数据结构定义模块

定义眼动追踪分析中使用的核心数据类型
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple, Dict


@dataclass
class GazePoint:
    """
    单个眼动点

    Attributes:
        timestamp: 时间戳（毫秒级精度）
        x: X坐标（像素）
        y: Y坐标（像素）
        is_click: 是否为点击事件
        target_number: 点击的目标数字（仅点击事件有值）
    """
    timestamp: datetime
    x: float
    y: float
    is_click: bool = False
    target_number: Optional[int] = None

    def distance_to(self, other: 'GazePoint') -> float:
        """计算到另一个点的欧氏距离"""
        import math
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def distance_to_position(self, pos: Tuple[float, float]) -> float:
        """计算到指定位置的欧氏距离"""
        import math
        return math.sqrt((self.x - pos[0]) ** 2 + (self.y - pos[1]) ** 2)


@dataclass
class SearchSegment:
    """
    搜索片段：从 Click(N-1) 到 Click(N) 之间的眼动数据

    Attributes:
        segment_id: 片段序号
        start_number: 起始数字 N-1
        target_number: 目标数字 N
        gaze_points: 该片段内的眼动点列表
        start_time: 片段开始时间
        end_time: 片段结束时间（点击时间）
        target_position: 目标位置 (x, y)
        clicked_positions: 已点击位置列表（用于回视计算）
    """
    segment_id: int
    start_number: int
    target_number: int
    gaze_points: List[GazePoint]
    start_time: datetime
    end_time: datetime
    target_position: Tuple[float, float]
    clicked_positions: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        """片段持续时间（毫秒）"""
        delta = self.end_time - self.start_time
        return delta.total_seconds() * 1000

    @property
    def point_count(self) -> int:
        """眼动点数量"""
        return len(self.gaze_points)

    @property
    def start_position(self) -> Optional[Tuple[float, float]]:
        """片段起始位置"""
        if self.gaze_points:
            p = self.gaze_points[0]
            return (p.x, p.y)
        return None


@dataclass
class TaskConfig:
    """
    任务配置

    Attributes:
        task_id: 题目编号（1-30）
        grid_size: 方格数量（如 25 = 5x5）
        number_range: 数字范围 (min, max)
        click_disappear: 点击后数字是否消失
        has_distractor: 是否有干扰项
        distractor_count: 干扰项数量
    """
    task_id: int
    grid_size: int = 25
    number_range: Tuple[int, int] = (1, 25)
    click_disappear: bool = False
    has_distractor: bool = False
    distractor_count: int = 0


@dataclass
class TaskTrial:
    """
    单个任务试次

    Attributes:
        subject_id: 被试编号
        task_id: 题目编号
        config: 任务配置
        segments: 搜索片段列表
        raw_gaze_points: 原始眼动点列表
    """
    subject_id: str
    task_id: int
    config: TaskConfig
    segments: List[SearchSegment] = field(default_factory=list)
    raw_gaze_points: List[GazePoint] = field(default_factory=list)

    @property
    def total_duration_ms(self) -> float:
        """任务总时长（毫秒）"""
        if not self.raw_gaze_points:
            return 0.0
        start = self.raw_gaze_points[0].timestamp
        end = self.raw_gaze_points[-1].timestamp
        return (end - start).total_seconds() * 1000

    @property
    def click_count(self) -> int:
        """点击次数"""
        return sum(1 for p in self.raw_gaze_points if p.is_click)


@dataclass
class SubjectData:
    """
    被试数据

    Attributes:
        subject_id: 被试编号
        total_score: 总分（连续值）
        category: 类别 (1:低分组, 2:中分组, 3:高分组)
        trials: 所有任务试次列表
    """
    subject_id: str
    total_score: float
    category: int
    trials: List[TaskTrial] = field(default_factory=list)

    @property
    def trial_count(self) -> int:
        """完成的任务数量"""
        return len(self.trials)


@dataclass
class MicroFeatures:
    """
    微观特征（片段级）

    Attributes:
        search_efficiency_ratio: 搜索效率比
        fixation_entropy: 注视熵
        hesitation_time: 犹豫度（毫秒）
        revisit_count: 回视次数
        path_length: 实际路径长度
        euclidean_distance: 欧氏距离
        duration_ms: 持续时间
        gaze_point_count: 眼动点数量
        velocity_mean: 平均速度
        direction_changes: 方向变化次数
    """
    search_efficiency_ratio: float = 1.0
    fixation_entropy: float = 0.0
    hesitation_time: float = 0.0
    revisit_count: int = 0
    path_length: float = 0.0
    euclidean_distance: float = 0.0
    duration_ms: float = 0.0
    gaze_point_count: int = 0
    velocity_mean: float = 0.0
    direction_changes: int = 0

    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            'search_efficiency_ratio': self.search_efficiency_ratio,
            'fixation_entropy': self.fixation_entropy,
            'hesitation_time': self.hesitation_time,
            'revisit_count': float(self.revisit_count),
            'path_length': self.path_length,
            'euclidean_distance': self.euclidean_distance,
            'duration_ms': self.duration_ms,
            'gaze_point_count': float(self.gaze_point_count),
            'velocity_mean': self.velocity_mean,
            'direction_changes': float(self.direction_changes),
        }


@dataclass
class MesoFeatures:
    """
    中观特征（任务级）
    """
    efficiency_mean: float = 0.0
    efficiency_std: float = 0.0
    efficiency_median: float = 0.0
    efficiency_max: float = 0.0
    efficiency_min: float = 0.0
    entropy_mean: float = 0.0
    entropy_std: float = 0.0
    total_duration: float = 0.0
    duration_mean: float = 0.0
    duration_std: float = 0.0
    hesitation_mean: float = 0.0
    hesitation_ratio: float = 0.0
    revisit_total: int = 0
    revisit_mean: float = 0.0
    velocity_mean: float = 0.0
    velocity_std: float = 0.0
    total_path_length: float = 0.0
    total_euclidean: float = 0.0
    distraction_cost: float = 0.0
    has_distractor: int = 0
    grid_size: int = 25
    distractor_count: int = 0

    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            'efficiency_mean': self.efficiency_mean,
            'efficiency_std': self.efficiency_std,
            'efficiency_median': self.efficiency_median,
            'efficiency_max': self.efficiency_max,
            'efficiency_min': self.efficiency_min,
            'entropy_mean': self.entropy_mean,
            'entropy_std': self.entropy_std,
            'total_duration': self.total_duration,
            'duration_mean': self.duration_mean,
            'duration_std': self.duration_std,
            'hesitation_mean': self.hesitation_mean,
            'hesitation_ratio': self.hesitation_ratio,
            'revisit_total': float(self.revisit_total),
            'revisit_mean': self.revisit_mean,
            'velocity_mean': self.velocity_mean,
            'velocity_std': self.velocity_std,
            'total_path_length': self.total_path_length,
            'total_euclidean': self.total_euclidean,
            'distraction_cost': self.distraction_cost,
            'has_distractor': float(self.has_distractor),
            'grid_size': float(self.grid_size),
            'distractor_count': float(self.distractor_count),
        }
