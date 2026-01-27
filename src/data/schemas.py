# -*- coding: utf-8 -*-
"""
数据结构定义模块

定义眼动追踪分析中使用的核心数据类型
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple


@dataclass
class ClickPoint:
    """
    点击事件

    Attributes:
        timestamp: 时间戳
        x: X坐标（像素）
        y: Y坐标（像素）
        target_number: 点击的目标数字
    """
    timestamp: datetime
    x: float
    y: float
    target_number: int


@dataclass
class GazePoint:
    """
    眼动轨迹点

    Attributes:
        timestamp: 时间戳
        x: X坐标（像素）
        y: Y坐标（像素）
    """
    timestamp: datetime
    x: float
    y: float


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
        clicks: 点击事件列表
        gaze_points: 眼动轨迹点列表
        segments: 搜索片段列表
    """
    subject_id: str
    task_id: int
    config: TaskConfig
    clicks: List[ClickPoint] = field(default_factory=list)
    gaze_points: List[GazePoint] = field(default_factory=list)
    segments: List[SearchSegment] = field(default_factory=list)

    @property
    def total_duration_ms(self) -> float:
        """任务总时长（毫秒）"""
        if not self.gaze_points:
            return 0.0
        start = self.gaze_points[0].timestamp
        end = self.gaze_points[-1].timestamp
        return (end - start).total_seconds() * 1000

    @property
    def click_count(self) -> int:
        """点击次数"""
        return len(self.clicks)


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
