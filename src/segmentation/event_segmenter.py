# -*- coding: utf-8 -*-
"""
事件分割模块

将点击序列切分为"搜索片段"，并整合连续眼动轨迹数据。

数据结构：
- sheet3: 点击事件（时间、正确、坐标X、坐标Y）
- sheet4: 连续眼动轨迹（时间、坐标X、坐标Y），约53Hz采样
"""

import logging
import math
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import numpy as np

from src.data.schemas import ClickPoint, GazePoint, SearchSegment, TaskConfig

logger = logging.getLogger(__name__)


class EventSegmenter:
    """
    事件分割器

    将点击序列切分为搜索片段，并整合连续眼动轨迹。

    每个片段表示从 Click(N-1) 到 Click(N) 的搜索过程，
    包含两个点击事件以及期间的连续眼动轨迹（如果有的话）。
    """

    def __init__(
        self,
        task_config: TaskConfig,
        grid_layout: Optional[Dict[int, Tuple[float, float]]] = None,
        screen_width: int = 1920,
        screen_height: int = 1080,
    ):
        """
        初始化分割器

        Args:
            task_config: 任务配置
            grid_layout: 数字到位置的映射 {数字: (x, y)}
            screen_width: 屏幕宽度
            screen_height: 屏幕高度
        """
        self.task_config = task_config
        self.screen_width = screen_width
        self.screen_height = screen_height

        # 如果未提供布局，自动计算
        if grid_layout is None:
            self.grid_layout = self.calculate_grid_layout(
                task_config.grid_size,
                screen_width,
                screen_height,
            )
        else:
            self.grid_layout = grid_layout

    def segment(
        self,
        clicks: List[ClickPoint],
        gaze_trajectory: Optional[List[GazePoint]] = None,
    ) -> List[SearchSegment]:
        """
        主分割方法

        算法流程:
        1. 从点击事件创建片段边界
        2. 将眼动轨迹数据填充到对应的片段中
        3. 片段包含起始点击、期间眼动点和目标点击

        Args:
            clicks: 点击事件列表
            gaze_trajectory: 连续眼动轨迹列表（可选）

        Returns:
            搜索片段列表
        """
        if not clicks:
            return []

        # 按时间排序点击事件
        sorted_clicks = sorted(clicks, key=lambda p: p.timestamp)

        if len(sorted_clicks) < 2:
            logger.warning(f"点击数量不足: {len(sorted_clicks)}")
            return []

        # 预排序眼动轨迹（如果有）
        gaze_sorted = []
        if gaze_trajectory:
            gaze_sorted = sorted(gaze_trajectory, key=lambda p: p.timestamp)

        # 创建片段
        segments = []
        clicked_positions: List[Tuple[float, float]] = []

        # 为每对相邻点击创建搜索片段
        for i in range(len(sorted_clicks) - 1):
            prev_click = sorted_clicks[i]
            curr_click = sorted_clicks[i + 1]

            # 获取目标位置
            target_pos = self._get_target_position(curr_click)

            # 提取期间的眼动轨迹
            if gaze_sorted:
                intermediate_gaze = self._extract_between(
                    gaze_sorted, prev_click.timestamp, curr_click.timestamp
                )
                # 片段包含：起始点击 + 中间眼动 + 目标点击
                segment_gaze_points = self._merge_clicks_and_gaze(prev_click, intermediate_gaze, curr_click)
            else:
                # 无眼动数据时只包含两个点击（转换为 GazePoint）
                segment_gaze_points = [
                    GazePoint(timestamp=prev_click.timestamp, x=prev_click.x, y=prev_click.y),
                    GazePoint(timestamp=curr_click.timestamp, x=curr_click.x, y=curr_click.y),
                ]

            segment = SearchSegment(
                segment_id=i,
                start_number=prev_click.target_number,
                target_number=curr_click.target_number,
                gaze_points=segment_gaze_points,
                start_time=prev_click.timestamp,
                end_time=curr_click.timestamp,
                target_position=target_pos,
                clicked_positions=clicked_positions.copy(),
            )
            segments.append(segment)

            # 更新已点击位置
            clicked_positions.append((prev_click.x, prev_click.y))

        logger.debug(
            f"分割完成: {len(segments)} 个片段, "
            f"平均眼动点数: {sum(len(s.gaze_points) for s in segments) / len(segments):.1f}"
        )
        return segments

    def _extract_between(
        self,
        gaze_points: List[GazePoint],
        start_time: datetime,
        end_time: datetime,
    ) -> List[GazePoint]:
        """
        提取两个时间点之间的眼动点

        Args:
            gaze_points: 眼动点列表
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            中间的眼动点列表
        """
        return [
            p for p in gaze_points
            if start_time < p.timestamp < end_time
        ]

    def _merge_clicks_and_gaze(
        self,
        start_click: ClickPoint,
        gaze_points: List[GazePoint],
        end_click: ClickPoint,
    ) -> List[GazePoint]:
        """
        合并点击和眼动轨迹点

        Args:
            start_click: 起始点击
            gaze_points: 中间眼动点
            end_click: 结束点击

        Returns:
            合并后的眼动点列表（含点击）
        """
        points = []

        # 起始点击转为 GazePoint
        points.append(GazePoint(
            timestamp=start_click.timestamp,
            x=start_click.x,
            y=start_click.y,
        ))

        # 中间眼动点
        points.extend(gaze_points)

        # 结束点击转为 GazePoint
        points.append(GazePoint(
            timestamp=end_click.timestamp,
            x=end_click.x,
            y=end_click.y,
        ))

        return points

    def _get_target_position(self, click: ClickPoint) -> Tuple[float, float]:
        """
        获取目标位置

        优先使用网格布局，否则使用点击坐标

        Args:
            click: 点击事件

        Returns:
            目标位置 (x, y)
        """
        if click.target_number in self.grid_layout:
            return self.grid_layout[click.target_number]
        # 使用点击坐标作为近似
        return (click.x, click.y)

    @staticmethod
    def calculate_grid_layout(
        grid_size: int,
        screen_width: int = 1920,
        screen_height: int = 1080,
        margin_ratio: float = 0.15,
    ) -> Dict[int, Tuple[float, float]]:
        """
        计算方格布局

        假设数字按顺序从左到右、从上到下排列在方格中

        Args:
            grid_size: 方格总数（如 25 对应 5x5）
            screen_width: 屏幕宽度
            screen_height: 屏幕高度
            margin_ratio: 边距比例

        Returns:
            数字到位置的映射 {数字: (center_x, center_y)}
        """
        # 计算网格维度
        n = int(math.ceil(math.sqrt(grid_size)))

        # 计算有效区域（居中显示）
        grid_width = min(screen_width, screen_height) * (1 - 2 * margin_ratio)
        grid_height = grid_width  # 保持正方形

        # 计算偏移（居中）
        offset_x = (screen_width - grid_width) / 2
        offset_y = (screen_height - grid_height) / 2

        # 计算单元格大小
        cell_width = grid_width / n
        cell_height = grid_height / n

        layout = {}
        for num in range(1, grid_size + 1):
            # 计算行列（0-indexed）
            row = (num - 1) // n
            col = (num - 1) % n

            # 计算中心位置
            center_x = offset_x + (col + 0.5) * cell_width
            center_y = offset_y + (row + 0.5) * cell_height

            layout[num] = (center_x, center_y)

        return layout

    @staticmethod
    def calculate_grid_layout_from_clicks(
        clicks: List[ClickPoint],
    ) -> Dict[int, Tuple[float, float]]:
        """
        从点击坐标推断网格布局

        Args:
            clicks: 点击事件列表

        Returns:
            数字到位置的映射
        """
        layout = {}
        for click in clicks:
            layout[click.target_number] = (click.x, click.y)
        return layout


class AdaptiveSegmenter(EventSegmenter):
    """
    自适应分割器

    扩展基础分割器，支持从点击坐标自动学习网格布局
    """

    def segment(
        self,
        clicks: List[ClickPoint],
        gaze_trajectory: Optional[List[GazePoint]] = None,
    ) -> List[SearchSegment]:
        """
        自适应分割

        如果网格布局不完整，从点击坐标学习
        """
        # 如果网格布局不完整，从点击坐标补充
        for click in clicks:
            if click.target_number not in self.grid_layout:
                self.grid_layout[click.target_number] = (click.x, click.y)

        return super().segment(clicks, gaze_trajectory)
