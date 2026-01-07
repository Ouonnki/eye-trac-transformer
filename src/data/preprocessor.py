# -*- coding: utf-8 -*-
"""
数据预处理模块

负责解析时间戳、清洗坐标、识别点击事件
"""

import logging
import re
from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np

from .schemas import GazePoint, TaskTrial

logger = logging.getLogger(__name__)


class GazePreprocessor:
    """
    眼动数据预处理器

    负责将原始 DataFrame 转换为 GazePoint 列表
    """

    def __init__(
        self,
        screen_width: int = 1920,
        screen_height: int = 1080,
    ):
        """
        初始化预处理器

        Args:
            screen_width: 屏幕宽度（像素）
            screen_height: 屏幕高度（像素）
        """
        self.screen_width = screen_width
        self.screen_height = screen_height

        # 时间戳解析正则表达式
        self._time_patterns = [
            # 格式: 2025/10/2 18:58:11:842
            re.compile(r'(\d{4})/(\d{1,2})/(\d{1,2})\s+(\d{1,2}):(\d{2}):(\d{2}):(\d{1,3})'),
            # 格式: 2025-10-02 18:58:11.842
            re.compile(r'(\d{4})-(\d{1,2})-(\d{1,2})\s+(\d{1,2}):(\d{2}):(\d{2})\.(\d{1,3})'),
        ]

    def preprocess_trial(self, trial: TaskTrial) -> TaskTrial:
        """
        预处理单个试次

        Args:
            trial: 包含 _raw_click_df 和 _raw_gaze_df 的 TaskTrial 对象

        Returns:
            预处理后的 TaskTrial 对象
        """
        # 处理点击数据
        if hasattr(trial, '_raw_click_df') and trial._raw_click_df is not None:
            click_points = self.preprocess(trial._raw_click_df, is_click_data=True)
            trial.raw_gaze_points = click_points
            delattr(trial, '_raw_click_df')
        elif hasattr(trial, '_raw_df') and trial._raw_df is not None:
            # 兼容旧格式
            click_points = self.preprocess(trial._raw_df, is_click_data=True)
            trial.raw_gaze_points = click_points
            delattr(trial, '_raw_df')

        # 处理眼动轨迹数据
        if hasattr(trial, '_raw_gaze_df') and trial._raw_gaze_df is not None:
            gaze_df = trial._raw_gaze_df
            if not gaze_df.empty:
                gaze_points = self.preprocess_gaze_trajectory(gaze_df)
                trial.gaze_trajectory = gaze_points
            else:
                trial.gaze_trajectory = []
            delattr(trial, '_raw_gaze_df')
        else:
            trial.gaze_trajectory = []

        return trial

    def preprocess(
        self, df: pd.DataFrame, is_click_data: bool = False
    ) -> List[GazePoint]:
        """
        预处理原始眼动数据

        处理步骤:
        1. 解析时间戳
        2. 清洗坐标值
        3. 识别点击事件
        4. 转换为 GazePoint 对象

        Args:
            df: 原始 DataFrame，包含列 [时间, 正确, 坐标X, 坐标Y]
            is_click_data: 是否为点击数据（所有行都是点击事件）

        Returns:
            GazePoint 列表
        """
        gaze_points = []

        # 确定列名
        time_col = self._find_column(df, ['时间', 'Timestamp', 'Time'])
        x_col = self._find_column(df, ['坐标X', 'X', 'x'])
        y_col = self._find_column(df, ['坐标Y', 'Y', 'y'])
        correct_col = self._find_column(df, ['正确', 'Correct', 'correct'])

        if time_col is None or x_col is None or y_col is None:
            logger.warning("无法找到必要的列")
            return gaze_points

        click_number = 0  # 当前点击的目标数字

        for idx, row in df.iterrows():
            # 1. 解析时间戳
            timestamp = self._parse_timestamp(row[time_col])
            if timestamp is None:
                continue

            # 2. 清洗坐标
            x = self._clean_coordinate(row[x_col], self.screen_width)
            y = self._clean_coordinate(row[y_col], self.screen_height)

            # 3. 识别点击事件
            if is_click_data:
                # 点击数据：所有行都是点击事件
                click_number += 1
                is_click = True
                target_number = click_number
            else:
                is_click, target_number = self._detect_click_event(
                    row, correct_col, click_number
                )
                if is_click and target_number is not None:
                    click_number = target_number

            point = GazePoint(
                timestamp=timestamp,
                x=x,
                y=y,
                is_click=is_click,
                target_number=target_number if is_click else None,
            )
            gaze_points.append(point)

        logger.debug(
            f"预处理完成: {len(gaze_points)} 个点, "
            f"{sum(1 for p in gaze_points if p.is_click)} 个点击"
        )
        return gaze_points

    def preprocess_gaze_trajectory(self, df: pd.DataFrame) -> List[GazePoint]:
        """
        预处理连续眼动轨迹数据（sheet4）

        Args:
            df: 眼动轨迹 DataFrame，列：[时间, 坐标X, 坐标Y]

        Returns:
            GazePoint 列表（所有点的 is_click=False）
        """
        gaze_points = []

        # 确定列名
        time_col = self._find_column(df, ['时间', 'Timestamp', 'Time'])
        x_col = self._find_column(df, ['坐标X', 'X', 'x'])
        y_col = self._find_column(df, ['坐标Y', 'Y', 'y'])

        if time_col is None or x_col is None or y_col is None:
            logger.warning("眼动轨迹数据缺少必要的列")
            return gaze_points

        for idx, row in df.iterrows():
            # 解析时间戳
            timestamp = self._parse_timestamp(row[time_col])
            if timestamp is None:
                continue

            # 清洗坐标（眼动数据可能超出屏幕范围，不强制裁剪）
            try:
                x = float(row[x_col]) if not pd.isna(row[x_col]) else 0.0
                y = float(row[y_col]) if not pd.isna(row[y_col]) else 0.0
            except (ValueError, TypeError):
                continue

            point = GazePoint(
                timestamp=timestamp,
                x=x,
                y=y,
                is_click=False,
                target_number=None,
            )
            gaze_points.append(point)

        logger.debug(f"眼动轨迹预处理完成: {len(gaze_points)} 个注视点")
        return gaze_points

    @staticmethod
    def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """查找匹配的列名"""
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def _parse_timestamp(self, time_str) -> Optional[datetime]:
        """
        解析时间戳字符串

        支持格式:
        - 2025/10/2 18:58:11:842
        - 2025-10-02 18:58:11.842
        """
        if pd.isna(time_str):
            return None

        time_str = str(time_str).strip()

        for pattern in self._time_patterns:
            match = pattern.match(time_str)
            if match:
                groups = match.groups()
                year = int(groups[0])
                month = int(groups[1])
                day = int(groups[2])
                hour = int(groups[3])
                minute = int(groups[4])
                second = int(groups[5])
                # 毫秒部分，补齐到3位
                ms_str = groups[6].ljust(3, '0')[:3]
                milliseconds = int(ms_str)

                try:
                    return datetime(
                        year, month, day, hour, minute, second,
                        milliseconds * 1000  # 转换为微秒
                    )
                except ValueError:
                    continue

        # 尝试使用 pandas 解析
        try:
            return pd.to_datetime(time_str).to_pydatetime()
        except Exception:
            logger.debug(f"无法解析时间戳: {time_str}")
            return None

    def _clean_coordinate(self, value, max_value: int) -> float:
        """
        坐标值清洗

        - 处理 NaN
        - 裁剪到有效范围 [0, max_value]
        """
        if pd.isna(value):
            return 0.0

        try:
            val = float(value)
        except (ValueError, TypeError):
            return 0.0

        # 裁剪到有效范围
        return max(0.0, min(val, float(max_value)))

    def _detect_click_event(
        self,
        row: pd.Series,
        correct_col: Optional[str],
        current_number: int
    ) -> Tuple[bool, Optional[int]]:
        """
        检测点击事件

        策略：
        1. 如果"正确"列有有效值，则为点击事件
        2. 目标数字从正确列推断，或递增

        Args:
            row: 数据行
            correct_col: "正确"列名
            current_number: 当前已点击的数字

        Returns:
            (is_click, target_number) 元组
        """
        if correct_col is None:
            return False, None

        correct_value = row.get(correct_col)

        # 检查是否为有效点击
        if pd.isna(correct_value) or correct_value == '':
            return False, None

        # 尝试解析目标数字
        try:
            if isinstance(correct_value, (int, float)):
                target_number = int(correct_value)
            elif isinstance(correct_value, str):
                # 可能是 "正确" 或数字
                if correct_value.isdigit():
                    target_number = int(correct_value)
                elif correct_value in ('正确', 'true', 'True', '1'):
                    target_number = current_number + 1
                else:
                    target_number = current_number + 1
            else:
                target_number = current_number + 1
        except (ValueError, TypeError):
            target_number = current_number + 1

        return True, target_number


class GazeDataCleaner:
    """
    眼动数据清洗器

    提供额外的数据清洗功能
    """

    @staticmethod
    def remove_outliers(
        gaze_points: List[GazePoint],
        velocity_threshold: float = 2000.0
    ) -> List[GazePoint]:
        """
        移除速度异常的眼动点

        Args:
            gaze_points: 眼动点列表
            velocity_threshold: 速度阈值（像素/秒）

        Returns:
            清洗后的眼动点列表
        """
        if len(gaze_points) < 2:
            return gaze_points

        cleaned = [gaze_points[0]]

        for i in range(1, len(gaze_points)):
            prev = cleaned[-1]
            curr = gaze_points[i]

            # 计算时间差（秒）
            dt = (curr.timestamp - prev.timestamp).total_seconds()
            if dt <= 0:
                continue

            # 计算距离
            dist = prev.distance_to(curr)

            # 计算速度
            velocity = dist / dt

            # 如果速度正常或是点击事件，保留该点
            if velocity <= velocity_threshold or curr.is_click:
                cleaned.append(curr)

        return cleaned

    @staticmethod
    def interpolate_missing(
        gaze_points: List[GazePoint],
        max_gap_ms: float = 100.0
    ) -> List[GazePoint]:
        """
        插值填补缺失的眼动点

        Args:
            gaze_points: 眼动点列表
            max_gap_ms: 最大允许的时间间隙（毫秒）

        Returns:
            插值后的眼动点列表
        """
        if len(gaze_points) < 2:
            return gaze_points

        interpolated = [gaze_points[0]]

        for i in range(1, len(gaze_points)):
            prev = interpolated[-1]
            curr = gaze_points[i]

            # 计算时间差（毫秒）
            dt_ms = (curr.timestamp - prev.timestamp).total_seconds() * 1000

            # 如果间隙过大，插入中间点
            if dt_ms > max_gap_ms and not curr.is_click:
                num_points = int(dt_ms / max_gap_ms)
                for j in range(1, num_points):
                    ratio = j / num_points
                    # 线性插值
                    new_x = prev.x + (curr.x - prev.x) * ratio
                    new_y = prev.y + (curr.y - prev.y) * ratio
                    new_time = prev.timestamp + (curr.timestamp - prev.timestamp) * ratio

                    interpolated.append(GazePoint(
                        timestamp=new_time,
                        x=new_x,
                        y=new_y,
                        is_click=False,
                    ))

            interpolated.append(curr)

        return interpolated
