# -*- coding: utf-8 -*-
"""
数据加载模块

负责从 Excel 文件加载眼动追踪数据
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Tuple

import pandas as pd

from .schemas import TaskConfig, TaskTrial, SubjectData

logger = logging.getLogger(__name__)


class DataLoader:
    """
    数据加载器

    负责加载个人标签、题目信息和眼动轨迹数据
    """

    def __init__(self, data_path: Union[str, Path]):
        """
        初始化数据加载器

        Args:
            data_path: 数据根目录路径（包含个人标签.xlsx、题目信息.xlsx和被试文件夹）
        """
        self.data_path = Path(data_path)
        self._labels_df: Optional[pd.DataFrame] = None
        self._tasks_df: Optional[pd.DataFrame] = None
        self._task_configs: Dict[int, TaskConfig] = {}

    def load_labels(self) -> pd.DataFrame:
        """
        加载个人标签数据

        Returns:
            DataFrame，包含列：[被试编号, 总分, 类别(μ±1σ)]
        """
        path = self.data_path / '个人标签.xlsx'
        if not path.exists():
            raise FileNotFoundError(f"标签文件不存在: {path}")

        self._labels_df = pd.read_excel(path)
        logger.info(f"加载标签数据: {len(self._labels_df)} 条记录")
        return self._labels_df

    def load_tasks(self) -> pd.DataFrame:
        """
        加载题目信息

        Returns:
            DataFrame，包含列：[题目, 方格数量, 数字范围, 点击是否消失, 是否有干扰项, 干扰项总数量]
        """
        path = self.data_path / '题目信息.xlsx'
        if not path.exists():
            raise FileNotFoundError(f"题目信息文件不存在: {path}")

        self._tasks_df = pd.read_excel(path)
        logger.info(f"加载题目信息: {len(self._tasks_df)} 条记录")

        # 构建任务配置映射
        self._build_task_configs()

        return self._tasks_df

    def _build_task_configs(self) -> None:
        """从题目信息构建任务配置映射"""
        if self._tasks_df is None:
            return

        for _, row in self._tasks_df.iterrows():
            task_id = int(row.get('题目', row.name + 1))

            # 解析方格数量（支持 "3×3", "5x5", "25" 等格式）
            grid_size = self._parse_grid_size(row.get('方格数量', 25))

            # 解析数字范围
            number_range_str = row.get('数字范围', '1-25')
            try:
                if isinstance(number_range_str, str) and '-' in number_range_str:
                    parts = number_range_str.split('-')
                    number_range = (int(parts[0]), int(parts[1]))
                else:
                    number_range = (1, grid_size)
            except (ValueError, IndexError):
                number_range = (1, grid_size)

            # 解析布尔值
            click_disappear = self._parse_bool(row.get('点击是否消失', False))
            has_distractor = self._parse_bool(row.get('是否有干扰项', False))

            # 解析干扰项数量
            distractor_count = int(row.get('干扰项总数量', 0) or 0)

            config = TaskConfig(
                task_id=task_id,
                grid_size=grid_size,
                number_range=number_range,
                click_disappear=click_disappear,
                has_distractor=has_distractor,
                distractor_count=distractor_count,
            )
            self._task_configs[task_id] = config

    @staticmethod
    def _parse_grid_size(value: Any) -> int:
        """
        解析方格数量

        支持格式：
        - 整数：25
        - 字符串 "3×3", "5x5", "3*3"
        """
        if isinstance(value, (int, float)):
            return int(value)

        if isinstance(value, str):
            value = value.strip()
            # 尝试匹配 "3×3", "5x5", "3*3" 格式
            import re
            match = re.match(r'(\d+)\s*[×xX*]\s*(\d+)', value)
            if match:
                rows = int(match.group(1))
                cols = int(match.group(2))
                return rows * cols
            # 尝试直接转换为整数
            try:
                return int(value)
            except ValueError:
                pass

        return 25  # 默认 5x5

    @staticmethod
    def _parse_bool(value: Any) -> bool:
        """解析布尔值"""
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.lower() in ('是', 'yes', 'true', '1')
        return False

    def get_task_config(self, task_id: int) -> TaskConfig:
        """
        获取任务配置

        Args:
            task_id: 题目编号

        Returns:
            TaskConfig 对象
        """
        if task_id in self._task_configs:
            return self._task_configs[task_id]
        # 返回默认配置
        return TaskConfig(task_id=task_id)

    def load_gaze_trajectory(
        self, subject_id: str, task_id: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        加载单个轨迹文件（包含点击数据和眼动轨迹）

        Args:
            subject_id: 被试编号
            task_id: 题目编号

        Returns:
            (click_df, gaze_df) 元组:
            - click_df: 点击数据，列：[时间, 正确, 坐标X, 坐标Y]
            - gaze_df: 眼动轨迹，列：[时间, 坐标X, 坐标Y]
        """
        path = self.data_path / str(subject_id) / f'{task_id}.xlsx'
        if not path.exists():
            raise FileNotFoundError(f"轨迹文件不存在: {path}")

        # 加载点击数据（sheet3）
        click_df = pd.read_excel(path, sheet_name='sheet3')

        # 加载眼动轨迹（sheet4）
        try:
            gaze_df = pd.read_excel(path, sheet_name='sheet4')
        except Exception as e:
            logger.warning(f"无法加载眼动数据 {path}: {e}")
            gaze_df = pd.DataFrame(columns=['时间', '坐标X', '坐标Y'])

        return click_df, gaze_df

    def get_all_subject_ids(self) -> List[str]:
        """
        获取所有被试编号

        Returns:
            被试编号列表
        """
        # 从标签文件获取
        if self._labels_df is not None and '被试编号' in self._labels_df.columns:
            return [str(sid) for sid in self._labels_df['被试编号'].tolist()]

        # 从目录结构获取
        subject_ids = []
        for d in self.data_path.iterdir():
            if d.is_dir() and d.name.isdigit():
                subject_ids.append(d.name)
        return sorted(subject_ids)

    def get_subject_label(self, subject_id: str) -> Tuple[float, int]:
        """
        获取被试的标签信息

        Args:
            subject_id: 被试编号

        Returns:
            (总分, 类别) 元组
        """
        if self._labels_df is None:
            self.load_labels()

        # 尝试匹配被试编号
        mask = self._labels_df['被试编号'].astype(str) == str(subject_id)
        if not mask.any():
            raise ValueError(f"找不到被试 {subject_id} 的标签")

        row = self._labels_df[mask].iloc[0]

        # 获取总分
        total_score = float(row.get('总分', 0))

        # 获取类别
        category_col = None
        for col in self._labels_df.columns:
            if '类别' in col:
                category_col = col
                break

        category = int(row.get(category_col, 2) if category_col else 2)

        return total_score, category

    def load_subject(self, subject_id: str) -> SubjectData:
        """
        加载完整的被试数据

        Args:
            subject_id: 被试编号

        Returns:
            SubjectData 对象
        """
        # 获取标签
        total_score, category = self.get_subject_label(subject_id)

        subject = SubjectData(
            subject_id=subject_id,
            total_score=total_score,
            category=category,
        )

        # 加载所有试次
        for task_id in range(1, 31):
            try:
                click_df, gaze_df = self.load_gaze_trajectory(subject_id, task_id)
                config = self.get_task_config(task_id)

                trial = TaskTrial(
                    subject_id=subject_id,
                    task_id=task_id,
                    config=config,
                )
                # 存储原始 DataFrame（点击和眼动）
                trial._raw_click_df = click_df
                trial._raw_gaze_df = gaze_df

                subject.trials.append(trial)
            except FileNotFoundError:
                logger.debug(f"跳过缺失的轨迹文件: {subject_id}/{task_id}.xlsx")
                continue

        logger.info(f"加载被试 {subject_id}: {len(subject.trials)} 个试次")
        return subject
