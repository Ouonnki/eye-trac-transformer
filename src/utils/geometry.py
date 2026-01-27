# -*- coding: utf-8 -*-
"""
几何计算工具模块
"""

import math
from typing import Tuple


def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    计算两点之间的欧氏距离

    Args:
        p1: 点1坐标 (x, y)
        p2: 点2坐标 (x, y)

    Returns:
        欧氏距离
    """
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def distance_to_point(point: Tuple[float, float], target: Tuple[float, float]) -> float:
    """
    计算点到目标位置的欧氏距离

    Args:
        point: 点坐标 (x, y)
        target: 目标位置 (x, y)

    Returns:
        欧氏距离
    """
    return math.sqrt((point[0] - target[0]) ** 2 + (point[1] - target[1]) ** 2)
