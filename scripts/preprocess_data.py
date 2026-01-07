# -*- coding: utf-8 -*-
"""
数据预处理脚本

将原始眼动数据转换为轻量级的numpy格式，供深度学习训练使用。
处理完成后数据约几十MB，可以快速加载。

使用方法：
    python scripts/preprocess_data.py --data_dir /data/gaze_trajectory_data --output_dir /data/processed

输出格式：
    {
        'subject_id': str,
        'label': float,
        'tasks': [
            {
                'task_id': int,
                'segments': [np.array(shape=(seq_len, 7)), ...]
            },
            ...
        ]
    }
"""

import os
import sys
import argparse
import pickle
import gc
import math
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from tqdm import tqdm

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import DataLoader as GazeDataLoader
from src.data.preprocessor import GazePreprocessor
from src.data.schemas import GazePoint
from src.segmentation.event_segmenter import AdaptiveSegmenter


def extract_features(gaze_points: List[GazePoint], screen_width: int = 1920, screen_height: int = 1080) -> np.ndarray:
    """
    从眼动点序列提取7维特征

    Features:
        - x: 归一化X坐标 [0, 1]
        - y: 归一化Y坐标 [0, 1]
        - dt: 时间差（毫秒）
        - velocity: 瞬时速度
        - acceleration: 瞬时加速度
        - direction: 移动方向（归一化到[-1, 1]）
        - direction_change: 方向变化量

    Args:
        gaze_points: 眼动点列表
        screen_width: 屏幕宽度
        screen_height: 屏幕高度

    Returns:
        (seq_len, 7) 的特征数组
    """
    n = len(gaze_points)
    if n < 2:
        return np.zeros((0, 7), dtype=np.float32)

    features = np.zeros((n, 7), dtype=np.float32)
    prev_velocity = 0.0
    prev_direction = 0.0

    for i, point in enumerate(gaze_points):
        # 基础坐标归一化
        features[i, 0] = point.x / screen_width
        features[i, 1] = point.y / screen_height

        if i == 0:
            features[i, 2:7] = 0.0
        else:
            prev_point = gaze_points[i - 1]

            # 时间差（毫秒）
            dt = (point.timestamp - prev_point.timestamp).total_seconds() * 1000
            dt = max(dt, 1.0)
            features[i, 2] = dt

            # 位移
            dx = point.x - prev_point.x
            dy = point.y - prev_point.y
            distance = math.sqrt(dx**2 + dy**2)

            # 速度
            velocity = distance / dt
            features[i, 3] = velocity

            # 加速度
            acceleration = (velocity - prev_velocity) / dt
            features[i, 4] = acceleration

            # 方向
            direction = math.atan2(dy, dx)
            features[i, 5] = direction / math.pi

            # 方向变化
            if i > 1:
                direction_change = abs(direction - prev_direction)
                if direction_change > math.pi:
                    direction_change = 2 * math.pi - direction_change
                features[i, 6] = direction_change / math.pi

            prev_velocity = velocity
            prev_direction = direction

    return features


def process_single_subject(
    subject_id: str,
    loader: GazeDataLoader,
    preprocessor: GazePreprocessor,
    screen_width: int,
    screen_height: int,
) -> Optional[Dict]:
    """
    处理单个被试，返回轻量级数据结构
    """
    try:
        subject = loader.load_subject(subject_id)

        subject_data = {
            'subject_id': subject.subject_id,
            'label': float(subject.total_score),
            'tasks': []
        }

        for trial in subject.trials:
            preprocessor.preprocess_trial(trial)

            # 获取眼动点
            gaze_trajectory = getattr(trial, 'gaze_trajectory', None)
            gaze_points = gaze_trajectory if gaze_trajectory else trial.raw_gaze_points

            if not gaze_points:
                continue

            # 分割
            segmenter = AdaptiveSegmenter(
                task_config=trial.config,
                screen_width=screen_width,
                screen_height=screen_height,
            )
            segments = segmenter.segment(gaze_points, gaze_trajectory)

            if not segments:
                continue

            task_data = {
                'task_id': trial.task_id,
                'segments': []
            }

            for segment in segments:
                # 提取特征并立即转换为numpy
                features = extract_features(segment.gaze_points, screen_width, screen_height)
                if len(features) > 0:
                    task_data['segments'].append(features)

            if task_data['segments']:
                subject_data['tasks'].append(task_data)

        if subject_data['tasks']:
            return subject_data

    except Exception as e:
        print(f"  Error processing {subject_id}: {e}")

    return None


def main():
    parser = argparse.ArgumentParser(description='预处理眼动数据')
    parser.add_argument('--data_dir', type=str, default='data/gaze_trajectory_data',
                        help='原始数据目录')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                        help='输出目录')
    parser.add_argument('--screen_width', type=int, default=1920)
    parser.add_argument('--screen_height', type=int, default=1080)
    args = parser.parse_args()

    print("=" * 60)
    print("眼动数据预处理")
    print("=" * 60)
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载器
    loader = GazeDataLoader(args.data_dir)
    loader.load_labels()
    loader.load_tasks()
    preprocessor = GazePreprocessor(screen_width=args.screen_width, screen_height=args.screen_height)

    subject_ids = loader.get_all_subject_ids()
    print(f"\n找到 {len(subject_ids)} 个被试")

    # 逐个处理
    all_data = []
    failed = 0

    for subject_id in tqdm(subject_ids, desc="处理被试"):
        data = process_single_subject(
            subject_id, loader, preprocessor,
            args.screen_width, args.screen_height
        )
        if data is not None:
            all_data.append(data)
        else:
            failed += 1

        # 定期垃圾回收
        if len(all_data) % 20 == 0:
            gc.collect()

    print(f"\n处理完成: {len(all_data)} 成功, {failed} 失败")

    # 统计
    total_tasks = sum(len(d['tasks']) for d in all_data)
    total_segments = sum(
        sum(len(t['segments']) for t in d['tasks'])
        for d in all_data
    )
    print(f"总任务数: {total_tasks}")
    print(f"总片段数: {total_segments}")

    # 保存
    output_file = output_dir / 'processed_data.pkl'
    print(f"\n保存到: {output_file}")

    with open(output_file, 'wb') as f:
        pickle.dump(all_data, f)

    # 检查文件大小
    file_size = output_file.stat().st_size / (1024 * 1024)
    print(f"文件大小: {file_size:.1f} MB")

    print("\n预处理完成!")


if __name__ == '__main__':
    main()
