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
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import numpy as np
from tqdm import tqdm

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import DataLoader as GazeDataLoader
from src.data.preprocessor import GazePreprocessor
from src.data.schemas import GazePoint
from src.segmentation.event_segmenter import AdaptiveSegmenter


def load_task_level_labels(data_path: Path) -> Dict[Tuple[str, int], int]:
    """
    加载题目级分类标签
    
    Returns: {(subject_id, task_id): label_class}  label_class为0-based (0,1,2)
    """
    import pandas as pd
    
    task_labels = {}
    excel_path = data_path / '题号1到30_分类结果.xlsx'
    
    if not excel_path.exists():
        raise FileNotFoundError(f"任务级标签文件不存在: {excel_path}")
    
    print(f"加载任务级标签: {excel_path}")
    
    for task_id in range(1, 31):
        sheet_name = f'题号{task_id}'
        try:
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
            for _, row in df.iterrows():
                subject_id = str(int(row['被试编号']))
                # 将1/2/3转换为0/1/2
                label = int(row['class_1_2_3']) - 1
                task_labels[(subject_id, task_id)] = label
        except Exception as e:
            print(f"  警告: 读取{sheet_name}失败: {e}")
            
    print(f"成功加载任务级标签: {len(task_labels)} 条记录")
    return task_labels


def extract_features(
    gaze_points: List[GazePoint], 
    screen_width: int = 1920, 
    screen_height: int = 1080,
    is_first_segment: bool = True
) -> np.ndarray:
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
        is_first_segment: 是否是第一个片段（搜索第1个数字）。
                         如果是第一个片段，第0点设为0（任务刚开始）；
                         如果不是第一个片段，第0点基于第0-1点计算（从上一个点击位置开始移动）。

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
            if is_first_segment:
                # 第一个片段的第0点：任务刚开始，没有历史速度
                features[i, 2:7] = 0.0
            else:
                # 非第一个片段的第0点：从上一个点击位置开始，使用第0-1点计算初始速度
                # 第0点是起始点击位置，第1点是接下来的眼动点或目标点击
                if n >= 2:
                    next_point = gaze_points[1]
                    
                    # 计算从第0点到第1点的时间差和速度作为初始值
                    dt = (next_point.timestamp - point.timestamp).total_seconds() * 1000
                    dt = max(dt, 1.0)
                    features[i, 2] = dt
                    
                    # 位移
                    dx = next_point.x - point.x
                    dy = next_point.y - point.y
                    distance = math.sqrt(dx**2 + dy**2)
                    
                    # 初始速度
                    velocity = distance / dt
                    features[i, 3] = velocity
                    features[i, 4] = 0.0  # 初始加速度设为0（没有历史）
                    
                    # 初始方向
                    direction = math.atan2(dy, dx)
                    features[i, 5] = direction / math.pi
                    features[i, 6] = 0.0  # 初始方向变化设为0
                    
                    # 更新用于后续计算的 prev 值
                    prev_velocity = velocity
                    prev_direction = direction
                else:
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
            if i > 1 or (not is_first_segment and i == 1):
                # 对于非第一个片段，第1点也有方向变化（相对于第0点）
                direction_change = abs(direction - prev_direction)
                if direction_change > math.pi:
                    direction_change = 2 * math.pi - direction_change
                features[i, 6] = direction_change / math.pi

            prev_velocity = velocity
            prev_direction = direction

    return features


def normalize_subject_features(subject_data: Dict, clip_sigma: float = 10.0) -> Dict:
    """
    对单个被试的所有片段进行被试内标准化

    标准化动态特征（消除个体的速度/时间习惯差异）：
    - dt, velocity, acceleration: 使用被试内的均值和标准差标准化

    保持不变的特征：
    - x, y: 保留原始屏幕归一化值（保留空间信息）
    - direction, direction_change: 已经归一化到 [-1, 1] 和 [0, 1]

    异常值处理：
    - 使用10倍标准差裁剪，只裁剪极端异常值（如dt=1ms导致的超大速度）

    Args:
        subject_data: 单个被试的数据字典
        clip_sigma: 裁剪阈值（标准差倍数），默认10倍

    Returns:
        标准化后的被试数据字典
    """
    # 收集该被试所有片段的动态特征（包括第0点，因为非第一片段的第0点也有值）
    all_dt = []
    all_velocity = []
    all_acceleration = []

    for task in subject_data['tasks']:
        for segment in task['segments']:
            if len(segment) > 0:
                all_dt.extend(segment[:, 2].tolist())
                all_velocity.extend(segment[:, 3].tolist())
                all_acceleration.extend(segment[:, 4].tolist())

    if not all_dt:
        return subject_data

    # 计算该被试的统计量
    eps = 1e-8

    dt_mean = np.mean(all_dt)
    dt_std = np.std(all_dt) + eps
    velocity_mean = np.mean(all_velocity)
    velocity_std = np.std(all_velocity) + eps
    acceleration_mean = np.mean(all_acceleration)
    acceleration_std = np.std(all_acceleration) + eps

    # 应用标准化到所有片段（只标准化动态特征）
    for task in subject_data['tasks']:
        for i, segment in enumerate(task['segments']):
            if len(segment) > 0:
                normalized = segment.copy()

                # 只标准化 dt, velocity, acceleration（保留 x, y 不变）
                normalized[:, 2] = (segment[:, 2] - dt_mean) / dt_std
                normalized[:, 3] = (segment[:, 3] - velocity_mean) / velocity_std
                normalized[:, 4] = (segment[:, 4] - acceleration_mean) / acceleration_std

                # 极端异常值裁剪（防止dt=1ms等导致的极端值破坏训练）
                normalized[:, 2] = np.clip(normalized[:, 2], -clip_sigma, clip_sigma)
                normalized[:, 3] = np.clip(normalized[:, 3], -clip_sigma, clip_sigma)
                normalized[:, 4] = np.clip(normalized[:, 4], -clip_sigma, clip_sigma)

                # 第一个片段的第0点：任务刚开始，动态特征应为0（在extract_features中设为0，标准化后需恢复）
                if i == 0:
                    normalized[0, 2:7] = 0.0

                task['segments'][i] = normalized

    # 保存该被试的归一化统计量
    subject_data['normalization_stats'] = {
        'dt_mean': float(dt_mean), 'dt_std': float(dt_std),
        'velocity_mean': float(velocity_mean), 'velocity_std': float(velocity_std),
        'acceleration_mean': float(acceleration_mean), 'acceleration_std': float(acceleration_std),
    }

    return subject_data


def process_single_subject(
    subject_id: str,
    loader: GazeDataLoader,
    preprocessor: GazePreprocessor,
    screen_width: int,
    screen_height: int,
    task_labels: Dict[Tuple[str, int], int],  # 新增: 任务级标签
) -> Optional[Dict]:
    """
    处理单个被试，返回轻量级数据结构（包含任务级标签）
    """
    try:
        subject = loader.load_subject(subject_id)

        subject_data = {
            'subject_id': subject.subject_id,
            'label': float(subject.total_score),
            'category': int(subject.category),  # 被试级分类标签 (1/2/3)
            'tasks': []
        }

        for trial in subject.trials:
            preprocessor.preprocess_trial(trial)

            # 获取点击事件（来自 clicks，即 sheet3）
            click_points = trial.clicks
            if not click_points:
                continue

            # 获取眼动轨迹（来自 gaze_points，即 sheet4）
            gaze_trajectory = trial.gaze_points if trial.gaze_points else None

            # 分割：第一个参数是点击事件，第二个参数是眼动轨迹
            segmenter = AdaptiveSegmenter(
                task_config=trial.config,
                screen_width=screen_width,
                screen_height=screen_height,
            )
            segments = segmenter.segment(click_points, gaze_trajectory)

            if not segments:
                continue

            # 获取该任务的任务级标签
            task_label = task_labels.get((subject_id, trial.task_id), None)
            if task_label is None:
                print(f"  警告: 缺少任务级标签 {subject_id}/{trial.task_id}")
                continue

            task_data = {
                'task_id': trial.task_id,
                'task_label': task_label,  # 新增: 任务级标签 (0/1/2)
                'segments': [],
                # 任务条件（用于任务嵌入）
                'task_conditions': {
                    'grid_size': trial.config.grid_size,
                    'number_range': trial.config.number_range,
                    'click_disappear': trial.config.click_disappear,
                    'has_distractor': trial.config.has_distractor,
                    'distractor_count': trial.config.distractor_count,
                    'grid_distractor_count': trial.config.grid_distractor_count,
                    'number_distractor_count': trial.config.number_distractor_count,
                }
            }

            for seg_idx, segment in enumerate(segments):
                # 提取特征并立即转换为numpy
                # 第一个片段（搜索第1个数字）的第0点设为0，其他片段的第0点有初始速度
                is_first_segment = (seg_idx == 0)
                features = extract_features(
                    segment.gaze_points, 
                    screen_width, 
                    screen_height,
                    is_first_segment=is_first_segment
                )
                if len(features) > 0:
                    task_data['segments'].append(features)

            if task_data['segments']:
                subject_data['tasks'].append(task_data)

        if subject_data['tasks']:
            return subject_data

    except Exception as e:
        print(f"  Error processing {subject_id}: {e}")

    return None


def process_subject_worker(args: Tuple[str, str, int, int, Dict]) -> Optional[Dict]:
    """
    多进程 worker 函数，处理单个被试

    Args:
        args: (subject_id, data_dir, screen_width, screen_height, task_labels)

    Returns:
        处理后的被试数据字典，失败返回 None
    """
    subject_id, data_dir, screen_width, screen_height, task_labels = args

    # 每个进程创建自己的 loader 和 preprocessor
    loader = GazeDataLoader(data_dir)
    loader.load_labels()
    loader.load_tasks()
    preprocessor = GazePreprocessor(screen_width=screen_width, screen_height=screen_height)

    return process_single_subject(subject_id, loader, preprocessor, screen_width, screen_height, task_labels)


def main():
    parser = argparse.ArgumentParser(description='预处理眼动数据')
    parser.add_argument('--data_dir', type=str, default='data/gaze_trajectory_data',
                        help='原始数据目录')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                        help='输出目录')
    parser.add_argument('--screen_width', type=int, default=1920)
    parser.add_argument('--screen_height', type=int, default=1080)
    parser.add_argument('--subject_normalize', action='store_true',
                        help='使用被试内标准化（消除个体差异）')
    parser.add_argument('--workers', type=int, default=0,
                        help='并行处理的进程数（默认1，即串行）')
    args = parser.parse_args()

    # 自动检测 CPU 核心数
    max_workers = args.workers if args.workers > 0 else multiprocessing.cpu_count()

    print("=" * 60)
    print("眼动数据预处理")
    print("=" * 60)
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"被试内标准化: {'是' if args.subject_normalize else '否'}")
    print(f"并行进程数: {max_workers}")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载器（用于获取被试列表）
    loader = GazeDataLoader(args.data_dir)
    loader.load_labels()
    loader.load_tasks()

    subject_ids = loader.get_all_subject_ids()
    print(f"\n找到 {len(subject_ids)} 个被试")
    
    # 加载任务级标签
    print("\n加载任务级标签...")
    task_labels = load_task_level_labels(Path(args.data_dir))

    # 处理被试
    all_data = []
    failed = 0

    if max_workers == 1:
        # 串行处理
        preprocessor = GazePreprocessor(screen_width=args.screen_width, screen_height=args.screen_height)
        for subject_id in tqdm(subject_ids, desc="处理被试"):
            data = process_single_subject(
                subject_id, loader, preprocessor,
                args.screen_width, args.screen_height, task_labels
            )
            if data is not None:
                all_data.append(data)
            else:
                failed += 1

            # 定期垃圾回收
            if len(all_data) % 20 == 0:
                gc.collect()
    else:
        # 并行处理
        worker_args = [
            (subject_id, args.data_dir, args.screen_width, args.screen_height, task_labels)
            for subject_id in subject_ids
        ]

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_subject_worker, arg): arg[0] for arg in worker_args}

            for future in tqdm(as_completed(futures), total=len(futures), desc="处理被试"):
                subject_id = futures[future]
                try:
                    data = future.result()
                    if data is not None:
                        all_data.append(data)
                    else:
                        failed += 1
                except Exception as e:
                    print(f"  Error processing {subject_id}: {e}")
                    failed += 1

    print(f"\n处理完成: {len(all_data)} 成功, {failed} 失败")

    # 被试内标准化
    if args.subject_normalize:
        print("\n应用被试内标准化...")
        for i, subject_data in enumerate(tqdm(all_data, desc="标准化")):
            all_data[i] = normalize_subject_features(subject_data)
        print("被试内标准化完成")

    # 统计
    total_tasks = sum(len(d['tasks']) for d in all_data)
    total_segments = sum(
        sum(len(t['segments']) for t in d['tasks'])
        for d in all_data
    )
    print(f"总任务数: {total_tasks}")
    print(f"总片段数: {total_segments}")

    # 保存
    if args.subject_normalize:
        output_file = output_dir / 'processed_data_subject_norm.pkl'
    else:
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
