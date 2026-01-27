# -*- coding: utf-8 -*-
"""
数据划分策略模块

提供 2×2 矩阵划分和 K-Fold 交叉验证的数据划分策略。
"""

import logging
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
import copy

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SplitInfo:
    """划分信息"""
    name: str  # 划分名称：train, test1, test2, test3
    description: str  # 描述
    subjects: List[str]  # 被试ID列表
    task_ids: Set[str]  # 任务ID集合
    n_subjects: int = 0  # 被试数量
    n_tasks: int = 0  # 任务数量

    def __post_init__(self):
        self.n_subjects = len(self.subjects)
        self.n_tasks = len(self.task_ids)


class TwoByTwoSplitter:
    """
    2×2 矩阵数据划分器

    将数据按被试和任务两个维度划分为4个部分：
    - train: 前K人 × 前N题（训练集）
    - test1: 后L人 × 前N题（新人旧题，跨被试泛化）
    - test2: 前K人 × 后M题（旧人新题，跨任务泛化）
    - test3: 后L人 × 后M题（新人新题，双重泛化）
    """

    def __init__(
        self,
        train_subjects: int = 100,
        train_tasks: int = 20,
        random_state: int = 42,
    ):
        """
        初始化

        Args:
            train_subjects: 训练集被试数量
            train_tasks: 训练集任务数量（按 task_id 排序后取前N个）
            random_state: 随机种子（用于被试随机打乱）
        """
        self.train_subjects = train_subjects
        self.train_tasks = train_tasks
        self.random_state = random_state

    def split(self, data: List[Dict]) -> Dict[str, List[Dict]]:
        """
        执行 2×2 划分

        Args:
            data: 预处理后的数据列表，每个元素包含：
                - subject_id: 被试ID
                - label: 标签（注意力分数）
                - tasks: 任务列表，每个任务包含 task_id 和 segments

        Returns:
            划分后的数据字典：
            {
                'train': [...],  # 前K人 × 前N题
                'test1': [...],  # 后L人 × 前N题
                'test2': [...],  # 前K人 × 后M题
                'test3': [...],  # 后L人 × 后M题
            }
        """
        if len(data) == 0:
            raise ValueError("数据为空")

        # 1. 收集所有被试ID并随机打乱
        subject_ids = [d['subject_id'] for d in data]
        np.random.seed(self.random_state)
        shuffled_indices = np.random.permutation(len(subject_ids))

        # 划分被试
        train_subject_indices = shuffled_indices[:self.train_subjects]
        test_subject_indices = shuffled_indices[self.train_subjects:]

        train_subject_ids = set(subject_ids[i] for i in train_subject_indices)
        test_subject_ids = set(subject_ids[i] for i in test_subject_indices)

        # 2. 收集所有任务ID并排序
        all_task_ids = set()
        for d in data:
            for task in d['tasks']:
                all_task_ids.add(task['task_id'])

        sorted_task_ids = sorted(all_task_ids)
        train_task_ids = set(sorted_task_ids[:self.train_tasks])
        test_task_ids = set(sorted_task_ids[self.train_tasks:])

        logger.info(f"被试划分: 训练={len(train_subject_ids)}, 测试={len(test_subject_ids)}")
        logger.info(f"任务划分: 训练={len(train_task_ids)}, 测试={len(test_task_ids)}")

        # 3. 按照 2×2 矩阵划分数据
        result = {
            'train': [],   # 前K人 × 前N题
            'test1': [],   # 后L人 × 前N题 (新人旧题)
            'test2': [],   # 前K人 × 后M题 (旧人新题)
            'test3': [],   # 后L人 × 后M题 (新人新题)
        }

        for d in data:
            subject_id = d['subject_id']
            is_train_subject = subject_id in train_subject_ids

            # 分离任务
            train_tasks = []
            test_tasks = []
            for task in d['tasks']:
                if task['task_id'] in train_task_ids:
                    train_tasks.append(task)
                else:
                    test_tasks.append(task)

            # 创建包含训练任务的数据副本
            if train_tasks:
                train_data = {
                    'subject_id': subject_id,
                    'label': d['label'],
                    'category': d.get('category', 2),  # 分类标签
                    'tasks': train_tasks,
                }
                if is_train_subject:
                    result['train'].append(train_data)
                else:
                    result['test1'].append(train_data)

            # 创建包含测试任务的数据副本
            if test_tasks:
                test_data = {
                    'subject_id': subject_id,
                    'label': d['label'],
                    'category': d.get('category', 2),  # 分类标签
                    'tasks': test_tasks,
                }
                if is_train_subject:
                    result['test2'].append(test_data)
                else:
                    result['test3'].append(test_data)

        # 记录划分统计
        for split_name, split_data in result.items():
            n_subjects = len(split_data)
            n_tasks = sum(len(d['tasks']) for d in split_data) if split_data else 0
            n_segments = sum(
                sum(len(t['segments']) for t in d['tasks'])
                for d in split_data
            ) if split_data else 0
            logger.info(f"{split_name}: {n_subjects}个被试, {n_tasks}个任务, {n_segments}个片段")

        return result

    def get_split_info(self, data: List[Dict]) -> Dict[str, SplitInfo]:
        """
        获取划分信息（不实际划分数据）

        Args:
            data: 预处理后的数据列表

        Returns:
            各划分的信息字典
        """
        # 执行划分
        splits = self.split(data)

        info = {}
        descriptions = {
            'train': '训练集（前K人×前N题）',
            'test1': '测试集1（新人旧题，跨被试泛化）',
            'test2': '测试集2（旧人新题，跨任务泛化）',
            'test3': '测试集3（新人新题，双重泛化）',
        }

        for name, split_data in splits.items():
            subjects = [d['subject_id'] for d in split_data]
            task_ids = set()
            for d in split_data:
                for task in d['tasks']:
                    task_ids.add(task['task_id'])

            info[name] = SplitInfo(
                name=name,
                description=descriptions[name],
                subjects=subjects,
                task_ids=task_ids,
            )

        return info

    def get_split_summary(self, splits: Dict[str, List[Dict]] = None) -> Dict[str, Dict]:
        """
        获取划分摘要

        Args:
            splits: split() 方法返回的划分数据字典，如果为 None 则返回配置信息

        Returns:
            各划分的摘要字典：
            {
                'train': {'samples': N, 'subjects': K, 'tasks': M, 'description': '...'},
                'test1': {...},
                ...
            }
        """
        descriptions = {
            'train': '训练集（前K人×前N题）',
            'test1': '新人旧题（跨被试泛化）',
            'test2': '旧人新题（跨任务泛化）',
            'test3': '新人新题（双重泛化）',
        }

        if splits is None:
            # 返回配置信息（无实际数据时）
            return {
                'train': {
                    'samples': 0,
                    'subjects': self.train_subjects,
                    'tasks': self.train_tasks,
                    'description': descriptions['train'],
                },
                'test1': {
                    'samples': 0,
                    'subjects': 0,
                    'tasks': self.train_tasks,
                    'description': descriptions['test1'],
                },
                'test2': {
                    'samples': 0,
                    'subjects': self.train_subjects,
                    'tasks': 0,
                    'description': descriptions['test2'],
                },
                'test3': {
                    'samples': 0,
                    'subjects': 0,
                    'tasks': 0,
                    'description': descriptions['test3'],
                },
                'config': {
                    'train_subjects': self.train_subjects,
                    'train_tasks': self.train_tasks,
                    'random_state': self.random_state,
                }
            }

        # 根据实际划分数据计算摘要
        summary = {}
        for split_name in ['train', 'test1', 'test2', 'test3']:
            split_data = splits.get(split_name, [])
            n_samples = len(split_data)
            n_subjects = len(set(d['subject_id'] for d in split_data)) if split_data else 0

            task_ids = set()
            for d in split_data:
                for task in d['tasks']:
                    task_ids.add(task['task_id'])
            n_tasks = len(task_ids)

            summary[split_name] = {
                'samples': n_samples,
                'subjects': n_subjects,
                'tasks': n_tasks,
                'description': descriptions.get(split_name, ''),
            }

        # 添加配置信息
        summary['config'] = {
            'train_subjects': self.train_subjects,
            'train_tasks': self.train_tasks,
            'random_state': self.random_state,
        }

        return summary
