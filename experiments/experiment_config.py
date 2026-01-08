# -*- coding: utf-8 -*-
"""
实验配置模块

提供统一的实验配置管理，支持 2×2 划分和 K-Fold 交叉验证。
通过环境变量配置，无需修改代码。
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class ExperimentConfig:
    """
    实验配置

    支持两种实验模式：
    - quick: 快速验证，单次训练
    - formal: 正式实验，多次随机划分取平均

    支持两种划分策略：
    - 2x2: 2×2 矩阵划分（被试 × 任务）
    - kfold: K-Fold 交叉验证
    """

    # 实验模式
    mode: str = 'quick'  # 'quick' 快速验证, 'formal' 正式实验

    # 划分策略
    split_type: str = '2x2'  # '2x2' 或 'kfold'

    # 2×2 划分配置
    train_subjects: int = 100  # 训练集被试数量
    train_tasks: int = 20      # 训练集任务数量 (按 task_id 排序后取前N个)

    # K-Fold 配置
    n_folds: int = 5  # K-Fold 折数

    # 正式实验配置
    n_repeats: int = 1  # 重复次数，quick=1, formal=5-10
    random_seed: int = 42  # 随机种子

    # 图表配置
    save_figures: bool = True  # 是否保存训练曲线图
    figure_dpi: int = 150  # 图表分辨率

    # 输出配置
    output_dir: str = 'outputs/dl_models'  # 输出目录
    experiment_name: str = ''  # 实验名称，空则自动生成时间戳

    def __post_init__(self):
        """初始化后处理"""
        # 自动生成实验名称
        if not self.experiment_name:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.experiment_name = f'exp_{timestamp}'

        # 正式实验模式自动设置重复次数
        if self.mode == 'formal' and self.n_repeats == 1:
            self.n_repeats = 5

    @classmethod
    def from_env(cls) -> 'ExperimentConfig':
        """
        从环境变量创建配置

        支持的环境变量：
        - EXPERIMENT_MODE: 'quick' 或 'formal'
        - SPLIT_TYPE: '2x2' 或 'kfold'
        - TRAIN_SUBJECTS: 训练集被试数量
        - TRAIN_TASKS: 训练集任务数量
        - N_FOLDS: K-Fold 折数
        - N_REPEATS: 重复次数
        - RANDOM_SEED: 随机种子
        - SAVE_FIGURES: 'true' 或 'false'
        - OUTPUT_DIR: 输出目录
        - EXPERIMENT_NAME: 实验名称

        Returns:
            ExperimentConfig 实例
        """
        return cls(
            mode=os.environ.get('EXPERIMENT_MODE', 'quick'),
            split_type=os.environ.get('SPLIT_TYPE', '2x2'),
            train_subjects=int(os.environ.get('TRAIN_SUBJECTS', '100')),
            train_tasks=int(os.environ.get('TRAIN_TASKS', '20')),
            n_folds=int(os.environ.get('N_FOLDS', '5')),
            n_repeats=int(os.environ.get('N_REPEATS', '1')),
            random_seed=int(os.environ.get('RANDOM_SEED', '42')),
            save_figures=os.environ.get('SAVE_FIGURES', 'true').lower() == 'true',
            output_dir=os.environ.get('OUTPUT_DIR', 'outputs/dl_models'),
            experiment_name=os.environ.get('EXPERIMENT_NAME', ''),
        )

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'mode': self.mode,
            'split_type': self.split_type,
            'train_subjects': self.train_subjects,
            'train_tasks': self.train_tasks,
            'n_folds': self.n_folds,
            'n_repeats': self.n_repeats,
            'random_seed': self.random_seed,
            'save_figures': self.save_figures,
            'output_dir': self.output_dir,
            'experiment_name': self.experiment_name,
        }

    def print_config(self) -> None:
        """打印配置信息"""
        print('\n' + '=' * 60)
        print('实验配置')
        print('=' * 60)
        print(f'  模式: {self.mode}')
        print(f'  划分策略: {self.split_type}')

        if self.split_type == '2x2':
            print(f'  训练被试数: {self.train_subjects}')
            print(f'  训练任务数: {self.train_tasks}')
        else:
            print(f'  K-Fold 折数: {self.n_folds}')

        if self.mode == 'formal':
            print(f'  重复次数: {self.n_repeats}')

        print(f'  随机种子: {self.random_seed}')
        print(f'  保存图表: {self.save_figures}')
        print(f'  输出目录: {self.output_dir}')
        print(f'  实验名称: {self.experiment_name}')
        print('=' * 60)
