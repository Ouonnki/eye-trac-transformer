# -*- coding: utf-8 -*-
"""
实验配置模块

提供统一的实验配置管理，支持 2×2 划分和 K-Fold 交叉验证。
支持分类/回归双模式和 CADT 域迁移。
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

    支持两种任务类型：
    - classification: 分类模式（支持 CADT 域迁移）
    - regression: 回归模式
    """

    # 实验模式
    mode: str = 'quick'  # 'quick' 快速验证, 'formal' 正式实验

    # 任务类型
    task_type: str = 'classification'  # 'classification' 或 'regression'
    num_classes: int = 3  # 分类模式下的类别数（0=低, 1=中, 2=高）

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

    # 域迁移配置（仅分类模式有效）
    enable_domain_adaptation: bool = True  # 是否启用域迁移
    target_domain: str = 'test2'  # 目标域：'test1', 'test2', 'test3', 'all'
    pretrain_epochs: int = 20  # 预训练阶段 epoch 数
    center_alignment_weight: float = 10.0  # 中心对齐损失权重 (λ1)
    domain_adversarial_weight: float = 1.0  # 域对抗损失权重 (λ2)
    center_diversity_weight: float = 0.1  # 中心多样性损失权重 (λ3)

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
        - TASK_TYPE: 'classification' 或 'regression'
        - NUM_CLASSES: 分类模式下的类别数
        - SPLIT_TYPE: '2x2' 或 'kfold'
        - TRAIN_SUBJECTS: 训练集被试数量
        - TRAIN_TASKS: 训练集任务数量
        - N_FOLDS: K-Fold 折数
        - N_REPEATS: 重复次数
        - RANDOM_SEED: 随机种子
        - ENABLE_DOMAIN_ADAPTATION: 'true' 或 'false'
        - TARGET_DOMAIN: 'test1', 'test2', 'test3', 'all'
        - PRETRAIN_EPOCHS: 预训练阶段 epoch 数
        - CENTER_ALIGNMENT_WEIGHT: 中心对齐损失权重
        - DOMAIN_ADVERSARIAL_WEIGHT: 域对抗损失权重
        - CENTER_DIVERSITY_WEIGHT: 中心多样性损失权重
        - SAVE_FIGURES: 'true' 或 'false'
        - OUTPUT_DIR: 输出目录
        - EXPERIMENT_NAME: 实验名称

        Returns:
            ExperimentConfig 实例
        """
        return cls(
            mode=os.environ.get('EXPERIMENT_MODE', 'quick'),
            task_type=os.environ.get('TASK_TYPE', 'classification'),
            num_classes=int(os.environ.get('NUM_CLASSES', '3')),
            split_type=os.environ.get('SPLIT_TYPE', '2x2'),
            train_subjects=int(os.environ.get('TRAIN_SUBJECTS', '100')),
            train_tasks=int(os.environ.get('TRAIN_TASKS', '20')),
            n_folds=int(os.environ.get('N_FOLDS', '5')),
            n_repeats=int(os.environ.get('N_REPEATS', '1')),
            random_seed=int(os.environ.get('RANDOM_SEED', '42')),
            enable_domain_adaptation=os.environ.get(
                'ENABLE_DOMAIN_ADAPTATION', 'true'
            ).lower() == 'true',
            target_domain=os.environ.get('TARGET_DOMAIN', 'test2'),
            pretrain_epochs=int(os.environ.get('PRETRAIN_EPOCHS', '20')),
            center_alignment_weight=float(os.environ.get(
                'CENTER_ALIGNMENT_WEIGHT', '10.0'
            )),
            domain_adversarial_weight=float(os.environ.get(
                'DOMAIN_ADVERSARIAL_WEIGHT', '1.0'
            )),
            center_diversity_weight=float(os.environ.get(
                'CENTER_DIVERSITY_WEIGHT', '0.1'
            )),
            save_figures=os.environ.get('SAVE_FIGURES', 'true').lower() == 'true',
            output_dir=os.environ.get('OUTPUT_DIR', 'outputs/dl_models'),
            experiment_name=os.environ.get('EXPERIMENT_NAME', ''),
        )

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'mode': self.mode,
            'task_type': self.task_type,
            'num_classes': self.num_classes,
            'split_type': self.split_type,
            'train_subjects': self.train_subjects,
            'train_tasks': self.train_tasks,
            'n_folds': self.n_folds,
            'n_repeats': self.n_repeats,
            'random_seed': self.random_seed,
            'enable_domain_adaptation': self.enable_domain_adaptation,
            'target_domain': self.target_domain,
            'pretrain_epochs': self.pretrain_epochs,
            'center_alignment_weight': self.center_alignment_weight,
            'domain_adversarial_weight': self.domain_adversarial_weight,
            'center_diversity_weight': self.center_diversity_weight,
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
        print(f'  任务类型: {self.task_type}')

        if self.task_type == 'classification':
            print(f'  类别数: {self.num_classes}')

        print(f'  划分策略: {self.split_type}')

        if self.split_type == '2x2':
            print(f'  训练被试数: {self.train_subjects}')
            print(f'  训练任务数: {self.train_tasks}')
        else:
            print(f'  K-Fold 折数: {self.n_folds}')

        if self.mode == 'formal':
            print(f'  重复次数: {self.n_repeats}')

        print(f'  随机种子: {self.random_seed}')

        # 域迁移配置（仅分类模式）
        if self.task_type == 'classification':
            print('-' * 60)
            print('  域迁移配置:')
            print(f'    启用域迁移: {self.enable_domain_adaptation}')
            if self.enable_domain_adaptation:
                print(f'    目标域: {self.target_domain}')
                print(f'    预训练 epochs: {self.pretrain_epochs}')
                print(f'    中心对齐权重 (λ1): {self.center_alignment_weight}')
                print(f'    域对抗权重 (λ2): {self.domain_adversarial_weight}')
                print(f'    中心多样性权重 (λ3): {self.center_diversity_weight}')

        print('-' * 60)
        print(f'  保存图表: {self.save_figures}')
        print(f'  输出目录: {self.output_dir}')
        print(f'  实验名称: {self.experiment_name}')
        print('=' * 60)
