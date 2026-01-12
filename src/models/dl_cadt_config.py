# -*- coding: utf-8 -*-
"""
CADT 域适应训练配置模块

提供 CADT-Transformer 域适应训练的配置类。
"""

import os
from dataclasses import dataclass, field
from typing import Literal

import torch

from src.models.dl_trainer import TrainingConfig


@dataclass
class CADTConfig(TrainingConfig):
    """
    CADT 域适应训练配置

    继承自 TrainingConfig，添加 CADT 特有的配置项。
    """

    # 目标域选择
    target_domain: Literal['test1', 'test2', 'test3'] = 'test1'

    # CADT 超参数
    cadt_kl_weight: float = 1.0       # 原型聚类损失权重 (kl_loss)
    cadt_dis_weight: float = 1.0      # 域对抗损失权重 (dis_loss)
    pre_train_epochs: int = 20        # 预训练阶段 epoch 数

    # 数据增强
    use_augmentation: bool = True     # 是否使用数据增强

    # 学习率配置（分离优化器）
    encoder_lr: float = 1e-4          # 编码器学习率
    classifier_lr: float = 1e-3       # 分类器学习率
    discriminator_lr: float = 1e-3    # 辨别器学习率

    # 输出目录（覆盖父类默认值）
    output_dir: str = 'outputs/cadt_res'

    def __post_init__(self):
        """初始化后处理"""
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)

    @classmethod
    def from_env(cls) -> 'CADTConfig':
        """
        从环境变量创建配置

        支持的环境变量：
        - TARGET_DOMAIN: 目标域选择 (test1/test2/test3)
        - KL_WEIGHT: 原型聚类损失权重
        - DIS_WEIGHT: 域对抗损失权重
        - PRE_TRAIN_EPOCHS: 预训练 epoch 数
        - MODEL_MODE: 模型模式 (full/light)
        - TASK_TYPE: 任务类型 (classification/regression)

        Returns:
            CADTConfig 实例
        """
        # 基础配置
        model_mode = os.environ.get('MODEL_MODE', 'light')

        if model_mode == 'full':
            # 完整模型配置（适合 24GB 显存，如 RTX 3090）
            # CADT 需要同时处理源域和目标域，显存消耗约为普通训练的2倍
            config = cls(
                input_dim=7,
                segment_d_model=128,
                segment_nhead=8,
                segment_num_layers=6,
                task_d_model=256,
                task_nhead=8,
                task_num_layers=4,
                attention_dim=64,
                dropout=0.1,
                max_seq_len=100,
                max_tasks=30,
                max_segments=30,
                batch_size=8,  # 24GB 显存可用较大批次
                use_amp=True,  # 混合精度训练
                use_gradient_checkpointing=False,  # 24GB 不需要梯度检查点
            )
        elif model_mode == 'medium':
            # 中等配置（适合24GB显存）
            config = cls(
                input_dim=7,
                segment_d_model=64,
                segment_nhead=4,
                segment_num_layers=4,
                task_d_model=128,
                task_nhead=4,
                task_num_layers=2,
                attention_dim=32,
                dropout=0.1,
                max_seq_len=100,
                max_tasks=30,
                max_segments=30,
                batch_size=4,
                use_gradient_checkpointing=True,
            )
        else:
            # 轻量级配置（适合8-16GB显存）
            config = cls(
                input_dim=7,
                segment_d_model=32,
                segment_nhead=4,
                segment_num_layers=2,
                task_d_model=64,
                task_nhead=4,
                task_num_layers=2,
                attention_dim=16,
                dropout=0.1,
                max_seq_len=50,
                max_tasks=20,
                max_segments=20,
                batch_size=2,
                use_gradient_checkpointing=True,
            )

        # CADT 特有配置（优先使用环境变量，否则使用类定义的默认值）
        config.target_domain = os.environ.get('TARGET_DOMAIN', config.target_domain)
        if 'KL_WEIGHT' in os.environ:
            config.cadt_kl_weight = float(os.environ['KL_WEIGHT'])
        if 'DIS_WEIGHT' in os.environ:
            config.cadt_dis_weight = float(os.environ['DIS_WEIGHT'])
        if 'PRE_TRAIN_EPOCHS' in os.environ:
            config.pre_train_epochs = int(os.environ['PRE_TRAIN_EPOCHS'])

        # 任务类型
        if 'TASK_TYPE' in os.environ:
            config.task_type = os.environ['TASK_TYPE']

        return config

    def get_feature_dim(self) -> int:
        """获取特征维度（即 task_d_model）"""
        return self.task_d_model

    def __str__(self) -> str:
        """返回配置的字符串表示"""
        return (
            f"CADTConfig(\n"
            f"  target_domain={self.target_domain},\n"
            f"  task_d_model={self.task_d_model},\n"
            f"  cadt_kl_weight={self.cadt_kl_weight},\n"
            f"  cadt_dis_weight={self.cadt_dis_weight},\n"
            f"  pre_train_epochs={self.pre_train_epochs},\n"
            f"  epochs={self.epochs},\n"
            f"  batch_size={self.batch_size},\n"
            f"  task_type={self.task_type},\n"
            f"  num_classes={self.num_classes},\n"
            f")"
        )
