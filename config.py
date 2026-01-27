# -*- coding: utf-8 -*-
"""
配置管理模块

包含项目所有可配置参数
"""

from pathlib import Path

# ============ 路径配置 ============
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data' / 'gaze_trajectory_data'
OUTPUT_DIR = BASE_DIR / 'outputs'

# 数据文件路径
LABELS_FILE = DATA_DIR / '个人标签.xlsx'
TASKS_FILE = DATA_DIR / '题目信息.xlsx'

# 输出目录
FEATURES_DIR = OUTPUT_DIR / 'features'
MODELS_DIR = OUTPUT_DIR / 'models'
FIGURES_DIR = OUTPUT_DIR / 'figures'

# ============ 屏幕参数 ============
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

# ============ 特征提取参数 ============
# 熵计算网格大小
ENTROPY_GRID_SIZE = 8

# 目标区域容差（像素）
TARGET_TOLERANCE = 50.0

# ============ 模型参数 ============
# 交叉验证折数
CV_SPLITS = 5

# ============ 深度学习配置 ============
DEEP_LEARNING_CONFIG = {
    # 数据参数
    'max_seq_len': 100,      # 每个片段最大眼动点数
    'max_tasks': 30,         # 每个被试最大任务数
    'max_segments': 30,      # 每个任务最大片段数

    # Transformer参数
    'input_dim': 7,          # (x, y, dt, velocity, acceleration, direction, direction_change)
    'segment_d_model': 64,   # 片段编码器维度
    'segment_nhead': 4,      # 片段编码器注意力头数
    'segment_num_layers': 4, # 片段编码器层数
    'task_d_model': 128,     # 任务编码器维度
    'task_nhead': 4,         # 任务编码器注意力头数
    'task_num_layers': 2,    # 任务编码器层数
    'attention_dim': 32,     # 聚合注意力维度
    'dropout': 0.1,

    # 训练参数
    'batch_size': 8,
    'learning_rate': 1e-4,   # Transformer通常用较小学习率
    'weight_decay': 1e-4,
    'warmup_epochs': 5,      # 学习率预热
    'epochs': 100,
    'patience': 15,
    'grad_clip': 1.0,

    # 交叉验证
    'n_splits': 5,
    'random_state': 42,
}

# ============ 任务配置映射 ============
# 题目编号到任务配置的默认映射（如果无法从文件读取）
DEFAULT_GRID_SIZE = 25  # 5x5 方格
