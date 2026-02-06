# -*- coding: utf-8 -*-
"""
统一配置管理模块

提供从 JSON 文件加载配置的功能，并支持配置快照保存。
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch


@dataclass
class ExperimentConfig:
    """实验配置"""
    train_subjects: int = 100
    train_tasks: int = 20
    n_repeats: int = 1
    random_seed: int = 42
    experiment_name: str = ''
    output_dir: str = 'outputs/dl_models'


@dataclass
class SequenceConfigData:
    """序列配置（数据相关参数，原 SequenceConfig 的可配置版本）"""
    max_seq_len: int = 100      # 每个片段最大眼动点数
    max_tasks: int = 30         # 每个被试最大任务数
    max_segments: int = 30      # 每个任务最大片段数
    screen_width: int = 1920    # 屏幕宽度
    screen_height: int = 1080   # 屏幕高度
    input_dim: int = 7          # 输入特征维度（7基础眼动特征）


@dataclass
class ModelConfig:
    """模型架构配置"""
    segment_d_model: int = 128  # 64 → 128 (序列+任务 concat 后的维度)
    segment_nhead: int = 8     # 4 → 8 (d_model/8)
    segment_num_layers: int = 4
    task_d_model: int = 256     # 128 → 256 (segment_d_model * 2)
    task_nhead: int = 8         # 4 → 8
    task_num_layers: int = 2
    attention_dim: int = 64     # 32 → 64
    dropout: float = 0.1

    # 任务嵌入配置
    # 是否使用任务嵌入（放在任务编码器前）
    use_task_embedding: bool = False
    # 任务嵌入维度（所有五个条件使用统一的嵌入维度）
    task_embedding_dim: int = 2


@dataclass
class TrainingConfig:
    """训练超参数配置"""
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    epochs: int = 400
    patience: int = 100
    grad_clip: float = 1.0
    label_smoothing: float = 0.1


@dataclass
class TaskConfig:
    """任务类型配置"""
    type: Literal['classification', 'regression'] = 'classification'
    num_classes: int = 3
    use_class_weights: bool = True


@dataclass
class DeviceConfig:
    """计算设备配置"""
    device: str = 'cuda'  # 自动检测
    use_multi_gpu: bool = True
    use_amp: bool = True
    use_gradient_checkpointing: bool = False
    num_workers: int = 8
    pin_memory: bool = True

    def __post_init__(self):
        """初始化后自动检测设备"""
        if self.device == 'cuda' and not torch.cuda.is_available():
            self.device = 'cpu'


@dataclass
class OutputConfig:
    """输出配置"""
    save_best: bool = True
    save_figures: bool = True
    figure_dpi: int = 150
    summary_interval: int = 10


@dataclass
class CADTConfig:
    """CADT域适应配置（仅CADT模型使用）"""
    target_domain: Literal['test1', 'test2', 'test3'] = 'test1'
    cadt_kl_weight: float = 1.0
    cadt_dis_weight: float = 1.0
    pre_train_epochs: int = 120
    pre_train_dis_weight: float = 0.1
    reset_mode: Literal['none', 'optimizer', 'full'] = 'optimizer'
    use_augmentation: bool = True
    encoder_lr: float = 1e-4
    classifier_lr: float = 1e-3
    discriminator_lr: float = 1e-3


@dataclass
class UnifiedConfig:
    """统一配置根节点"""
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    sequence: SequenceConfigData = field(default_factory=SequenceConfigData)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    cadt: CADTConfig = field(default_factory=CADTConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_json(cls, path: str) -> 'UnifiedConfig':
        """从 JSON 文件加载配置"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> 'UnifiedConfig':
        """从字典创建配置"""
        return cls(
            experiment=ExperimentConfig(**data.get('experiment', {})),
            sequence=SequenceConfigData(**data.get('sequence', {})),
            model=ModelConfig(**data.get('model', {})),
            training=TrainingConfig(**data.get('training', {})),
            task=TaskConfig(**data.get('task', {})),
            cadt=CADTConfig(**data.get('cadt', {})),
            device=DeviceConfig(**data.get('device', {})),
            output=OutputConfig(**data.get('output', {})),
        )

    def to_json(self, path: str) -> None:
        """保存配置到 JSON 文件"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def to_dict(self) -> dict:
        """转换为字典"""
        from dataclasses import asdict
        return asdict(self)

    def save_snapshot(self, output_dir: str) -> str:
        """保存配置快照到输出目录"""
        output_path = Path(output_dir) / 'config.json'
        self.to_json(str(output_path))
        return str(output_path)

    def to_seq_config(self) -> 'SequenceConfig':
        """转换为 SequenceConfig（用于数据集和模型）"""
        from src.models.dl_dataset import SequenceConfig
        return SequenceConfig(
            max_seq_len=self.sequence.max_seq_len,
            max_tasks=self.sequence.max_tasks,
            max_segments=self.sequence.max_segments,
            screen_width=self.sequence.screen_width,
            screen_height=self.sequence.screen_height,
            input_dim=self.sequence.input_dim,
        )
