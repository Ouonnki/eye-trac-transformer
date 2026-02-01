# -*- coding: utf-8 -*-
"""
集成预测器模块

结合层级模型和片段模型的预测，实现条件感知的动态加权集成。

策略说明：
- 层级模型在已知任务条件（grid_size ≤ 25）上可用
- 片段模型对任务条件更鲁棒，但在新被试上表现较弱
- 通过条件感知的动态加权，充分利用两个模型的优势
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.config import UnifiedConfig
from src.models.dl_models import HierarchicalTransformerNetwork
from src.models.segment_model import SegmentEncoder
from src.models.dl_dataset import (
    SequenceConfig, SegmentGazeDataset, segment_collate_fn, collate_fn
)

logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    """集成模型配置"""
    # 模型权重路径
    hierarchical_weight_path: str = ""
    segment_weight_path: str = ""

    # 集成策略
    strategy: str = "condition_aware"  # "fixed", "condition_aware", "dual_aware", "learned"

    # 固定权重模式参数
    hierarchical_weight: float = 0.5
    segment_weight: float = 0.5

    # 条件感知模式参数
    known_task_hier_weight: float = 0.7  # 已知任务时层级模型权重
    new_task_hier_weight: float = 0.1    # 新任务时层级模型权重
    max_known_grid_size: int = 25        # 训练时见过的最大 grid_size
    max_known_distractor: int = 8        # 训练时见过的最大 distractor_count

    # 双重感知模式参数（同时考虑任务和被试）
    new_subject_new_task_hier_weight: float = 0.0  # 新被试+新任务时层级权重（纯片段模型）
    known_subject_new_task_hier_weight: float = 0.1  # 已知被试+新任务时层级权重

    # 学习融合模式参数
    fusion_hidden_dim: int = 64  # 融合网络隐藏层维度
    fusion_dropout: float = 0.1     # 融合网络dropout

    # 推理参数
    batch_size: int = 8
    device: str = "cuda"


class LearnedFusion(nn.Module):
    """
    可学习的融合网络

    输入：两个模型的概率输出和条件信息
    输出：最终预测
    """

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        use_task_conditions: bool = True,
        use_subject_info: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_task_conditions = use_task_conditions
        self.use_subject_info = use_subject_info

        # 计算输入维度
        input_dim = 2 * num_classes  # hier_probs + seg_probs
        if use_task_conditions:
            input_dim += 5  # task_conditions (grid_scale, continuous_thinking, click_disappear, has_distractor, has_task_distractor)
        if use_subject_info:
            input_dim += 1  # is_known_subject

        # 融合网络
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = nn.Dropout(dropout if dropout > 0 else 0)
        self.activation = nn.ReLU()

    def forward(
        self,
        hier_probs: torch.Tensor,  # (batch, num_classes)
        seg_probs: torch.Tensor,   # (batch, num_classes)
        task_conditions: Optional[torch.Tensor] = None,  # (batch, 5) or (batch, max_tasks, 5)
        is_known_subject: Optional[torch.Tensor] = None,  # (batch,)
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            hier_probs: 层级模型概率
            seg_probs: 片段模型概率
            task_conditions: 任务条件
            is_known_subject: 被试是否已知

        Returns:
            (batch, num_classes) 融合后的logits
        """
        features = [hier_probs, seg_probs]

        # 处理任务条件
        if task_conditions is not None:
            if task_conditions.dim() == 3:
                # (batch, max_tasks, 5) -> 转换为浮点数后取平均
                task_cond = task_conditions.float().mean(dim=1)  # (batch, 5)
            else:
                task_cond = task_conditions.float()  # (batch, 5)
            # 归一化到 [0, 1]
            task_cond = task_cond / torch.tensor([4., 1., 1., 1., 1.], device=task_cond.device)
            features.append(task_cond)

        # 处理被试信息
        if is_known_subject is not None:
            if isinstance(is_known_subject, np.ndarray):
                is_known_subject = torch.from_numpy(is_known_subject).float()
            features.append(is_known_subject.unsqueeze(1))

        # 拼接特征
        x = torch.cat(features, dim=-1)  # (batch, input_dim)

        # 通过MLP
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        logits = self.fc3(x)

        return logits


class EnsemblePredictor:
    """
    集成预测器

    结合层级模型和片段模型的预测结果。

    支持五种集成策略：
    1. fixed: 固定权重加权平均
    2. condition_aware: 根据任务条件动态调整权重
    3. dual_aware: 同时考虑任务条件和被试条件
    4. learned: 使用神经网络学习融合
    """

    def __init__(
        self,
        config: UnifiedConfig,
        seq_config: SequenceConfig,
        ensemble_config: EnsembleConfig,
    ):
        """
        初始化集成预测器

        Args:
            config: 统一配置
            seq_config: 序列配置
            ensemble_config: 集成配置
        """
        self.config = config
        self.seq_config = seq_config
        self.ensemble_config = ensemble_config
        self.device = torch.device(ensemble_config.device)

        # 确定任务类型和类别数
        self.num_classes = config.task.num_classes
        self.task_type = config.task.type

        # 初始化模型
        self.hierarchical_model = None
        self.segment_model = None
        self.fusion_network = None

        logger.info(f"初始化集成预测器，策略: {ensemble_config.strategy}")

    def load_models(
        self,
        hierarchical_weight_path: Optional[str] = None,
        segment_weight_path: Optional[str] = None,
    ) -> None:
        """
        加载两个模型的权重

        Args:
            hierarchical_weight_path: 层级模型权重路径
            segment_weight_path: 片段模型权重路径
        """
        hier_path = hierarchical_weight_path or self.ensemble_config.hierarchical_weight_path
        seg_path = segment_weight_path or self.ensemble_config.segment_weight_path

        # 加载层级模型
        logger.info(f"加载层级模型: {hier_path}")
        self.hierarchical_model = HierarchicalTransformerNetwork.from_config(
            self.config, self.seq_config, num_classes=self.num_classes
        )
        checkpoint = torch.load(hier_path, map_location=self.device, weights_only=False)
        # 检查是否是完整检查点（包含 model_state_dict）还是纯 state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        # 处理 DataParallel 保存的权重
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.hierarchical_model.load_state_dict(state_dict)
        self.hierarchical_model.to(self.device)
        self.hierarchical_model.eval()

        # 加载片段模型
        logger.info(f"加载片段模型: {seg_path}")
        self.segment_model = SegmentEncoder.from_config(
            self.config, self.seq_config, num_classes=self.num_classes
        )
        checkpoint = torch.load(seg_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.segment_model.load_state_dict(state_dict)
        self.segment_model.to(self.device)
        self.segment_model.eval()

        # 初始化融合网络（如果使用learned策略）
        if self.ensemble_config.strategy == "learned":
            logger.info("初始化学习融合网络...")
            self.fusion_network = LearnedFusion(
                num_classes=self.num_classes,
                hidden_dim=self.ensemble_config.fusion_hidden_dim,
                dropout=self.ensemble_config.fusion_dropout,
                use_task_conditions=True,
                use_subject_info=True,
            ).to(self.device)
            # 融合网络默认使用简单权重初始化，最后偏向片段模型
            # 因为片段模型泛化更好
            with torch.no_grad():
                self.fusion_network.fc3.weight.data.fill_(0.01)
                self.fusion_network.fc3.bias.data[:] = torch.tensor([0.2, 0.3, 0.5], device=self.device)

        logger.info("两个模型加载完成")

    def is_known_task_condition(self, task_conditions: torch.Tensor) -> torch.Tensor:
        """
        判断任务条件是否在训练分布内

        Args:
            task_conditions: (batch, max_tasks, 5) 或 (batch, 5) 任务条件张量
                列顺序: grid_scale, continuous_thinking, click_disappear,
                       has_distractor, has_task_distractor

        Returns:
            (batch,) 布尔张量，True 表示已知任务条件
        """
        # grid_scale 是第一列，值为 1-4 对应 grid_size 9, 16, 25, 36
        # grid_scale <= 3 表示 grid_size <= 25
        max_known_grid_scale = 3  # 对应 grid_size=25

        if task_conditions.dim() == 3:
            # (batch, max_tasks, 5) -> 取所有任务的最大 grid_scale
            grid_scales = task_conditions[:, :, 0]  # (batch, max_tasks)
            max_grid_scale = grid_scales.max(dim=1)[0]  # (batch,)
        else:
            # (batch, 5)
            max_grid_scale = task_conditions[:, 0]  # (batch,)

        return max_grid_scale <= max_known_grid_scale

    def get_dynamic_weights(
        self,
        task_conditions: Optional[torch.Tensor],
        subject_ids: Optional[List[str]],
        batch_size: int,
        known_subjects: Optional[set] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根据任务条件和被试条件计算动态权重

        Args:
            task_conditions: 任务条件张量
            subject_ids: 被试ID列表
            batch_size: 批次大小
            known_subjects: 已知被试ID集合

        Returns:
            (hier_weights, seg_weights): 两个 (batch, 1) 权重张量
        """
        strategy = self.ensemble_config.strategy

        if strategy == "fixed":
            # 固定权重
            hier_w = torch.full((batch_size, 1), self.ensemble_config.hierarchical_weight,
                               device=self.device)
            seg_w = torch.full((batch_size, 1), self.ensemble_config.segment_weight,
                              device=self.device)

        elif strategy == "condition_aware":
            # 条件感知权重（只考虑任务）
            if task_conditions is None:
                # 没有任务条件信息，使用新任务的保守权重
                hier_w = torch.full((batch_size, 1), self.ensemble_config.new_task_hier_weight,
                                   device=self.device)
            else:
                is_known = self.is_known_task_condition(task_conditions)  # (batch,)
                hier_w = torch.where(
                    is_known.unsqueeze(1),
                    torch.full((batch_size, 1), self.ensemble_config.known_task_hier_weight,
                              device=self.device),
                    torch.full((batch_size, 1), self.ensemble_config.new_task_hier_weight,
                              device=self.device),
                )
            seg_w = 1.0 - hier_w

        elif strategy == "dual_aware":
            # 双重感知权重（同时考虑任务和被试）
            if task_conditions is None or subject_ids is None or known_subjects is None:
                # 信息不足，使用默认权重
                hier_w = torch.full((batch_size, 1), self.ensemble_config.new_task_hier_weight,
                                   device=self.device)
            else:
                # 判断任务是否已知
                is_known_task = self.is_known_task_condition(task_conditions)  # (batch,)

                # 判断被试是否已知
                is_known_subject = torch.tensor([
                    sid in known_subjects for sid in subject_ids
                ], dtype=torch.bool, device=self.device)  # (batch,)

                # 四种情况：
                # 1. 已知被试 + 已知任务: 高层级权重
                # 2. 已知被试 + 新任务: 低层级权重
                # 3. 新被试 + 已知任务: 中等层级权重（层级模型擅长已知任务）
                # 4. 新被试 + 新任务: 零层级权重（纯片段模型）
                hier_w_values = torch.zeros(batch_size, 1, device=self.device)
                for i in range(batch_size):
                    if is_known_subject[i] and is_known_task[i]:
                        # 已知被试 + 已知任务
                        hier_w_values[i] = self.ensemble_config.known_task_hier_weight
                    elif is_known_subject[i] and not is_known_task[i]:
                        # 已知被试 + 新任务
                        hier_w_values[i] = self.ensemble_config.known_subject_new_task_hier_weight
                    elif not is_known_subject[i] and is_known_task[i]:
                        # 新被试 + 已知任务
                        hier_w_values[i] = self.ensemble_config.known_task_hier_weight
                    else:
                        # 新被试 + 新任务：纯片段模型
                        hier_w_values[i] = self.ensemble_config.new_subject_new_task_hier_weight

                hier_w = hier_w_values
            seg_w = 1.0 - hier_w

        else:
            raise ValueError(f"未知的集成策略: {strategy}")

        return hier_w, seg_w

    @torch.no_grad()
    def predict_hierarchical(
        self,
        dataloader: DataLoader,
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[Optional[torch.Tensor]]]:
        """
        使用层级模型进行预测

        Args:
            dataloader: 层级数据集的 DataLoader

        Returns:
            (probs, labels, subject_ids, task_conditions_list): 概率预测、真实标签、被试ID列表、任务条件列表
        """
        all_probs = []
        all_labels = []
        all_subject_ids = []
        all_task_conditions = []

        for batch in dataloader:
            # 移动数据到设备
            segments = batch['segments'].to(self.device)
            segment_mask = batch['segment_mask'].to(self.device)
            task_mask = batch['task_mask'].to(self.device)
            segment_lengths = batch.get('segment_lengths')
            if segment_lengths is not None:
                segment_lengths = segment_lengths.to(self.device)
            task_conditions = batch.get('task_conditions')
            if task_conditions is not None:
                task_conditions = task_conditions.to(self.device)

            # 前向传播
            outputs = self.hierarchical_model(
                segments=segments,
                segment_mask=segment_mask,
                task_mask=task_mask,
                segment_lengths=segment_lengths,
                task_conditions=task_conditions,
            )

            logits = outputs['prediction']  # (batch, num_classes)
            probs = F.softmax(logits, dim=-1)

            all_probs.append(probs.cpu().numpy())
            # 注意：collate_fn 返回的是 'label' 而非 'labels'
            all_labels.append(batch['label'].numpy())
            # 收集被试ID
            all_subject_ids.extend(batch['subject_ids'])
            all_task_conditions.append(task_conditions)

        return (
            np.concatenate(all_probs, axis=0),
            np.concatenate(all_labels, axis=0),
            all_subject_ids,
            all_task_conditions,
        )

    @torch.no_grad()
    def predict_segment_aggregated(
        self,
        segment_dataset: SegmentGazeDataset,
    ) -> Dict[str, Tuple[np.ndarray, int]]:
        """
        使用片段模型进行预测并聚合到被试级

        Args:
            segment_dataset: 片段数据集

        Returns:
            字典：{subject_id: (probs, label)}
        """
        dataloader = DataLoader(
            segment_dataset,
            batch_size=self.ensemble_config.batch_size,
            shuffle=False,
            collate_fn=segment_collate_fn,
        )

        # 收集所有片段预测
        segment_probs = []
        segment_labels = []
        segment_subject_ids = []

        for batch in dataloader:
            features = batch['features'].to(self.device)
            # 注意：segment_collate_fn 返回的是 'length' 而非 'lengths'
            lengths = batch.get('length')
            if lengths is not None:
                lengths = lengths.to(self.device)
            task_conditions = batch.get('task_conditions')
            if task_conditions is not None:
                # task_conditions 可能是字典格式
                if isinstance(task_conditions, dict):
                    task_conditions = None  # 暂时不支持字典格式的任务条件
                else:
                    task_conditions = task_conditions.to(self.device)

            # 前向传播
            logits = self.segment_model(
                features=features,
                lengths=lengths,
                task_conditions=task_conditions,
            )

            probs = F.softmax(logits, dim=-1)
            segment_probs.append(probs.cpu().numpy())
            segment_labels.append(batch['labels'].numpy().flatten())  # 确保是 1D
            segment_subject_ids.extend(batch['subject_ids'])

        segment_probs = np.concatenate(segment_probs, axis=0)
        segment_labels = np.concatenate(segment_labels, axis=0)

        # 聚合到被试级（返回字典便于按ID查找）
        subject_results = {}
        unique_subjects = list(dict.fromkeys(segment_subject_ids))

        for subject_id in unique_subjects:
            mask = [i for i, sid in enumerate(segment_subject_ids) if sid == subject_id]
            # 取该被试所有片段概率的平均
            avg_probs = segment_probs[mask].mean(axis=0)
            # 标签应该都相同，确保是标量
            label = int(segment_labels[mask[0]])
            subject_results[subject_id] = (avg_probs, label)

        return subject_results

    def predict(
        self,
        hierarchical_dataset: Dataset,
        segment_dataset: SegmentGazeDataset,
        return_individual: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        集成预测

        Args:
            hierarchical_dataset: 层级数据集
            segment_dataset: 片段数据集
            return_individual: 是否返回各模型的单独预测

        Returns:
            结果字典，包含:
            - ensemble_probs: 集成概率
            - ensemble_preds: 集成预测
            - labels: 真实标签
            - (可选) hier_probs, seg_probs: 各模型概率
        """
        if self.hierarchical_model is None or self.segment_model is None:
            raise RuntimeError("请先调用 load_models() 加载模型权重")

        # 层级模型预测
        hier_dataloader = DataLoader(
            hierarchical_dataset,
            batch_size=self.ensemble_config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        hier_probs, hier_labels, hier_subject_ids, task_conditions_list = self.predict_hierarchical(hier_dataloader)

        # 片段模型预测（聚合到被试级，返回字典）
        seg_results = self.predict_segment_aggregated(segment_dataset)

        # 按照层级模型的被试顺序对齐片段模型的预测
        seg_probs = []
        seg_labels = []
        missing_subjects = []

        for subject_id in hier_subject_ids:
            if subject_id in seg_results:
                probs, label = seg_results[subject_id]
                seg_probs.append(probs)
                seg_labels.append(label)
            else:
                missing_subjects.append(subject_id)
                # 如果片段模型没有这个被试，使用均匀分布
                seg_probs.append(np.ones(self.num_classes) / self.num_classes)
                seg_labels.append(0)

        if missing_subjects:
            logger.warning(f"片段模型缺少 {len(missing_subjects)} 个被试: {missing_subjects[:5]}...")

        seg_probs = np.array(seg_probs)
        seg_labels = np.array(seg_labels)

        # 验证标签一致性
        if not np.array_equal(hier_labels, seg_labels):
            # 打印调试信息
            mismatch_count = np.sum(hier_labels != seg_labels)
            logger.warning(f"层级模型和片段模型有 {mismatch_count}/{len(hier_labels)} 个标签不匹配")

        # 获取已知被试集合（用于 dual_aware 和 learned 策略）
        known_subjects = getattr(hierarchical_dataset, 'known_subjects', None)

        # 计算集成预测
        batch_size = len(hier_labels)

        if self.ensemble_config.strategy == "learned":
            # 使用学习融合网络
            if self.fusion_network is None:
                raise RuntimeError("learned 策略需要先加载模型（融合网络未初始化）")

            # 准备输入数据
            hier_probs_tensor = torch.from_numpy(hier_probs).to(self.device)  # (N, num_classes)
            seg_probs_tensor = torch.from_numpy(seg_probs).to(self.device)   # (N, num_classes)

            # 准备任务条件
            task_cond_input = None
            if task_conditions_list and task_conditions_list[0] is not None:
                all_task_conditions = torch.cat(task_conditions_list, dim=0)  # (N, max_tasks, 5)
                # 转换为浮点数后再取平均
                task_cond_input = all_task_conditions.float().mean(dim=1)  # (N, 5)

            # 准备被试信息
            if known_subjects is not None:
                is_known_subject = torch.tensor([
                    sid in known_subjects for sid in hier_subject_ids
                ], dtype=torch.float32, device=self.device)
            else:
                is_known_subject = None

            # 融合网络前向传播
            with torch.no_grad():
                logits = self.fusion_network(
                    hier_probs_tensor,
                    seg_probs_tensor,
                    task_conditions=task_cond_input,
                    is_known_subject=is_known_subject,
                )
                ensemble_probs = F.softmax(logits, dim=-1).cpu().numpy()
                ensemble_preds = ensemble_probs.argmax(axis=1)

            # 计算平均权重（用于日志）
            # 通过比较预测结果来推断权重
            hier_contribution = np.mean(
                np.abs(ensemble_probs - seg_probs) /
                (np.abs(ensemble_probs - hier_probs) + np.abs(ensemble_probs - seg_probs) + 1e-8)
            )
            hier_weights = 1.0 - hier_contribution
            seg_weights = 1.0 - hier_weights

        elif self.ensemble_config.strategy in ["condition_aware", "dual_aware"] and task_conditions_list and task_conditions_list[0] is not None:
            # 合并所有批次的任务条件
            all_task_conditions = torch.cat(task_conditions_list, dim=0)  # (N, max_tasks, 5)

            if self.ensemble_config.strategy == "dual_aware" and known_subjects is not None:
                # 双重感知策略：同时考虑任务和被试
                is_known_task = self.is_known_task_condition(all_task_conditions)  # (N,)
                is_known_subject = torch.tensor([
                    sid in known_subjects for sid in hier_subject_ids
                ], dtype=torch.bool)

                # 计算权重
                hier_weights_list = []
                for i in range(batch_size):
                    if is_known_subject[i] and is_known_task[i]:
                        # 已知被试 + 已知任务
                        hier_weights_list.append(self.ensemble_config.known_task_hier_weight)
                    elif is_known_subject[i] and not is_known_task[i]:
                        # 已知被试 + 新任务
                        hier_weights_list.append(self.ensemble_config.known_subject_new_task_hier_weight)
                    elif not is_known_subject[i] and is_known_task[i]:
                        # 新被试 + 已知任务
                        hier_weights_list.append(self.ensemble_config.known_task_hier_weight)
                    else:
                        # 新被试 + 新任务：纯片段模型
                        hier_weights_list.append(self.ensemble_config.new_subject_new_task_hier_weight)

                hier_weights = np.array(hier_weights_list)[:, None]
                seg_weights = 1.0 - hier_weights
            else:
                # 条件感知策略：只考虑任务
                is_known = self.is_known_task_condition(all_task_conditions)  # (N,)
                hier_weights = np.where(
                    is_known.cpu().numpy(),
                    self.ensemble_config.known_task_hier_weight,
                    self.ensemble_config.new_task_hier_weight,
                )[:, None]
                seg_weights = 1.0 - hier_weights
        else:
            # 使用固定权重
            hier_weights = np.full((batch_size, 1), self.ensemble_config.hierarchical_weight)
            seg_weights = np.full((batch_size, 1), self.ensemble_config.segment_weight)

        # 加权融合（learned策略已经计算了ensemble_probs）
        if self.ensemble_config.strategy != "learned":
            ensemble_probs = hier_weights * hier_probs + seg_weights * seg_probs
            ensemble_preds = ensemble_probs.argmax(axis=1)

        results = {
            'ensemble_probs': ensemble_probs,
            'ensemble_preds': ensemble_preds,
            'labels': hier_labels,
            'hier_weights': hier_weights.flatten(),
            'seg_weights': seg_weights.flatten(),
        }

        if return_individual:
            results['hier_probs'] = hier_probs
            results['hier_preds'] = hier_probs.argmax(axis=1)
            results['seg_probs'] = seg_probs
            results['seg_preds'] = seg_probs.argmax(axis=1)

        return results

    def evaluate(
        self,
        hierarchical_dataset: Dataset,
        segment_dataset: SegmentGazeDataset,
    ) -> Dict[str, float]:
        """
        评估集成模型

        Args:
            hierarchical_dataset: 层级数据集
            segment_dataset: 片段数据集

        Returns:
            评估指标字典
        """
        results = self.predict(hierarchical_dataset, segment_dataset, return_individual=True)

        labels = results['labels']
        ensemble_preds = results['ensemble_preds']
        hier_preds = results['hier_preds']
        seg_preds = results['seg_preds']

        # 计算指标
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        metrics = {
            # 集成模型指标
            'ensemble_accuracy': accuracy_score(labels, ensemble_preds),
            'ensemble_f1': f1_score(labels, ensemble_preds, average='macro'),
            'ensemble_precision': precision_score(labels, ensemble_preds, average='macro', zero_division=0),
            'ensemble_recall': recall_score(labels, ensemble_preds, average='macro', zero_division=0),
            # 层级模型指标
            'hier_accuracy': accuracy_score(labels, hier_preds),
            'hier_f1': f1_score(labels, hier_preds, average='macro'),
            # 片段模型指标
            'seg_accuracy': accuracy_score(labels, seg_preds),
            'seg_f1': f1_score(labels, seg_preds, average='macro'),
            # 权重统计
            'avg_hier_weight': float(results['hier_weights'].mean()),
            'avg_seg_weight': float(results['seg_weights'].mean()),
        }

        return metrics


def search_optimal_weights(
    predictor: EnsemblePredictor,
    val_hierarchical_dataset: Dataset,
    val_segment_dataset: SegmentGazeDataset,
    weight_range: Tuple[float, float] = (0.0, 1.0),
    num_steps: int = 21,
) -> Tuple[float, float, Dict]:
    """
    在验证集上搜索最优权重

    Args:
        predictor: 集成预测器（已加载模型）
        val_hierarchical_dataset: 验证集层级数据
        val_segment_dataset: 验证集片段数据
        weight_range: 权重搜索范围
        num_steps: 搜索步数

    Returns:
        (best_hier_weight, best_f1, search_results): 最优权重、最佳F1、搜索结果
    """
    # 获取两个模型的预测
    predictor.ensemble_config.strategy = "fixed"
    predictor.ensemble_config.hierarchical_weight = 0.5
    predictor.ensemble_config.segment_weight = 0.5

    results = predictor.predict(
        val_hierarchical_dataset, val_segment_dataset, return_individual=True
    )

    hier_probs = results['hier_probs']
    seg_probs = results['seg_probs']
    labels = results['labels']

    # 网格搜索
    weights = np.linspace(weight_range[0], weight_range[1], num_steps)
    search_results = []
    best_f1 = -1
    best_weight = 0.5

    from sklearn.metrics import accuracy_score, f1_score

    for hier_w in weights:
        seg_w = 1.0 - hier_w
        ensemble_probs = hier_w * hier_probs + seg_w * seg_probs
        ensemble_preds = ensemble_probs.argmax(axis=1)

        acc = accuracy_score(labels, ensemble_preds)
        f1 = f1_score(labels, ensemble_preds, average='macro')

        search_results.append({
            'hier_weight': hier_w,
            'seg_weight': seg_w,
            'accuracy': acc,
            'f1': f1,
        })

        if f1 > best_f1:
            best_f1 = f1
            best_weight = hier_w

    logger.info(f"最优权重搜索完成: hier_weight={best_weight:.2f}, f1={best_f1:.4f}")

    return best_weight, best_f1, search_results
