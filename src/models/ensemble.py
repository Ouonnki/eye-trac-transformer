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
    strategy: str = "condition_aware"  # "fixed", "condition_aware", "learned"

    # 固定权重模式参数
    hierarchical_weight: float = 0.5
    segment_weight: float = 0.5

    # 条件感知模式参数
    known_task_hier_weight: float = 0.7  # 已知任务时层级模型权重
    new_task_hier_weight: float = 0.1    # 新任务时层级模型权重
    max_known_grid_size: int = 25        # 训练时见过的最大 grid_size
    max_known_distractor: int = 8        # 训练时见过的最大 distractor_count

    # 推理参数
    batch_size: int = 8
    device: str = "cuda"


class EnsemblePredictor:
    """
    集成预测器

    结合层级模型和片段模型的预测结果。

    支持三种集成策略：
    1. fixed: 固定权重加权平均
    2. condition_aware: 根据任务条件动态调整权重
    3. learned: 使用学习的融合层
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
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根据任务条件计算动态权重

        Args:
            task_conditions: 任务条件张量
            batch_size: 批次大小

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
            # 条件感知权重
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

        else:
            raise ValueError(f"未知的集成策略: {strategy}")

        return hier_w, seg_w

    @torch.no_grad()
    def predict_hierarchical(
        self,
        dataloader: DataLoader,
    ) -> Tuple[np.ndarray, np.ndarray, List[Optional[torch.Tensor]]]:
        """
        使用层级模型进行预测

        Args:
            dataloader: 层级数据集的 DataLoader

        Returns:
            (probs, labels, task_conditions_list): 概率预测、真实标签、任务条件列表
        """
        all_probs = []
        all_labels = []
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
            all_task_conditions.append(task_conditions)

        return (
            np.concatenate(all_probs, axis=0),
            np.concatenate(all_labels, axis=0),
            all_task_conditions,
        )

    @torch.no_grad()
    def predict_segment_aggregated(
        self,
        segment_dataset: SegmentGazeDataset,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用片段模型进行预测并聚合到被试级

        Args:
            segment_dataset: 片段数据集

        Returns:
            (probs, labels): 被试级概率预测和真实标签
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
            segment_labels.append(batch['labels'].numpy())
            segment_subject_ids.extend(batch['subject_ids'])

        segment_probs = np.concatenate(segment_probs, axis=0)
        segment_labels = np.concatenate(segment_labels, axis=0)

        # 聚合到被试级（按被试ID分组取平均）
        unique_subjects = list(dict.fromkeys(segment_subject_ids))  # 保持顺序
        subject_probs = []
        subject_labels = []

        for subject_id in unique_subjects:
            mask = [i for i, sid in enumerate(segment_subject_ids) if sid == subject_id]
            # 取该被试所有片段概率的平均
            subject_probs.append(segment_probs[mask].mean(axis=0))
            # 标签应该都相同
            subject_labels.append(segment_labels[mask[0]])

        return np.array(subject_probs), np.array(subject_labels)

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
        hier_probs, hier_labels, task_conditions_list = self.predict_hierarchical(hier_dataloader)

        # 片段模型预测（聚合到被试级）
        seg_probs, seg_labels = self.predict_segment_aggregated(segment_dataset)

        # 验证标签一致性
        if not np.array_equal(hier_labels, seg_labels):
            logger.warning("层级模型和片段模型的标签顺序不一致，可能影响集成结果")

        # 计算集成权重
        batch_size = len(hier_labels)

        # 对于条件感知策略，需要基于任务条件计算权重
        if self.ensemble_config.strategy == "condition_aware" and task_conditions_list[0] is not None:
            # 逐样本计算权重
            all_hier_weights = []
            for tc in task_conditions_list:
                if tc is not None:
                    is_known = self.is_known_task_condition(tc)
                    weights = torch.where(
                        is_known,
                        torch.tensor(self.ensemble_config.known_task_hier_weight),
                        torch.tensor(self.ensemble_config.new_task_hier_weight),
                    )
                    all_hier_weights.append(weights.cpu().numpy())
            hier_weights = np.concatenate(all_hier_weights)[:, None]  # (N, 1)
            seg_weights = 1.0 - hier_weights
        else:
            # 使用固定权重
            hier_weights = np.full((batch_size, 1), self.ensemble_config.hierarchical_weight)
            seg_weights = np.full((batch_size, 1), self.ensemble_config.segment_weight)

        # 加权融合
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
