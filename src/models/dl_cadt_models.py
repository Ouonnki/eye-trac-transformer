# -*- coding: utf-8 -*-
"""
CADT-Transformer 域适应模型模块

基于 CADT 思想实现的 Transformer 域适应模型。
借鉴 CADT 的双辨别器机制和原型学习，适配 Transformer 特征提取器。
"""

from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
from torch.autograd import Function

from src.models.dl_models import (
    HierarchicalTransformerNetwork,
    GazeTransformerEncoder,
    TaskTransformerEncoder,
)
from src.models.attention import AttentionPooling


class GradientReversalFunction(Function):
    """
    梯度反转函数

    前向传播时不改变输入，反向传播时将梯度乘以 -lambda。
    用于域对抗训练，使编码器学习域不变特征。
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    """
    梯度反转层 (GRL)

    在前向传播时作为恒等映射，在反向传播时反转梯度。
    这使得编码器学习欺骗辨别器的特征表示。
    """

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_: float):
        self.lambda_ = lambda_


class Classifier(nn.Module):
    """
    任务分类器

    简单的线性分类器，将特征映射到类别空间。
    """

    def __init__(self, input_size: int, num_classes: int):
        """
        初始化

        Args:
            input_size: 输入特征维度
            num_classes: 类别数量
        """
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: (batch, input_size) 输入特征

        Returns:
            (batch, num_classes) 类别 logits
        """
        return self.linear(x)


class Discriminator(nn.Module):
    """
    域辨别器

    三层 MLP，用于区分源域和目标域样本。
    输出 logits（不含 Sigmoid），配合 BCEWithLogitsLoss 使用以支持 AMP。
    """

    def __init__(self, input_size: int, hidden_size: int = 64):
        """
        初始化

        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层维度
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(hidden_size, 1),
            # 移除 Sigmoid，配合 BCEWithLogitsLoss 使用
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: (batch, input_size) 输入特征

        Returns:
            (batch, 1) 域预测 logits（0=源域，1=目标域）
        """
        return self.model(x)


class HierarchicalTransformerEncoder(nn.Module):
    """
    层级 Transformer 编码器（仅特征提取部分）

    复用 HierarchicalTransformerNetwork 的结构，但不包含预测头。
    输出 subject_repr 用于域适应。
    """

    def __init__(
        self,
        input_dim: int = 7,
        segment_d_model: int = 64,
        segment_nhead: int = 4,
        segment_num_layers: int = 4,
        task_d_model: int = 128,
        task_nhead: int = 4,
        task_num_layers: int = 2,
        attention_dim: int = 32,
        dropout: float = 0.1,
        max_seq_len: int = 100,
        max_tasks: int = 30,
        max_segments: int = 30,
        use_gradient_checkpointing: bool = False,
    ):
        """
        初始化

        Args:
            input_dim: 输入特征维度
            segment_d_model: 片段编码器维度
            segment_nhead: 片段编码器注意力头数
            segment_num_layers: 片段编码器层数
            task_d_model: 任务编码器维度
            task_nhead: 任务编码器注意力头数
            task_num_layers: 任务编码器层数
            attention_dim: 聚合注意力维度
            dropout: Dropout 比例
            max_seq_len: 最大序列长度
            max_tasks: 最大任务数
            max_segments: 最大片段数
            use_gradient_checkpointing: 是否使用梯度检查点
        """
        super().__init__()

        self.max_tasks = max_tasks
        self.max_segments = max_segments
        self.output_dim = task_d_model

        # 片段编码器
        self.segment_encoder = GazeTransformerEncoder(
            input_dim=input_dim,
            d_model=segment_d_model,
            nhead=segment_nhead,
            num_layers=segment_num_layers,
            dim_feedforward=segment_d_model * 4,
            dropout=dropout,
            max_seq_len=max_seq_len,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

        # 任务聚合器
        self.task_aggregator = AttentionPooling(
            input_dim=segment_d_model,
            attention_dim=attention_dim,
            dropout=dropout,
        )

        # 任务序列编码器
        self.task_encoder = TaskTransformerEncoder(
            input_dim=segment_d_model,
            d_model=task_d_model,
            nhead=task_nhead,
            num_layers=task_num_layers,
            dim_feedforward=task_d_model * 4,
            dropout=dropout,
            max_tasks=max_tasks,
        )

        # 被试聚合器
        self.subject_aggregator = AttentionPooling(
            input_dim=task_d_model,
            attention_dim=attention_dim,
            dropout=dropout,
        )

    def forward(
        self,
        segments: torch.Tensor,
        segment_mask: torch.Tensor,
        task_mask: torch.Tensor,
        segment_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播

        Args:
            segments: (batch, max_tasks, max_segments, max_seq_len, input_dim) 眼动序列
            segment_mask: (batch, max_tasks, max_segments) 有效片段掩码
            task_mask: (batch, max_tasks) 有效任务掩码
            segment_lengths: (batch, max_tasks, max_segments) 每个片段的实际长度

        Returns:
            subject_repr: (batch, task_d_model) 被试级特征表示
            extras: 包含注意力权重等额外信息的字典
        """
        batch_size = segments.size(0)
        device = segments.device

        # 1. 编码所有片段
        flat_segments = segments.view(-1, segments.size(-2), segments.size(-1))

        if segment_lengths is not None:
            flat_lengths = segment_lengths.view(-1)
            max_len = segments.size(-2)
            seq_mask = torch.arange(max_len, device=device).unsqueeze(0) < flat_lengths.unsqueeze(1)
        else:
            seq_mask = None

        segment_reprs, _ = self.segment_encoder(flat_segments, seq_mask)
        segment_reprs = segment_reprs.view(batch_size, self.max_tasks, self.max_segments, -1)

        # 2. 聚合片段到任务
        segment_reprs_flat = segment_reprs.view(batch_size * self.max_tasks, self.max_segments, -1)
        segment_mask_flat = segment_mask.view(batch_size * self.max_tasks, self.max_segments)

        task_reprs_flat, segment_attns_flat = self.task_aggregator(segment_reprs_flat, segment_mask_flat)
        task_reprs = task_reprs_flat.view(batch_size, self.max_tasks, -1)
        segment_attentions = segment_attns_flat.view(batch_size, self.max_tasks, self.max_segments)

        # 3. 编码任务序列
        task_encoded = self.task_encoder(task_reprs, task_mask)

        # 4. 聚合任务到被试
        subject_repr, task_attention = self.subject_aggregator(task_encoded, task_mask)

        extras = {
            'segment_attention': segment_attentions,
            'task_attention': task_attention,
        }

        return subject_repr, extras


class CADTTransformerModel(nn.Module):
    """
    CADT-Transformer 域适应模型

    借鉴 CADT 的域适应思想：
    1. 双辨别器机制：discriminator（特征+距离）和 discriminator2（纯特征）
    2. 原型学习：计算源域各类别的特征中心，通过 kl_loss 对齐
    3. 两阶段训练：预训练阶段 + 正式训练阶段
    """

    def __init__(self, config):
        """
        初始化

        Args:
            config: CADTConfig 配置对象
        """
        super().__init__()

        self.config = config
        self.num_classes = config.num_classes
        self.feature_dim = config.task_d_model  # 直接使用 Transformer 的特征维度

        # 特征编码器
        self.encoder = HierarchicalTransformerEncoder(
            input_dim=config.input_dim,
            segment_d_model=config.segment_d_model,
            segment_nhead=config.segment_nhead,
            segment_num_layers=config.segment_num_layers,
            task_d_model=config.task_d_model,
            task_nhead=config.task_nhead,
            task_num_layers=config.task_num_layers,
            attention_dim=config.attention_dim,
            dropout=config.dropout,
            max_seq_len=config.max_seq_len,
            max_tasks=config.max_tasks,
            max_segments=config.max_segments,
            use_gradient_checkpointing=config.use_gradient_checkpointing,
        )

        # 分类器
        self.classifier = Classifier(self.feature_dim, self.num_classes)

        # 双辨别器（与原始 CADT 一致）
        # discriminator: 输入 (特征 + 到各类原型的最小距离)，用于对抗训练
        self.discriminator = Discriminator(self.feature_dim + 1, hidden_size=64)
        # discriminator2: 输入纯特征
        self.discriminator2 = Discriminator(self.feature_dim, hidden_size=64)

        # 损失函数
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label Smoothing 防止过拟合
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.bce_loss = nn.BCEWithLogitsLoss()  # 使用 BCEWithLogitsLoss 以支持 AMP

        # 类别原型中心（在 init_center_c 中初始化）
        self.c = None  # shape: (num_classes, feature_dim)

        # one-hot 编码矩阵
        self.register_buffer('eye_matrix', torch.eye(self.num_classes))

    def reset(self):
        """
        重置模型参数（完全重置）

        重新初始化所有网络模块的参数，包括 BatchNorm/LayerNorm 的运行统计量。
        用于原始 CADT 的两阶段训练策略：预训练后完全重置网络。
        """
        def _reset_module(module):
            """递归重置模块参数"""
            for child in module.children():
                _reset_module(child)
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
            # 重置 BatchNorm/LayerNorm 的运行统计量
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                if hasattr(module, 'running_mean') and module.running_mean is not None:
                    module.running_mean.zero_()
                if hasattr(module, 'running_var') and module.running_var is not None:
                    module.running_var.fill_(1)
                if hasattr(module, 'num_batches_tracked'):
                    module.num_batches_tracked.zero_()

        _reset_module(self.encoder)
        _reset_module(self.classifier)
        _reset_module(self.discriminator)
        _reset_module(self.discriminator2)

    def init_center_c(self, source_loader, device):
        """
        初始化类别原型中心

        计算源域各类别特征的均值作为原型。

        Args:
            source_loader: 源域数据加载器
            device: 计算设备
        """
        n_samples = torch.zeros(self.num_classes, device=device)
        c = torch.zeros(self.num_classes, self.feature_dim, device=device)

        self.eval()
        with torch.no_grad():
            for batch in source_loader:
                segments = batch['segments'].to(device)
                segment_mask = batch['segment_mask'].to(device)
                task_mask = batch['task_mask'].to(device)
                segment_lengths = batch['segment_lengths'].to(device)
                labels = batch['label'].to(device)

                # 获取特征
                features, _ = self.encoder(segments, segment_mask, task_mask, segment_lengths)

                # 累加各类别特征
                y_onehot = self.eye_matrix[labels]  # (batch, num_classes)
                n_samples = n_samples + y_onehot.sum(dim=0)
                c = c + torch.matmul(y_onehot.T, features)

        # 计算均值
        n_samples[n_samples == 0] = 1.0  # 避免除零
        c = c / n_samples.unsqueeze(1)
        self.c = c.clone().detach()

    def compute_distance_to_centers(self, features: torch.Tensor) -> torch.Tensor:
        """
        计算特征到各类别中心的最小距离

        Args:
            features: (batch, feature_dim) 特征向量

        Returns:
            (batch, 1) 到最近中心的归一化距离
        """
        if self.c is None:
            return torch.zeros(features.size(0), 1, device=features.device)

        # 计算到各类别中心的距离
        distances = []
        for i in range(self.num_classes):
            dist = torch.norm(features - self.c[i].detach(), dim=1, keepdim=True)
            distances.append(dist)

        distances = torch.cat(distances, dim=1)  # (batch, num_classes)
        min_distance = distances.min(dim=1, keepdim=True).values / self.feature_dim

        return min_distance

    def forward(
        self,
        segments: torch.Tensor,
        segment_mask: torch.Tensor,
        task_mask: torch.Tensor,
        segment_lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            segments: (batch, max_tasks, max_segments, max_seq_len, input_dim)
            segment_mask: (batch, max_tasks, max_segments)
            task_mask: (batch, max_tasks)
            segment_lengths: (batch, max_tasks, max_segments)

        Returns:
            字典包含：
            - prediction: (batch, num_classes) 类别 logits
            - features: (batch, feature_dim) 特征向量
            - segment_attention: 片段注意力权重
            - task_attention: 任务注意力权重
        """
        features, extras = self.encoder(segments, segment_mask, task_mask, segment_lengths)
        prediction = self.classifier(features)

        return {
            'prediction': prediction,
            'features': features,
            'segment_attention': extras['segment_attention'],
            'task_attention': extras['task_attention'],
        }

    def train_step(
        self,
        source_batch: Dict[str, torch.Tensor],
        target_batch: Dict[str, torch.Tensor],
        device: torch.device,
        change_center: bool = False,
        kl_w: float = 0.0,
        dis_w: float = 1.0,
    ) -> Dict[str, float]:
        """
        CADT 训练步骤

        Args:
            source_batch: 源域批次数据（带标签）
            target_batch: 目标域批次数据（无标签）
            device: 计算设备
            change_center: 是否处于预训练阶段（True=预训练，False=正式训练）
            kl_w: 原型聚类损失权重
            dis_w: 域对抗损失权重

        Returns:
            损失值字典
        """
        # 提取源域数据
        src_segments = source_batch['segments'].to(device)
        src_segment_mask = source_batch['segment_mask'].to(device)
        src_task_mask = source_batch['task_mask'].to(device)
        src_segment_lengths = source_batch['segment_lengths'].to(device)
        src_labels = source_batch['label'].to(device)

        # 提取目标域数据
        tgt_segments = target_batch['segments'].to(device)
        tgt_segment_mask = target_batch['segment_mask'].to(device)
        tgt_task_mask = target_batch['task_mask'].to(device)
        tgt_segment_lengths = target_batch['segment_lengths'].to(device)

        batch_size_src = src_segments.size(0)
        batch_size_tgt = tgt_segments.size(0)

        # 获取特征
        src_features, _ = self.encoder(src_segments, src_segment_mask, src_task_mask, src_segment_lengths)
        tgt_features, _ = self.encoder(tgt_segments, tgt_segment_mask, tgt_task_mask, tgt_segment_lengths)

        # 预训练阶段：只使用分类损失 + 域分类损失（discriminator2）
        # 注意：原始 CADT 中 discriminator2 不使用对抗训练，直接学习域分类
        if change_center:
            # 分类损失
            src_pred = self.classifier(src_features)
            ce_loss = self.ce_loss(src_pred, src_labels)

            # 域分类损失（discriminator2，非对抗）
            # 源域=0, 目标域=1
            src_label = torch.zeros(batch_size_src, 1, device=device)
            tgt_label = torch.ones(batch_size_tgt, 1, device=device)

            domain_loss = self.bce_loss(self.discriminator2(src_features), src_label) + \
                          self.bce_loss(self.discriminator2(tgt_features), tgt_label)

            total_loss = ce_loss + domain_loss * dis_w

            return {
                'total_loss': total_loss,
                'ce_loss': ce_loss.item(),
                'domain_loss': domain_loss.item(),
                'kl_loss': 0.0,
                'dis_loss': 0.0,
                'need_discriminator_step': False,  # 预训练阶段不需要单独更新 discriminator
            }

        # 正式训练阶段：完整损失 + 两阶段更新（与原始 CADT 一致）
        # 获取原型对齐目标
        py = self.c[src_labels].detach()  # (batch, feature_dim)

        # 计算到各中心的最小距离
        src_dist = self.compute_distance_to_centers(src_features)  # (batch, 1)
        tgt_dist = self.compute_distance_to_centers(tgt_features)  # (batch, 1)

        # 原型聚类损失
        kl_loss = self.mse_loss(src_features, py)

        # 分类损失
        src_pred = self.classifier(src_features)
        ce_loss = self.ce_loss(src_pred, src_labels)

        # === 第一阶段：对抗损失（用翻转标签训练 encoder 欺骗 discriminator）===
        # 原始 CADT: target 标记为 source (0), source 标记为 target (1)
        # 这让 encoder 学习生成让 discriminator 混淆的特征
        src_fake_label = torch.ones(batch_size_src, 1, device=device)   # source 伪装成 target
        tgt_real_label = torch.zeros(batch_size_tgt, 1, device=device)  # target 伪装成 source

        dis_loss_adv = self.bce_loss(
            self.discriminator(torch.cat([src_features, src_dist], dim=1)), src_fake_label
        ) + self.bce_loss(
            self.discriminator(torch.cat([tgt_features, tgt_dist], dim=1)), tgt_real_label
        )

        # 域分类损失（discriminator2，非对抗，直接训练）
        src_label = torch.zeros(batch_size_src, 1, device=device)
        tgt_label = torch.ones(batch_size_tgt, 1, device=device)

        domain_loss = self.bce_loss(self.discriminator2(src_features), src_label) + \
                      self.bce_loss(self.discriminator2(tgt_features), tgt_label)

        # 总损失（用于更新 encoder, classifier, discriminator2）
        total_loss = ce_loss + kl_loss * kl_w + dis_loss_adv * dis_w + domain_loss

        # === 第二阶段：单独训练 discriminator（使用 detach 和真实标签）===
        # 这部分在 trainer 中单独处理
        # 保存用于第二阶段的信息
        src_real_label = torch.zeros(batch_size_src, 1, device=device)  # source 真实标签
        tgt_fake_label = torch.ones(batch_size_tgt, 1, device=device)   # target 真实标签

        dis_loss_real = self.bce_loss(
            self.discriminator(torch.cat([src_features.detach(), src_dist.detach()], dim=1)), src_real_label
        ) + self.bce_loss(
            self.discriminator(torch.cat([tgt_features.detach(), tgt_dist.detach()], dim=1)), tgt_fake_label
        )

        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss.item(),
            'kl_loss': kl_loss.item(),
            'dis_loss': dis_loss_adv.item(),
            'domain_loss': domain_loss.item(),
            'need_discriminator_step': True,  # 需要单独更新 discriminator
            'dis_loss_for_discriminator': dis_loss_real,  # 用于单独更新 discriminator 的损失
        }

    def compute_classification_loss(
        self,
        batch: Dict[str, torch.Tensor],
        device: torch.device,
    ) -> Tuple[float, int, int]:
        """
        计算分类损失（用于验证）

        Args:
            batch: 数据批次
            device: 计算设备

        Returns:
            (loss, correct, total) 损失值、正确数、总数
        """
        segments = batch['segments'].to(device)
        segment_mask = batch['segment_mask'].to(device)
        task_mask = batch['task_mask'].to(device)
        segment_lengths = batch['segment_lengths'].to(device)
        labels = batch['label'].to(device)

        with torch.no_grad():
            features, _ = self.encoder(segments, segment_mask, task_mask, segment_lengths)
            pred_logits = self.classifier(features)
            _, pred = torch.max(pred_logits, dim=1)

            loss = self.ce_loss(pred_logits, labels).item()
            correct = pred.eq(labels).sum().item()
            total = labels.size(0)

        return loss, correct, total
