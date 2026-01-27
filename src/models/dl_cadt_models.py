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

from src.config import UnifiedConfig
from src.models.base import BaseModel
from src.models.encoders import HierarchicalEncoder
from src.models.dl_dataset import SequenceConfig


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


class CADTTransformerModel(BaseModel):
    """
    CADT-Transformer 域适应模型

    借鉴 CADT 的域适应思想：
    1. 双辨别器机制：discriminator（特征+距离）和 discriminator2（纯特征）
    2. 原型学习：计算源域各类别的特征中心，通过 kl_loss 对齐
    3. 两阶段训练：预训练阶段 + 正式训练阶段

    使用 from_config() 从配置创建实例。
    """

    def __init__(
        self,
        model_config: 'ModelConfig',
        seq_config: SequenceConfig,
        device_config: 'DeviceConfig',
        cadt_config: 'CADTConfig',
        num_classes: int,
    ):
        """
        初始化

        Args:
            model_config: 模型架构配置
            seq_config: 序列配置
            device_config: 设备配置
            cadt_config: CADT 域适应配置
            num_classes: 类别数量
        """
        super().__init__()

        self.num_classes = num_classes
        self.feature_dim = model_config.task_d_model

        # 特征编码器（复用层级编码器）
        self.encoder = HierarchicalEncoder(
            model_config=model_config,
            seq_config=seq_config,
            use_gradient_checkpointing=device_config.use_gradient_checkpointing,
        )

        # 分类器
        self.classifier = Classifier(self.feature_dim, num_classes)

        # 双辨别器（与原始 CADT 一致）
        # discriminator: 输入 (特征 + 到各类原型的最小距离)，用于对抗训练
        self.discriminator = Discriminator(self.feature_dim + 1, hidden_size=64)
        # discriminator2: 输入纯特征
        self.discriminator2 = Discriminator(self.feature_dim, hidden_size=64)

        # 损失函数
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=cadt_config.label_smoothing)
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.bce_loss = nn.BCEWithLogitsLoss()

        # 类别原型中心（在 init_center_c 中初始化）
        self.c = None  # shape: (num_classes, feature_dim)

        # one-hot 编码矩阵
        self.register_buffer('eye_matrix', torch.eye(num_classes))

    @classmethod
    def from_config(
        cls,
        config: UnifiedConfig,
        seq_config: SequenceConfig,
        **kwargs,
    ) -> 'CADTTransformerModel':
        """
        从配置创建模型

        Args:
            config: 统一配置对象
            seq_config: 序列配置对象
            **kwargs: 额外参数（如 num_classes）

        Returns:
            模型实例
        """
        # 从 kwargs 获取 num_classes，默认根据 task 类型决定
        num_classes = kwargs.get('num_classes', None)
        if num_classes is None:
            num_classes = config.task.num_classes

        return cls(
            model_config=config.model,
            seq_config=seq_config,
            device_config=config.device,
            cadt_config=config.cadt,
            num_classes=num_classes,
        )

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
        if change_center:
            # 分类损失
            src_pred = self.classifier(src_features)
            ce_loss = self.ce_loss(src_pred, src_labels)

            # 域分类损失（discriminator2，非对抗）
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
                'need_discriminator_step': False,
            }

        # 正式训练阶段：完整损失 + 两阶段更新
        py = self.c[src_labels].detach()  # (batch, feature_dim)

        # 计算到各中心的最小距离
        src_dist = self.compute_distance_to_centers(src_features)
        tgt_dist = self.compute_distance_to_centers(tgt_features)

        # 原型聚类损失
        kl_loss = self.mse_loss(src_features, py)

        # 分类损失
        src_pred = self.classifier(src_features)
        ce_loss = self.ce_loss(src_pred, src_labels)

        # 对抗损失（用翻转标签训练 encoder 欺骗 discriminator）
        src_fake_label = torch.ones(batch_size_src, 1, device=device)
        tgt_real_label = torch.zeros(batch_size_tgt, 1, device=device)

        dis_loss_adv = self.bce_loss(
            self.discriminator(torch.cat([src_features, src_dist], dim=1)), src_fake_label
        ) + self.bce_loss(
            self.discriminator(torch.cat([tgt_features, tgt_dist], dim=1)), tgt_real_label
        )

        # 域分类损失（discriminator2，非对抗）
        src_label = torch.zeros(batch_size_src, 1, device=device)
        tgt_label = torch.ones(batch_size_tgt, 1, device=device)

        domain_loss = self.bce_loss(self.discriminator2(src_features), src_label) + \
                      self.bce_loss(self.discriminator2(tgt_features), tgt_label)

        # 总损失
        total_loss = ce_loss + kl_loss * kl_w + dis_loss_adv * dis_w + domain_loss

        # 第二阶段：单独训练 discriminator 的损失
        src_real_label = torch.zeros(batch_size_src, 1, device=device)
        tgt_fake_label = torch.ones(batch_size_tgt, 1, device=device)

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
            'need_discriminator_step': True,
            'dis_loss_for_discriminator': dis_loss_real,
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
