# -*- coding: utf-8 -*-
"""
片段级 CADT 域适应模型

将 CADT 域适应方法应用于片段级编码器（SegmentEncoder）。
采用包装器模式，复用现有的 SegmentEncoder 和 CADT 组件。
"""

from typing import Optional, Dict

import torch
import torch.nn as nn

from src.config import UnifiedConfig
from src.models.segment_model import SegmentEncoder
from src.models.dl_cadt_models import Discriminator
from src.models.dl_dataset import SequenceConfig


class SegmentCADTModel(nn.Module):
    """
    片段级 CADT 域适应模型

    包装 SegmentEncoder，添加双辨别器和原型学习支持。

    结构：
    - encoder: SegmentEncoder（复用）
    - discriminator: 特征+距离输入的域辨别器
    - discriminator2: 纯特征输入的域辨别器
    - c: 类别原型中心

    训练策略：
    1. 预训练阶段：分类损失 + 域分类损失（discriminator2）
    2. 正式训练阶段：分类损失 + 原型聚类损失 + 对抗损失（discriminator）
    """

    def __init__(
        self,
        config: UnifiedConfig,
        seq_config: SequenceConfig,
        num_classes: int = 3,
    ):
        """
        初始化

        Args:
            config: 统一配置
            seq_config: 序列配置
            num_classes: 类别数量
        """
        super().__init__()

        self.num_classes = num_classes
        self.feature_dim = config.model.segment_d_model

        # 复用 SegmentEncoder 作为特征提取器
        self.encoder = SegmentEncoder.from_config(
            config=config,
            seq_config=seq_config,
            num_classes=num_classes,
        )

        # 双辨别器（复用 CADT 组件）
        # discriminator: 输入为 特征 + 到原型的最小距离
        self.discriminator = Discriminator(self.feature_dim + 1, hidden_size=64)
        # discriminator2: 输入为纯特征
        self.discriminator2 = Discriminator(self.feature_dim, hidden_size=64)

        # 损失函数
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.bce_loss = nn.BCEWithLogitsLoss()

        # 类别原型中心 (num_classes, feature_dim)
        self.c: Optional[torch.Tensor] = None
        self.register_buffer('eye_matrix', torch.eye(num_classes))

    @classmethod
    def from_config(
        cls,
        config: UnifiedConfig,
        seq_config: SequenceConfig,
        **kwargs,
    ) -> 'SegmentCADTModel':
        """
        从配置创建模型

        Args:
            config: 统一配置
            seq_config: 序列配置
            **kwargs: 额外参数

        Returns:
            SegmentCADTModel 实例
        """
        num_classes = kwargs.get('num_classes', config.task.num_classes)
        return cls(config, seq_config, num_classes)

    def forward(
        self,
        features: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            features: (batch, seq_len, input_dim) 眼动序列
            lengths: (batch,) 序列长度

        Returns:
            字典包含 'prediction' 和 'features'
        """
        prediction, segment_features = self.encoder.forward_features(features, lengths)
        return {
            'prediction': prediction,
            'features': segment_features,
        }

    def init_center_c(self, source_loader, device: torch.device) -> None:
        """
        初始化类别原型中心

        计算源域各类别特征的均值作为原型中心。

        Args:
            source_loader: 源域数据加载器
            device: 计算设备
        """
        n_samples = torch.zeros(self.num_classes, device=device)
        c = torch.zeros(self.num_classes, self.feature_dim, device=device)

        self.eval()
        with torch.no_grad():
            for batch in source_loader:
                features = batch['features'].to(device)
                lengths = batch['length'].to(device)
                labels = batch['label'].to(device)

                _, segment_features = self.encoder.forward_features(features, lengths)

                # 累加各类别的特征
                y_onehot = self.eye_matrix[labels]
                n_samples = n_samples + y_onehot.sum(dim=0)
                c = c + torch.matmul(y_onehot.T, segment_features)

        # 计算均值
        n_samples[n_samples == 0] = 1.0
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

        distances = []
        for i in range(self.num_classes):
            dist = torch.norm(features - self.c[i].detach(), dim=1, keepdim=True)
            distances.append(dist)

        distances = torch.cat(distances, dim=1)  # (batch, num_classes)
        min_distance = distances.min(dim=1, keepdim=True).values / self.feature_dim
        return min_distance

    def train_step(
        self,
        source_batch: Dict[str, torch.Tensor],
        target_batch: Dict[str, torch.Tensor],
        device: torch.device,
        is_pretrain: bool = False,
        kl_weight: float = 1.0,
        dis_weight: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        CADT 训练步骤

        Args:
            source_batch: 源域批次数据
            target_batch: 目标域批次数据
            device: 计算设备
            is_pretrain: 是否为预训练阶段
            kl_weight: 原型聚类损失权重
            dis_weight: 对抗损失权重

        Returns:
            损失字典
        """
        # 提取源域数据
        src_features = source_batch['features'].to(device)
        src_lengths = source_batch['length'].to(device)
        src_labels = source_batch['label'].to(device)

        # 提取目标域数据
        tgt_features = target_batch['features'].to(device)
        tgt_lengths = target_batch['length'].to(device)

        batch_size_src = src_features.size(0)
        batch_size_tgt = tgt_features.size(0)

        # 获取特征
        src_pred, src_repr = self.encoder.forward_features(src_features, src_lengths)
        _, tgt_repr = self.encoder.forward_features(tgt_features, tgt_lengths)

        # 预训练阶段：分类损失 + 域分类损失
        if is_pretrain:
            ce_loss = self.ce_loss(src_pred, src_labels)

            # 域分类损失（使用 discriminator2）
            src_label = torch.zeros(batch_size_src, 1, device=device)
            tgt_label = torch.ones(batch_size_tgt, 1, device=device)

            domain_loss = self.bce_loss(self.discriminator2(src_repr), src_label) + \
                          self.bce_loss(self.discriminator2(tgt_repr), tgt_label)

            total_loss = ce_loss + domain_loss * dis_weight

            return {
                'total_loss': total_loss,
                'ce_loss': ce_loss,
                'domain_loss': domain_loss,
                'kl_loss': torch.tensor(0.0, device=device),
                'dis_loss': torch.tensor(0.0, device=device),
                'need_discriminator_step': False,
            }

        # 正式训练阶段：完整 CADT 损失
        # 分类损失
        ce_loss = self.ce_loss(src_pred, src_labels)

        # 原型聚类损失
        py = self.c[src_labels].detach()
        kl_loss = self.mse_loss(src_repr, py)

        # 计算到原型的距离
        src_dist = self.compute_distance_to_centers(src_repr)
        tgt_dist = self.compute_distance_to_centers(tgt_repr)

        # 对抗损失（让编码器生成让辨别器无法区分的特征）
        src_fake_label = torch.ones(batch_size_src, 1, device=device)
        tgt_real_label = torch.zeros(batch_size_tgt, 1, device=device)

        dis_loss_adv = self.bce_loss(
            self.discriminator(torch.cat([src_repr, src_dist], dim=1)), src_fake_label
        ) + self.bce_loss(
            self.discriminator(torch.cat([tgt_repr, tgt_dist], dim=1)), tgt_real_label
        )

        # 域分类损失
        src_label = torch.zeros(batch_size_src, 1, device=device)
        tgt_label = torch.ones(batch_size_tgt, 1, device=device)

        domain_loss = self.bce_loss(self.discriminator2(src_repr), src_label) + \
                      self.bce_loss(self.discriminator2(tgt_repr), tgt_label)

        total_loss = ce_loss + kl_loss * kl_weight + dis_loss_adv * dis_weight + domain_loss

        # 辨别器损失（用于单独更新辨别器）
        src_real_label = torch.zeros(batch_size_src, 1, device=device)
        tgt_fake_label = torch.ones(batch_size_tgt, 1, device=device)

        dis_loss_real = self.bce_loss(
            self.discriminator(torch.cat([src_repr.detach(), src_dist.detach()], dim=1)), src_real_label
        ) + self.bce_loss(
            self.discriminator(torch.cat([tgt_repr.detach(), tgt_dist.detach()], dim=1)), tgt_fake_label
        )

        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'kl_loss': kl_loss,
            'dis_loss': dis_loss_adv,
            'domain_loss': domain_loss,
            'need_discriminator_step': True,
            'dis_loss_for_discriminator': dis_loss_real,
        }
