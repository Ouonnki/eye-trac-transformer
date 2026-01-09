# -*- coding: utf-8 -*-
"""
域迁移损失函数模块

提供 CADT 域迁移方法的损失函数。
参考：D:\Ouonnki\cadt\cadt\model\CADT.py
"""

import torch
import torch.nn as nn


class CenterAlignmentLoss(nn.Module):
    """
    中心对齐损失

    拉近源域样本特征到对应类中心的距离。
    对应 CADT 的 kl_loss (self.ae_loss(zsc, py))。

    用途：
    - 使同类样本聚集到类中心附近
    - 增强类内紧凑性
    """

    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction='mean')

    def forward(
        self,
        features: torch.Tensor,
        centers: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算中心对齐损失

        Args:
            features: (batch, feature_dim) 样本特征
            centers: (batch, feature_dim) 对应的类中心（已detach）

        Returns:
            标量损失值
        """
        return self.mse_loss(features, centers.detach())


class CenterDiversityLoss(nn.Module):
    """
    中心多样性损失

    保持不同类中心之间的距离，防止类中心塌缩。
    对应 CADT 的 dloss（类中心间的距离之和）。

    用途：
    - 使不同类的中心保持分离
    - 增强类间可分性
    """

    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction='mean')

    def forward(self, centers: torch.Tensor) -> torch.Tensor:
        """
        计算中心多样性损失

        Args:
            centers: (num_classes, feature_dim) 类中心

        Returns:
            标量损失值（越大表示中心越相似，需要最小化负值或最大化正值）
        """
        num_classes = centers.size(0)
        total_loss = torch.tensor(0.0, device=centers.device)
        count = 0

        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                # 计算两个中心之间的MSE距离
                total_loss = total_loss + self.mse_loss(centers[i], centers[j])
                count += 1

        if count > 0:
            return total_loss / count
        return torch.tensor(0.0, device=centers.device)


class DomainAdversarialLoss(nn.Module):
    """
    域对抗损失

    训练判别器区分源域/目标域，
    同时通过 GRL 让编码器混淆判别器。
    对应 CADT 的 domain_class_loss。

    用途：
    - 学习域不变特征表示
    - 减小源域和目标域之间的分布差异
    """

    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCELoss(reduction='mean')

    def forward(
        self,
        source_probs: torch.Tensor,
        target_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算域对抗损失

        Args:
            source_probs: (batch_s, 1) 源域样本的判别概率（应该接近0）
            target_probs: (batch_t, 1) 目标域样本的判别概率（应该接近1）

        Returns:
            标量损失值
        """
        # 源域标签为0，目标域标签为1
        source_labels = torch.zeros_like(source_probs)
        target_labels = torch.ones_like(target_probs)

        source_loss = self.bce_loss(source_probs, source_labels)
        target_loss = self.bce_loss(target_probs, target_labels)

        return source_loss + target_loss


class CADTLoss(nn.Module):
    """
    CADT 综合损失

    组合所有 CADT 相关的损失函数。

    Loss = L_ce + λ1 * L_center + λ2 * (L_domain + L_dist_domain) - λ3 * L_diversity
    """

    def __init__(
        self,
        center_weight: float = 10.0,
        adversarial_weight: float = 1.0,
        diversity_weight: float = 0.1,
    ):
        """
        初始化

        Args:
            center_weight: 中心对齐损失权重 (λ1)
            adversarial_weight: 域对抗损失权重 (λ2)
            diversity_weight: 中心多样性损失权重 (λ3)
        """
        super().__init__()

        self.center_weight = center_weight
        self.adversarial_weight = adversarial_weight
        self.diversity_weight = diversity_weight

        self.ce_loss = nn.CrossEntropyLoss()
        self.center_loss = CenterAlignmentLoss()
        self.adversarial_loss = DomainAdversarialLoss()
        self.diversity_loss = CenterDiversityLoss()

    def forward(
        self,
        # 分类损失
        logits: torch.Tensor,
        labels: torch.Tensor,
        # 中心对齐损失
        features: torch.Tensor,
        centers: torch.Tensor,
        # 域对抗损失
        source_domain_probs: torch.Tensor,
        target_domain_probs: torch.Tensor,
        # 距离感知域对抗损失
        source_dist_probs: torch.Tensor,
        target_dist_probs: torch.Tensor,
        # 中心多样性损失
        all_centers: torch.Tensor,
    ) -> dict:
        """
        计算综合损失

        Args:
            logits: (batch, num_classes) 分类logits
            labels: (batch,) 真实类别
            features: (batch, feature_dim) 样本特征
            centers: (batch, feature_dim) 对应的类中心
            source_domain_probs: (batch_s, 1) 源域判别概率
            target_domain_probs: (batch_t, 1) 目标域判别概率
            source_dist_probs: (batch_s, 1) 源域距离感知判别概率
            target_dist_probs: (batch_t, 1) 目标域距离感知判别概率
            all_centers: (num_classes, feature_dim) 所有类中心

        Returns:
            字典包含：
            - total: 总损失
            - ce: 分类损失
            - center: 中心对齐损失
            - domain: 域对抗损失
            - dist_domain: 距离感知域对抗损失
            - diversity: 中心多样性损失
        """
        # 分类损失
        ce = self.ce_loss(logits, labels)

        # 中心对齐损失
        center = self.center_loss(features, centers)

        # 域对抗损失
        domain = self.adversarial_loss(source_domain_probs, target_domain_probs)

        # 距离感知域对抗损失
        dist_domain = self.adversarial_loss(source_dist_probs, target_dist_probs)

        # 中心多样性损失（希望最大化，所以取负）
        diversity = self.diversity_loss(all_centers)

        # 总损失
        total = (
            ce
            + self.center_weight * center
            + self.adversarial_weight * (domain + dist_domain)
            - self.diversity_weight * diversity
        )

        return {
            'total': total,
            'ce': ce,
            'center': center,
            'domain': domain,
            'dist_domain': dist_domain,
            'diversity': diversity,
        }
