# -*- coding: utf-8 -*-
"""
片段级CADT模型

参考外部CADT项目，将域适应能力集成到片段级Transformer中。

核心设计：
1. 双流编码器：深层提取不变特征，浅层提取特有特征
2. 类中心对齐：KL loss让特征对齐到类中心
3. 类别超球体损失：让不同类中心尽量分开
4. 双判别器：一个带距离，一个用特有特征
5. GRL：使用梯度反转层实现对抗训练（DANN模式）
6. Specific噪声：防止特有流退化
7. 任务条件解耦：特有流不接收任务条件
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple


class GradientReversalFunction(torch.autograd.Function):
    """
    梯度反转层 (GRL)

    前向传播：直接传递输入
    反向传播：梯度乘以 -lambda
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    """梯度反转层模块"""

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)


class Classifier(nn.Module):
    """分类器"""

    def __init__(self, input_size: int, n_class: int):
        super().__init__()
        self.linear = nn.Linear(input_size, n_class)

    def forward(self, inputs):
        return self.linear(inputs)


class Discriminator(nn.Module):
    """域判别器"""

    def __init__(self, input_size: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(16, 16),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        return self.model(inputs.view(inputs.size(0), -1))


class DualStreamTransformerEncoder(nn.Module):
    """
    双流Transformer编码器

    设计思路：
    - invariant_stream: 深层Transformer → 领域不变特征 (f1)
    - specific_stream: 浅层Transformer → 领域特有特征 (f2)

    关键修复：
    - invariant_stream接收task_conditions（任务相关信息）
    - specific_stream不接收task_conditions（强制解耦）
    - specific_stream输入添加高斯噪声（防止退化）
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,  # invariant使用的层数
        dim_feedforward: int,
        dropout: float,
        max_seq_len: int,
        use_gradient_checkpointing: bool = False,
        use_task_embedding: bool = False,
        task_embedding_dim: int = 16,
        specific_noise_std: float = 0.1,  # 特有流输入噪声标准差
    ):
        super().__init__()

        self.d_model = d_model
        self.specific_noise_std = specific_noise_std

        # 导入GazeTransformerEncoder
        from src.models.encoders import GazeTransformerEncoder

        # 不变特征流：深层Transformer
        self.invariant_stream = GazeTransformerEncoder(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,  # 如4层
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=max_seq_len,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_task_embedding=use_task_embedding,
            task_embedding_dim=task_embedding_dim,
        )

        # 特有特征流：浅层Transformer (1层)
        self.specific_stream = GazeTransformerEncoder(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=1,  # 固定1层
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=max_seq_len,
            use_gradient_checkpointing=False,  # 浅层不需要
            use_task_embedding=False,  # 特有流不使用任务嵌入
            task_embedding_dim=task_embedding_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        task_conditions: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: (batch, seq_len, input_dim) 输入序列
            mask: (batch, seq_len) 有效位置掩码
            task_conditions: 任务条件字典

        Returns:
            f_invariant: (batch, d_model) 领域不变特征
            f_specific: (batch, d_model) 领域特有特征
        """
        # 不变特征流：使用任务条件
        f_invariant, _ = self.invariant_stream(x, mask, task_conditions)

        # 特有特征流：添加噪声，不使用任务条件
        if self.training and self.specific_noise_std > 0:
            x_noisy = x + torch.randn_like(x) * self.specific_noise_std
        else:
            x_noisy = x
        f_specific, _ = self.specific_stream(x_noisy, mask, None)  # 不传task_conditions

        return f_invariant, f_specific


class SegmentCADTModel(nn.Module):
    """
    片段级CADT模型

    双流设计：
    - invariant_stream (深层): 提取领域不变特征 f1
    - specific_stream (浅层): 提取领域特有特征 f2

    特征使用（按原CADT）：
    - f1 → 分类器
    - f1 + 距离 → 域判别器 (通过GRL)
    - f2 → 域判别器2 (通过GRL)

    关键修复：
    - 使用GRL替代交替优化（DANN模式）
    - 距离特征除以sqrt(d_model)而非d_model
    - 添加类别超球体损失
    """

    def __init__(
        self,
        dual_stream_encoder,  # 复用的双流编码器
        n_class: int = 3,
        d_model: int = 64,
        device: torch.device = torch.device('cuda'),
        grl_lambda: float = 1.0,  # GRL强度
    ):
        super().__init__()

        self.n_class = n_class
        self.d_model = d_model
        self.device = device
        self.grl_lambda = grl_lambda

        # 复用双流编码器
        self.dual_stream = dual_stream_encoder

        # GRL层
        self.grl = GradientReversalLayer(lambda_=grl_lambda)

        # CADT组件
        self.classifier = Classifier(d_model, n_class)
        self.discriminator = Discriminator(d_model + 1)  # f1 + 距离
        self.discriminator2 = Discriminator(d_model)  # f2

        # 类中心参数 [n_class, d_model]
        self.c = nn.Parameter(torch.zeros(n_class, d_model))

        # 损失函数
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()

        self.to(device)

    def encode(
        self,
        features: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        task_conditions: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        双流编码

        Args:
            features: (batch, seq_len, input_dim) 输入序列
            lengths: (batch,) 实际序列长度（未使用）
            task_conditions: 任务条件字典

        Returns:
            f_invariant: (batch, d_model) 领域不变特征
            f_specific: (batch, d_model) 领域特有特征
        """
        return self.dual_stream(features, None, task_conditions)

    def compute_distance_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        计算到各类中心的最小距离

        修复：除以sqrt(d_model)而非d_model，避免梯度过小

        Args:
            features: (batch, d_model) 特征

        Returns:
            min_distance: (batch, 1) 最小距离
        """
        distances = []
        for i in range(self.n_class):
            dist = torch.norm(features - self.c[i].detach(), dim=1, keepdim=True)
            distances.append(dist)
        distances = torch.cat(distances, dim=1)
        # 修复：除以sqrt(d_model)而非d_model
        min_distance = torch.min(distances, dim=1, keepdim=True).values / (self.d_model ** 0.5)
        return min_distance

    def compute_hyper_sphere_loss(self) -> torch.Tensor:
        """
        计算类别超球体损失

        让不同类中心之间尽量分开

        Returns:
            loss: 标量损失
        """
        loss = torch.tensor(0.0).to(self.device)
        for i in range(self.n_class):
            for j in range(i + 1, self.n_class):
                loss = loss - self.mse_loss(self.c[i], self.c[j])
        return loss

    def init_centers(self, data_loader):
        """
        初始化类中心（基于源域数据的不变特征）

        Args:
            data_loader: 源域数据加载器
        """
        n_samples = torch.zeros(self.n_class).to(self.device)
        c = torch.zeros(self.n_class, self.d_model).to(self.device)

        with torch.no_grad():
            for batch in data_loader:
                features = batch['features'].to(self.device)
                task_conditions = batch.get('task_conditions')

                # 将 task_conditions 中的张量移动到 GPU
                if task_conditions is not None:
                    task_conditions = {
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in task_conditions.items()
                    }

                # 双流编码，使用不变特征
                f_invariant, _ = self.encode(features, None, task_conditions)
                labels = batch['labels'].to(self.device)

                # 累加到对应类中心
                for i in range(self.n_class):
                    mask = (labels == i)
                    if mask.sum() > 0:
                        z_class = f_invariant[mask]
                        c[i] += z_class.sum(dim=0)
                        n_samples[i] += mask.sum()

        c = c / n_samples.unsqueeze(1)
        self.c.data.copy_(c.data)
        self.c.requires_grad = False

    def reset_full(self):
        """完全重置模型（与原CADT一致）"""
        d_model = self.d_model
        n_class = self.n_class

        self.classifier = Classifier(d_model, n_class).to(self.device)
        self.discriminator = Discriminator(d_model + 1).to(self.device)
        self.discriminator2 = Discriminator(d_model).to(self.device)

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
        labels: torch.Tensor,
        domain_labels: torch.Tensor,
        change_center: bool = False,
        kl_w: float = 0.0,
        dis_w: float = 1.0,  # 域判别损失权重
        num_source: int = None,  # 源域样本数量
    ) -> Dict[str, torch.Tensor]:
        """
        CADT前向传播（DANN模式：单步更新）

        使用GRL实现对抗训练，无需交替优化

        Args:
            features: (batch, seq_len, input_dim) 输入序列
            lengths: (batch,) 实际序列长度（未使用）
            labels: (batch,) 类别标签
            domain_labels: (batch, 1) 域标签 (0=源域, 1=目标域)
            change_center: 预训练阶段
            kl_w: KL损失权重
            dis_w: 域判别损失权重
            num_source: 源域样本数量（用于切片，仅在源域上计算分类损失）

        Returns:
            losses: 包含各损失项的字典
        """
        batch_size = features.size(0)
        # 默认全部为源域（兼容没有域适应的场景）
        if num_source is None:
            num_source = batch_size

        # 1. 双流编码
        f_invariant, f_specific = self.encode(features, lengths)

        # 2. 分类损失（仅在源域数据上计算）
        source_logits = self.classifier(f_invariant[:num_source])
        source_labels = labels[:num_source]
        ce_loss = self.ce_loss(source_logits, source_labels)

        # 3. KL损失：不变特征对齐到类中心（仅在源域数据上计算）
        source_f_invariant = f_invariant[:num_source]
        py = self.c[source_labels].detach()
        kl_loss = self.mse_loss(source_f_invariant, py)

        # 4. 类别超球体损失
        hyper_sphere_loss = self.compute_hyper_sphere_loss()

        # 5. 距离特征
        distance_features = self.compute_distance_features(f_invariant)

        # 6. 域判别器（不变特征 + 距离，通过GRL）
        f1_with_distance = torch.cat([f_invariant, distance_features], dim=1)
        f1_with_distance_grl = self.grl(f1_with_distance)
        domain_pred = self.discriminator(f1_with_distance_grl)

        # 7. 域判别器2（特有特征，通过GRL）
        f_specific_grl = self.grl(f_specific)
        domain_pred2 = self.discriminator2(f_specific_grl)

        # 8. 域分类损失（GRL自动反转梯度）
        domain_class_loss = self.bce_loss(domain_pred2, domain_labels)
        domain_class_loss2 = self.bce_loss(domain_pred, domain_labels)

        # 9. 总损失
        if change_center:
            # 预训练阶段：分类 + 域判别
            total_loss = ce_loss + (domain_class_loss2 + domain_class_loss) * dis_w
        else:
            # 域适应阶段：分类 + KL + 超球体 + 域判别
            total_loss = (
                ce_loss * 1.0 +
                kl_loss * kl_w +
                hyper_sphere_loss * 0.1 +
                domain_class_loss * dis_w +
                domain_class_loss2 * dis_w
            )

        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'kl_loss': kl_loss,
            'hyper_sphere_loss': hyper_sphere_loss,
            'domain_class_loss': domain_class_loss,
            'domain_class_loss2': domain_class_loss2,
        }

    def compute_loss(self, features: torch.Tensor, labels: torch.Tensor) -> tuple:
        """
        计算分类损失和准确率（评估时用）

        Args:
            features: (batch, seq_len, input_dim) 输入序列
            labels: (batch,) 类别标签

        Returns:
            (loss, correct, total)
        """
        with torch.no_grad():
            f_invariant, _ = self.encode(features)
            pred_logits = self.classifier(f_invariant)
            pred_labels = pred_logits.argmax(dim=1)
            loss = self.ce_loss(pred_logits, labels)
            correct = (pred_labels == labels).sum().item()
            total = labels.size(0)
        return loss.item(), correct, total

    def parameters(self, recurse=True):
        """获取模型参数"""
        params = []
        params.extend(list(self.dual_stream.parameters()))
        params.extend(list(self.classifier.parameters()))
        params.extend(list(self.discriminator.parameters()))
        params.extend(list(self.discriminator2.parameters()))
        return iter(params)
