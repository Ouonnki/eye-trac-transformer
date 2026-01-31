# 验证损失上升诊断报告

生成时间: 2026-01-30T16:56:36.008236

## 1. 问题概述

### 观察到的现象

| 指标 | 现象 |
|------|------|
| 训练 Loss | 上升 21-28% |
| 训练准确率 | 保持在 71% 左右 |
| 预测分布 | 极度偏向类别1 |
| Test2/Test3 准确率 | 12-25% (极低) |

## 2. 根本原因分析

### 2.1 极端类别偏向问题 ⚠️

**真实类别分布 vs 预测类别分布:**

| 类别 | 真实样本数 | 预测样本数 | 偏差 |
|------|-----------|-----------|------|
| 1 | 25 | 155 | +130 |
| 2 | 111 | 2 | -109 |
| 3 | 21 | 0 | -21 |

**各类别准确率:**

| 类别 | 准确率 | 样本数 |
|------|--------|--------|
| 1 | 100.0% (✅) | 25 |
| 2 | 0.9% (❌) | 111 |
| 3 | 0.0% (❌) | 21 |

**混淆矩阵** (行=真实, 列=预测):

```
         预测:1   预测:2   预测:3
真实:1     25       0       0
真实:2    110       1       0
真实:3     20       1       0
```

**类别不平衡比例: 5.3:1**

### 2.2 Loss 上升但准确率稳定的解释

这是**类别不平衡导致的典型现象**：

1. **模型学会预测少数类**：
   - 训练数据中类别1可能更容易预测（Loss 更低）
   - 模型倾向于将多数类（类别2）也预测为类别1

2. **Loss 持续累积**：
   - 每次将类别2错误预测为类别1时，产生高Loss
   - 类别2样本最多（111个），因此累积Loss很高

3. **准确率看似稳定**：
   - 类别1的预测是准确的（25/25 = 100%）
   - 但类别2和类别3几乎全部预测错误

### 2.3 跨任务泛化失败

**观察:**
- Test1 (新人旧题): 66-75% 准确率 - 尚可
- Test2 (旧人新题): 12-15% 准确率 - **极差**
- Test3 (新人新题): 19-25% 准确率 - **极差**

**原因分析:**
1. `use_task_embedding: false` - 模型无法区分不同任务
2. 模型可能记忆了训练集中特定任务的样本模式
3. 当遇到新任务时，模型退化为预测多数类

## 3. 配置问题分析

### 3.1 当前的正则化配置

| 配置项 | 当前值 | 评估 |
|--------|--------|------|
| dropout | 0.1 | ⚠️ 偏低，建议增加到 0.2-0.3 |
| weight_decay | 1e-4 | ⚠️ 偏低，建议增加到 1e-3 |
| learning_rate | 1e-4 | ✅ 适中 |
| patience | 100 | ⚠️ 过高，可能导致过拟合 |
| use_task_embedding | false | ❌ 应该启用 |
| label_smoothing | 无 | ⚠️ 建议添加 |

### 3.2 类别权重计算

**当前计算方式** (`src/models/dl_trainer.py:183-201`):
```python
class_weights = 1.0 / (class_counts + 1e-6)
class_weights = class_weights / class_weights.mean()  # 归一化
```

**问题**：对于极端不平衡的情况，这种权重计算方式可能不够强。

## 4. 改进建议

### 4.1 高优先级（立即实施）

#### 1. 启用任务嵌入
```json
{
  "model": {
    "use_task_embedding": true
  }
}
```

**预期效果**: 模型能够区分不同任务，提升 Test2/Test3 性能

#### 2. 调整类别权重策略
```python
# 方案1: 使用平方根缩放
class_weights = 1.0 / np.sqrt(class_counts)
class_weights = class_weights / class_weights.mean()

# 方案2: 使用中位数缩放
median_count = np.median(class_counts)
class_weights = median_count / class_counts
```

#### 3. 降低早停 Patience
```json
{
  "training": {
    "patience": 20
  }
}
```

### 4.2 中优先级（如果高优先级改进无效）

#### 1. 增加 Dropout
```json
{
  "model": {
    "dropout": 0.3
  }
}
```

#### 2. 增加 Weight Decay
```json
{
  "training": {
    "weight_decay": 0.001
  }
}
```

#### 3. 使用 Focal Loss
Focal Loss 对难分类样本给予更高权重：
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()
```

### 4.3 低优先级（长期改进）

1. **数据增强**: 添加眼动轨迹特定的增强方法
2. **模型集成**: 使用多个模型的集成预测
3. **测试时增强**: 提高预测稳定性
## 5. 验证计划

| 实验 | 配置变化 | 预期结果 |
|------|---------|---------|
| Baseline | 当前配置 | 重现问题 |
| Exp1 | +use_task_embedding | Test2/Test3 准确率提升 |
| Exp2 | +调整类别权重 | 类别2准确率提升 |
| Exp3 | +dropout=0.3 | 过拟合缓解 |
| Exp4 | +weight_decay=1e-3 | 正则化增强 |
| Exp5 | +patience=20 | 更早停止 |
| Exp6 | +focal_loss | 整体性能提升 |
## 6. 结论

### 问题根因

**验证损失上升的根本原因是严重的类别不平衡问题。**

具体表现为：
1. 模型学会预测少数类（类别1）来降低Loss
2. 多数类（类别2）被持续错误预测，累积高Loss
3. 任务嵌入未启用，导致跨任务泛化失败

### 优先行动

1. **立即**: 启用任务嵌入
2. **立即**: 调整类别权重计算策略
3. **短期**: 降低 patience，防止过拟合
4. **中期**: 如果问题持续，考虑使用 Focal Loss

