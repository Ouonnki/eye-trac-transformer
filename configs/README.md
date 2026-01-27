# 训练配置说明

本目录包含模型训练的 JSON 配置文件。

## 快速开始

```bash
# 使用默认配置
python experiments/dl_transformer_experiment.py

# 使用自定义配置
CONFIG=configs/my_config.json python experiments/dl_transformer_experiment.py
```

---

## 配置文件结构

配置文件分为 7 个组：

```json
{
  "experiment": {...},  // 实验配置
  "model": {...},      // 模型架构
  "training": {...},    // 训练超参数
  "task": {...},       // 任务类型
  "cadt": {...},       // CADT域适应（可选）
  "device": {...},     // 计算设备
  "output": {...}      // 输出设置
}
```

---

## 配置项详解

### experiment - 实验配置

控制实验运行模式和数据集划分策略。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `train_subjects` | int | 100 | 训练集被试数量（从总数据中随机选择） |
| `train_tasks` | int | 20 | 训练集任务数量（按 task_id 排序后取前 N 个） |
| `n_repeats` | int | 1 | 重复实验次数（多次随机划分取平均结果） |
| `random_seed` | int | 42 | 随机种子（确保实验可复现） |
| `experiment_name` | str | "" | 实验名称（空则自动生成时间戳） |
| `output_dir` | str | "outputs/dl_models" | 输出目录路径 |

**典型配置：**
```json
{
  "experiment": {
    "train_subjects": 100,
    "train_tasks": 20,
    "n_repeats": 1,
    "random_seed": 42,
    "experiment_name": "baseline_20260127",
    "output_dir": "outputs/dl_models"
  }
}
```

---

### model - 模型架构配置

控制层级 Transformer 网络结构。

| 参数 | 类型 | 默认值 | 可选值 | 说明 |
|------|------|--------|--------|------|
| `input_dim` | int | 7 | 7 | 输入特征维度（固定为7） |
| `segment_d_model` | int | 64 | 32/64/128/256 | 片段级 Transformer 隐藏维度 |
| `segment_nhead` | int | 4 | 能整除 d_model | 片段级注意力头数 |
| `segment_num_layers` | int | 4 | 2/4/6/8 | 片段级 Transformer 层数 |
| `task_d_model` | int | 128 | 64/128/256 | 任务级 Transformer 隐藏维度 |
| `task_nhead` | int | 4 | 能整除 d_model | 任务级注意力头数 |
| `task_num_layers` | int | 2 | 1/2/4 | 任务级 Transformer 层数 |
| `attention_dim` | int | 32 | 16/32/64 | 注意力池化层维度 |
| `dropout` | float | 0.1 | 0.0-0.5 | Dropout 正则化比例 |

**模型规模预设：**

| 规模 | segment_d_model | task_d_model | segment_num_layers | batch_size |
|------|-----------------|---------------|-------------------|------------|
| light | 64 | 128 | 2 | 4-8 |
| medium | 96 | 192 | 4 | 4-8 |
| full | 128 | 256 | 6 | 8-16 |

**典型配置：**
```json
{
  "model": {
    "input_dim": 7,
    "segment_d_model": 64,
    "segment_nhead": 4,
    "segment_num_layers": 4,
    "task_d_model": 128,
    "task_nhead": 4,
    "task_num_layers": 2,
    "attention_dim": 32,
    "dropout": 0.1
  }
}
```

---

### training - 训练超参数配置

控制优化器、学习率和训练流程。

| 参数 | 类型 | 默认值 | 可选值 | 说明 |
|------|------|--------|--------|------|
| `batch_size` | int | 8 | 2/4/8/16 | 批次大小（显存不足时减小） |
| `learning_rate` | float | 0.0001 | 1e-5 ~ 1e-3 | 初始学习率 |
| `weight_decay` | float | 0.0001 | 0 ~ 0.1 | L2 正则化系数 |
| `warmup_epochs` | int | 5 | 0 ~ 10 | 学习率预热轮数 |
| `epochs` | int | 400 | 100 ~ 500 | 最大训练轮数 |
| `patience` | int | 100 | 50 ~ 150 | 早停耐心值 |
| `grad_clip` | float | 1.0 | 0.5 ~ 5.0 | 梯度裁剪阈值 |

**典型配置：**
```json
{
  "training": {
    "batch_size": 8,
    "learning_rate": 0.0001,
    "weight_decay": 0.0001,
    "warmup_epochs": 5,
    "epochs": 400,
    "patience": 100,
    "grad_clip": 1.0
  }
}
```

---

### task - 任务类型配置

控制预测任务的类型。

| 参数 | 类型 | 默认值 | 可选值 | 说明 |
|------|------|--------|--------|------|
| `type` | str | "classification" | "classification" / "regression" | 任务类型 |
| `num_classes` | int | 3 | 正整数 | 分类任务类别数 |
| `use_class_weights` | bool | true | true / false | 是否使用类别权重 |

**典型配置：**
```json
{
  "task": {
    "type": "classification",
    "num_classes": 3,
    "use_class_weights": true
  }
}
```

---

### cadt - CADT 域适应配置

仅在使用 CADT（域适应）模型时需要配置。

| 参数 | 类型 | 默认值 | 可选值 | 说明 |
|------|------|--------|--------|------|
| `target_domain` | str | "test1" | "test1" / "test2" / "test3" | 目标域选择 |
| `cadt_kl_weight` | float | 1.0 | 0.1 ~ 10.0 | 原型聚类损失权重 |
| `cadt_dis_weight` | float | 1.0 | 0.1 ~ 10.0 | 域对抗损失权重 |
| `pre_train_epochs` | int | 120 | 50 ~ 200 | 预训练阶段 epoch 数 |
| `pre_train_dis_weight` | float | 0.1 | 0.0 ~ 1.0 | 预训练阶段域对抗权重 |
| `reset_mode` | str | "optimizer" | "none" / "optimizer" / "full" | 阶段切换重置模式 |
| `use_augmentation` | bool | true | true / false | 是否使用数据增强 |
| `encoder_lr` | float | 0.0001 | 1e-5 ~ 1e-3 | 编码器学习率 |
| `classifier_lr` | float | 0.001 | 1e-4 ~ 1e-2 | 分类器学习率 |
| `discriminator_lr` | float | 0.001 | 1e-4 ~ 1e-2 | 域辨别器学习率 |

**目标域说明：**
- `test1`: 新人旧题（跨被试泛化）
- `test2`: 旧人新题（跨任务泛化）
- `test3`: 新人新题（双重泛化）

**典型配置：**
```json
{
  "cadt": {
    "target_domain": "test1",
    "cadt_kl_weight": 1.0,
    "cadt_dis_weight": 1.0,
    "pre_train_epochs": 120,
    "pre_train_dis_weight": 0.1,
    "reset_mode": "optimizer",
    "use_augmentation": true,
    "encoder_lr": 0.0001,
    "classifier_lr": 0.001,
    "discriminator_lr": 0.001
  }
}
```

---

### device - 计算设备配置

控制 GPU 使用和性能优化。

| 参数 | 类型 | 默认值 | 可选值 | 说明 |
|------|------|--------|--------|------|
| `device` | str | "cuda" | "cuda" / "cpu" / "mps" | 计算设备 |
| `use_multi_gpu` | bool | true | true / false | 是否使用多 GPU |
| `use_amp` | bool | true | true / false | 是否使用混合精度训练（FP16） |
| `use_gradient_checkpointing` | bool | false | true / false | 是否使用梯度检查点 |
| `num_workers` | int | 8 | 0 ~ 16 | 数据加载 worker 进程数 |
| `pin_memory` | bool | true | true / false | 是否固定内存加速传输 |

**典型配置：**
```json
{
  "device": {
    "device": "cuda",
    "use_multi_gpu": true,
    "use_amp": true,
    "use_gradient_checkpointing": false,
    "num_workers": 8,
    "pin_memory": true
  }
}
```

---

### output - 输出配置

控制模型保存和日志输出。

| 参数 | 类型 | 默认值 | 可选值 | 说明 |
|------|------|--------|--------|------|
| `save_best` | bool | true | true / false | 是否保存最佳模型 |
| `save_figures` | bool | true | true / false | 是否保存训练曲线图 |
| `figure_dpi` | int | 150 | 100 ~ 300 | 保存图片的分辨率 |
| `summary_interval` | int | 10 | 1 ~ 50 | 训练日志输出间隔 |

**典型配置：**
```json
{
  "output": {
    "save_best": true,
    "save_figures": true,
    "figure_dpi": 150,
    "summary_interval": 10
  }
}
```

---

## 完整配置示例

### 分类任务（默认配置）

参见 `default.json`

### CADT 域适应配置

参见 `cadt_test1.json`（如存在）

---

## 配置调优建议

### 显存不足时

1. 减小 `batch_size`（8 → 4 → 2）
2. 启用 `use_amp: true`
3. 启用 `use_gradient_checkpointing: true`
4. 减小模型规模（`segment_d_model`、`task_d_model`）

### 过拟合时

1. 增加 `dropout`（0.1 → 0.2 → 0.3）
2. 增加 `weight_decay`（0.0001 → 0.001）
3. 减少 `segment_num_layers` 或 `task_num_layers`

### 欠拟合时

1. 减少 `dropout`
2. 增加 `epochs` 和 `patience`
3. 增加模型规模（更多层或更大维度）

### 类别不平衡时

设置 `use_class_weights: true`（默认已开启）
