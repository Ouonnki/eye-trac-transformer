# 眼动追踪注意力预测系统 - 项目分析报告

生成时间：2026-01-27
更新时间：2026-01-27（移除XGBoost和传统特征工程）

---

## 目录

- [一、项目概览](#一项目概览)
- [二、目录结构详解](#二目录结构详解)
- [三、核心文件功能详解](#三核心文件功能详解)
- [四、数据流和核心流程](#四数据流和核心流程)
- [五、关键技术特点](#五关键技术特点)
- [六、项目入口和使用方法](#六项目入口和使用方法)
- [七、输出文件结构](#七输出文件结构)
- [八、模块依赖关系图](#八模块依赖关系图)
- [九、文件索引](#九文件索引)

---

## 一、项目概览

### 1.1 项目基本信息

| 属性 | 描述 |
|------|------|
| **项目名称** | eye-trac-transformer（眼动追踪注意力预测系统） |
| **项目目标** | 基于舒尔特方格任务的眼动轨迹数据，使用层级Transformer深度学习模型预测个体的注意力水平 |
| **技术栈** | Python, PyTorch, pandas, numpy, scikit-learn |
| **代码规模** | 约 8,200 行 Python 代码（精简后） |

### 1.2 核心依赖

```
# 数据处理
pandas>=2.0.0
numpy>=1.24.0
openpyxl>=3.1.0

# 机器学习（通用工具）
scikit-learn>=1.3.0

# 深度学习
torch>=2.0.0

# 可视化
matplotlib>=3.7.0

# 工具
tqdm>=4.65.0
```

### 1.3 项目架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                         眼动追踪注意力预测系统                    │
├─────────────────────────────────────────────────────────────────┤
│  输入层: Excel眼动数据 (点击事件 + 连续轨迹)                     │
├─────────────────────────────────────────────────────────────────┤
│  数据处理: loader → preprocessor → segmenter                    │
├─────────────────────────────────────────────────────────────────┤
│  特征提取: 7维原始特征 (x, y, dt, velocity, acceleration, ...)  │
├─────────────────────────────────────────────────────────────────┤
│  模型层:   HierarchicalTransformer / DomainAdaptiveNetwork      │
├─────────────────────────────────────────────────────────────────┤
│  输出层:   注意力分数/类别 + 域迁移泛化                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 二、目录结构详解

```
/Users/ouonnki/Projects/py_projects/eye-trac-transformer/
├── .git/                          # Git 版本控制目录
├── .gitignore                     # Git 忽略规则
├── README.md                      # 项目说明文档
├── config.py                      # 全局配置文件 (约70行)
├── main.py                        # 主入口（占位符，指向实验脚本）
├── requirements.txt               # 项目依赖列表 (20行)
├── PROJECT_ANALYSIS_REPORT.md     # 项目分析报告（本文件）
├── experiments/                   # 实验脚本目录
│   ├── dl_transformer_experiment.py       # Transformer 深度学习实验 (774行)
│   ├── domain_adaptation_experiment.py    # CADT 域迁移实验 (777行)
│   └── experiment_config.py              # 实验配置管理 (203行)
├── scripts/                       # 工具脚本目录
│   ├── preprocess_data.py        # 数据预处理脚本 (279行)
│   └── diagnose_clicks.py        # 点击数据诊断工具 (337行)
└── src/                          # 核心源代码目录
    ├── __init__.py
    ├── data/                     # 数据处理模块
    │   ├── __init__.py
    │   ├── loader.py             # 数据加载器 (289行)
    │   ├── preprocessor.py       # 眼动数据预处理 (415行)
    │   ├── schemas.py            # 数据结构定义 (260行)
    │   └── split_strategy.py     # 数据划分策略 (359行)
    ├── segmentation/             # 眼动分割模块
    │   ├── __init__.py
    │   └── event_segmenter.py    # 事件分割器 (搜索片段划分)
    ├── models/                   # 模型模块
    │   ├── __init__.py
    │   ├── attention.py          # 注意力机制 (264行)
    │   ├── dl_dataset.py         # 深度学习数据集 (371行)
    │   ├── dl_models.py          # Transformer 模型架构 (674行)
    │   ├── dl_trainer.py         # 深度学习训练器 (897行)
    │   ├── domain_adaptation.py  # CADT 域迁移组件 (283行)
    │   └── domain_trainer.py     # 域迁移训练器 (568行)
    └── features/                 # 特征模块（占位，说明使用7维原始特征）
        └── __init__.py
```

---

## 三、核心文件功能详解

### 3.1 根目录文件

#### [config.py](config.py) (约70行)

**功能**：全局配置管理

**主要配置项**：
- **路径配置**：数据目录、输出目录
- **屏幕参数**：1920×1080 分辨率
- **特征提取参数**：
  - 熵计算网格大小：8×8
  - 目标区域容差：50像素
- **深度学习配置**：
  - max_seq_len: 100
  - max_tasks: 30
  - max_segments: 30
  - Transformer维度配置
  - 训练参数（batch_size, learning_rate等）
- **交叉验证配置**：5折

#### [main.py](main.py) (13行)

**功能**：项目主入口（占位符）

实际运行请使用 `experiments/` 目录下的实验脚本：
- `dl_transformer_experiment.py`: Transformer模型训练
- `domain_adaptation_experiment.py`: 域迁移实验

#### [requirements.txt](requirements.txt) (20行)

**核心依赖列表**，详见 1.2 节

---

### 3.2 数据处理模块 ([src/data/](src/data/))

#### [src/data/schemas.py](src/data/schemas.py) (260行)

**功能**：核心数据结构定义

**关键数据类**：

| 数据类 | 描述 | 主要属性 |
|--------|------|----------|
| `GazePoint` | 单个眼动点 | timestamp, x, y, is_click, target_number |
| `SearchSegment` | 搜索片段 (Click(N-1)→Click(N)) | segment_id, start_number, target_number, gaze_points |
| `TaskConfig` | 任务配置 | grid_size, number_range, click_disappear, distractor_config |
| `TaskTrial` | 单个任务试次 | subject_id, task_id, config, segments, raw_gaze_points |
| `SubjectData` | 被试完整数据 | subject_id, total_score, category, trials |

#### [src/data/loader.py](src/data/loader.py) (289行)

**功能**：从 Excel 文件加载眼动数据

**数据源**：
- `个人标签.xlsx`：被试标签（总分、类别）
- `题目信息.xlsx`：任务配置（方格数量、数字范围、干扰项）
- `被试文件夹/*.xlsx`：眼动轨迹数据

**核心类 `DataLoader`**：

| 方法 | 功能 |
|------|------|
| `load_labels()` | 加载被试标签 |
| `load_tasks()` | 加载任务配置 |
| `load_gaze_trajectory()` | 加载单个轨迹文件 |
| `load_subject()` | 加载完整被试数据 |
| `_parse_grid_size()` | 智能解析方格数量格式 |

#### [src/data/preprocessor.py](src/data/preprocessor.py) (415行)

**功能**：眼动数据清洗和预处理

**核心类 `GazePreprocessor`**：

| 方法 | 功能 |
|------|------|
| `parse_timestamp()` | 解析多种时间戳格式 |
| `clean_coordinates()` | 清洗坐标值 |
| `identify_clicks()` | 识别点击事件 |
| `preprocess_gaze_data()` | 预处理连续眼动轨迹 |

#### [src/data/split_strategy.py](src/data/split_strategy.py) (359行)

**功能**：数据划分策略

**核心类**：

**1. TwoByTwoSplitter - 2×2矩阵划分**

创新的泛化性评估方案：

```
              训练任务(1-N)  |  测试任务(N+1-30)
训练被试(1-K) |    train     |     test2
              |  (基线性能)   |  (旧人新题)
--------------+-------------+-------------
测试被试(K+1) |    test1     |     test3
              |  (新人旧题)   |  (新人新题)
```

- **train**: 前100人 × 前20题（训练集）
- **test1**: 后L人 × 前20题（新人旧题，跨被试泛化）
- **test2**: 前100人 × 后10题（旧人新题，跨任务泛化）
- **test3**: 后L人 × 后10题（新人新题，双重泛化）

**2. KFoldSplitter - K折交叉验证**

传统的按被试划分的交叉验证。

---

### 3.3 分割模块 ([src/segmentation/](src/segmentation/))

#### [src/segmentation/event_segmenter.py](src/segmentation/event_segmenter.py)

**功能**：将点击序列切分为搜索片段

**核心类 `EventSegmenter`**：

**算法逻辑**：
```
1. 按时间排序点击事件
2. 为每对相邻点击创建搜索片段
3. 将眼动轨迹数据填充到对应片段中
4. 每个片段包含：起始点击 + 中间眼动 + 目标点击
```

**子类 `AdaptiveSegmenter`**：
- 从点击坐标自动学习网格布局
- 支持不完整的网格布局

---

### 3.4 模型模块 ([src/models/](src/models/))

#### [src/models/attention.py](src/models/attention.py) (264行)

**功能**：注意力机制组件

**核心类**：

**1. AttentionPooling - 注意力聚合**
```python
score = W2 * tanh(W1 * h)
alpha = softmax(score, mask)
output = sum(alpha * h)
```

**2. MultiHeadAttentionPooling - 多头注意力聚合**

**3. PositionalEncoding - 正弦位置编码**

#### [src/models/dl_models.py](src/models/dl_models.py) (674行)

**功能**：深度学习模型架构

**核心模型**：

**1. HierarchicalTransformerNetwork - 层级预测网络**

```
输入: 7维眼动特征 (x, y, dt, velocity, acceleration, direction, direction_change)
  ↓
Level 1: GazeTransformerEncoder (片段编码)
  - 片段表示 (64维)
  ↓
Level 2: TaskAggregator (AttentionPooling)
  - 任务表示
  ↓
Level 3: TaskTransformerEncoder (任务编码)
  - 任务序列 (128维)
  ↓
Level 4: SubjectAggregator (AttentionPooling)
  - 被试表示
  ↓
Level 5: PredictionHead
  - 注意力分数/类别
```

**2. DomainAdaptiveHierarchicalNetwork - 域适应网络**

继承层级网络，添加CADT域迁移组件：
- **ClassCenterBank**: 类中心库
- **DomainDiscriminator**: 标准域判别器
- **DistanceAwareDomainDiscriminator**: 距离感知域判别器

#### [src/models/domain_adaptation.py](src/models/domain_adaptation.py) (283行)

**功能**：CADT域迁移组件

**核心组件**：
- `GradientReversalFunction`: 梯度反转函数
- `ClassCenterBank`: 类中心库
- `DomainDiscriminator`: 标准域判别器
- `DistanceAwareDomainDiscriminator`: 距离感知域判别器

#### [src/models/dl_dataset.py](src/models/dl_dataset.py) (371行)

**功能**：深度学习数据集

**核心类**：

**1. SequenceFeatureExtractor**

**7维增强特征提取**：
1. x: 归一化X坐标 [0, 1]
2. y: 归一化Y坐标 [0, 1]
3. dt: 时间差（毫秒）
4. velocity: 瞬时速度
5. acceleration: 瞬时加速度
6. direction: 移动方向（归一化到[-1, 1]）
7. direction_change: 方向变化量

**2. HierarchicalGazeDataset**

数据组织：被试 → 任务 → 片段 → 眼动序列

**3. LightweightGazeDataset**（实验用）

#### [src/models/dl_trainer.py](src/models/dl_trainer.py) (897行)

**功能**：深度学习训练器

**核心类 `DeepLearningTrainer`**：

**核心功能**：
- 混合精度训练（FP16）
- 梯度检查点（节省显存）
- 学习率预热和衰减
- 早停机制
- 训练曲线可视化

#### [src/models/domain_trainer.py](src/models/domain_trainer.py) (568行)

**功能**：CADT域迁移训练器

**训练流程**：
1. **预训练阶段**：源域分类器训练
2. **域迁移阶段**：加入域对抗和类中心对齐损失

---

### 3.5 实验模块 ([experiments/](experiments/))

#### [experiments/experiment_config.py](experiments/experiment_config.py) (203行)

**功能**：统一的实验配置管理

**配置项**：
- 实验模式：quick/formal
- 任务类型：classification/regression
- 划分策略：2x2/kfold
- 域迁移配置

#### [experiments/dl_transformer_experiment.py](experiments/dl_transformer_experiment.py) (774行)

**功能**：Transformer深度学习实验

**支持的实验设计**：
1. 2×2矩阵划分
2. K-Fold交叉验证

**配置模式**：full/medium/light

#### [experiments/domain_adaptation_experiment.py](experiments/domain_adaptation_experiment.py) (777行)

**功能**：CADT域迁移实验

**CADT域迁移流程**：
1. 预训练阶段
2. 初始化类中心
3. 域迁移阶段
4. 评估域迁移效果

---

### 3.6 工具脚本 ([scripts/](scripts/))

#### [scripts/preprocess_data.py](scripts/preprocess_data.py) (279行)

**功能**：将原始Excel数据转换为轻量级numpy格式

#### [scripts/diagnose_clicks.py](scripts/diagnose_clicks.py) (337行)

**功能**：追踪数据流向，定位片段数量膨胀

---

## 四、数据流和核心流程

### 4.1 Transformer 深度学习流程

```
原始 Excel 数据
    ↓
scripts/preprocess_data.py - 数据预处理
    ↓
pickle 格式数据（轻量级）
    ↓
HierarchicalGazeDataset - 数据集加载
    ↓
DataLoader - 批处理
    ↓
HierarchicalTransformerNetwork - 前向传播
  ├─ GazeTransformerEncoder（片段编码）
  ├─ AttentionPooling（片段聚合）
  ├─ TaskTransformerEncoder（任务编码）
  ├─ AttentionPooling（任务聚合）
  └─ PredictionHead（预测）
    ↓
DeepLearningTrainer - 训练
    ↓
结果输出（模型权重、评估指标、曲线图）
```

### 4.2 CADT 域迁移流程

```
预处理数据
    ↓
TwoByTwoSplitter - 数据划分
  ├─ train: 源域
  ├─ test1: 目标域1
  ├─ test2: 目标域2
  └─ test3: 目标域3
    ↓
DomainAdaptiveHierarchicalNetwork
  ├─ 预训练阶段
  │   └─ 源域分类器训练 + 初始化类中心
  ├─ 域迁移阶段
  │   ├─ 分类损失
  │   ├─ 中心对齐损失
  │   ├─ 域对抗损失
  │   └─ 中心多样性损失
    ↓
跨域性能评估
```

---

## 五、关键技术特点

### 5.1 7维原始特征

深度学习使用7维原始特征，无需手工特征工程：

| 维度 | 名称 | 描述 |
|------|------|------|
| 1 | x | 归一化X坐标 [0, 1] |
| 2 | y | 归一化Y坐标 [0, 1] |
| 3 | dt | 时间差（毫秒） |
| 4 | velocity | 瞬时速度 |
| 5 | acceleration | 瞬时加速度 |
| 6 | direction | 移动方向（归一化到[-1, 1]） |
| 7 | direction_change | 方向变化量 |

### 5.2 层级 Transformer 架构

```
输入：7维眼动特征序列
  ↓
片段编码器：100点 × 7维 → 64维片段表示
  ↓
注意力聚合：30个片段 → 任务表示
  ↓
任务编码器：30个任务 → 128维任务序列
  ↓
注意力聚合：30个任务 → 被试表示
  ↓
预测头：注意力分数/类别
```

### 5.3 CADT 域迁移方法

**核心思想**：通过类中心对齐和域对抗训练，学习域不变特征

**三重损失**：
1. **分类损失**：确保预测准确
2. **中心对齐损失**：拉近同类样本到类中心
3. **域对抗损失**：混淆源域和目标域

### 5.4 2×2 矩阵划分策略

| 集合 | 被试 | 任务 | 目的 |
|------|------|------|------|
| **train** | 前100人 | 前20题 | 训练集（基线性能） |
| **test1** | 后L人 | 前20题 | 新人旧题（跨被试泛化） |
| **test2** | 前100人 | 后10题 | 旧人新题（跨任务泛化） |
| **test3** | 后L人 | 后10题 | 新人新题（双重泛化） |

---

## 六、项目入口和使用方法

### 6.1 数据预处理

```bash
# 预处理数据（首次运行）
python scripts/preprocess_data.py
```

### 6.2 Transformer 深度学习模型

```bash
# 训练 Transformer 模型
python experiments/dl_transformer_experiment.py

# 使用环境变量配置
EXPERIMENT_MODE=formal \
TASK_TYPE=classification \
SPLIT_TYPE=2x2 \
python experiments/dl_transformer_experiment.py
```

### 6.3 CADT 域迁移实验

```bash
# 运行域迁移实验
python experiments/domain_adaptation_experiment.py

# 配置目标域
TARGET_DOMAIN=test2 \
PRETRAIN_EPOCHS=20 \
CENTER_ALIGNMENT_WEIGHT=10.0 \
python experiments/domain_adaptation_experiment.py
```

---

## 七、输出文件结构

```
outputs/
├── features/              # 特征文件
│   └── （暂未使用）
├── models/                # 模型文件
│   └── *.pt               # PyTorch 模型权重
├── figures/               # 可视化图表
│   └── training_curves_*.png
└── dl_models/             # 深度学习输出
    ├── model_fold{i}.pt
    ├── cv_results.json
    └── predictions.csv
```

---

## 八、模块依赖关系图

```
config.py (全局配置)
    ↓
experiments/*.py (实验脚本)
    ↓
src/data/
    ├── schemas.py (数据结构)
    ├── loader.py (数据加载)
    ├── preprocessor.py (预处理)
    └── split_strategy.py (划分策略)
    ↓
src/segmentation/
    └── event_segmenter.py (事件分割)
    ↓
src/models/
    ├── dl_dataset.py (数据集)
    ├── attention.py (注意力机制)
    ├── dl_models.py (模型架构)
    ├── domain_adaptation.py (域迁移)
    ├── dl_trainer.py (训练器)
    └── domain_trainer.py (域迁移训练器)
```

---

## 九、文件索引

### 根目录

| 文件 | 行数 | 功能描述 |
|------|------|----------|
| [config.py](config.py) | ~70 | 全局配置管理 |
| [main.py](main.py) | 13 | 主入口（占位符） |
| [requirements.txt](requirements.txt) | 20 | 项目依赖列表 |

### 数据处理模块 (src/data/)

| 文件 | 行数 | 功能描述 |
|------|------|----------|
| [schemas.py](src/data/schemas.py) | 260 | 数据结构定义 |
| [loader.py](src/data/loader.py) | 289 | 数据加载器 |
| [preprocessor.py](src/data/preprocessor.py) | 415 | 数据预处理器 |
| [split_strategy.py](src/data/split_strategy.py) | 359 | 数据划分策略 |

### 模型模块 (src/models/)

| 文件 | 行数 | 功能描述 |
|------|------|----------|
| [attention.py](src/models/attention.py) | 264 | 注意力机制 |
| [dl_models.py](src/models/dl_models.py) | 674 | Transformer 模型架构 |
| [dl_dataset.py](src/models/dl_dataset.py) | 371 | 深度学习数据集 |
| [dl_trainer.py](src/models/dl_trainer.py) | 897 | 深度学习训练器 |
| [domain_adaptation.py](src/models/domain_adaptation.py) | 283 | CADT 域迁移组件 |
| [domain_trainer.py](src/models/domain_trainer.py) | 568 | 域迁移训练器 |

### 实验模块 (experiments/)

| 文件 | 行数 | 功能描述 |
|------|------|----------|
| [dl_transformer_experiment.py](experiments/dl_transformer_experiment.py) | 774 | Transformer 深度学习实验 |
| [domain_adaptation_experiment.py](experiments/domain_adaptation_experiment.py) | 777 | CADT 域迁移实验 |
| [experiment_config.py](experiments/experiment_config.py) | 203 | 实验配置管理 |

### 脚本工具 (scripts/)

| 文件 | 行数 | 功能描述 |
|------|------|----------|
| [preprocess_data.py](scripts/preprocess_data.py) | 279 | 数据预处理脚本 |
| [diagnose_clicks.py](scripts/diagnose_clicks.py) | 337 | 点击数据诊断工具 |

---

## 十、更新记录

| 日期 | 变更说明 |
|------|----------|
| 2026-01-27 | 移除XGBoost相关内容（trainer.py, explainer.py） |
| 2026-01-27 | 移除传统特征工程（micro/meso/macro_features.py） |
| 2026-01-27 | 更新依赖配置，移除xgboost和shap |
| 2026-01-27 | 专注于深度学习（Transformer）模型 |

---

## 十一、总结

这是一个专注于深度学习的眼动追踪注意力预测系统。

### 项目核心价值

1. **端到端深度学习**：使用7维原始特征，无需手工特征工程
2. **层级Transformer架构**：有效处理眼动序列的时序依赖
3. **CADT 域迁移方法**：提升跨域泛化能力
4. **创新的实验设计**：2×2矩阵划分，全面评估模型性能
5. **GPU 优化**：支持混合精度训练、梯度检查点、多卡并行

### 适用场景

- **注意力评估**：基于眼动模式预测个体注意力水平
- **认知负荷分析**：通过眼动特征评估任务难度和认知负荷
- **跨域泛化研究**：验证模型在不同人群和任务场景下的泛化能力

---

*报告生成时间：2026-01-27*
*项目路径：/Users/ouonnki/Projects/py_projects/eye-trac-transformer*
