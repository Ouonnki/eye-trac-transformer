# 眼动追踪注意力预测系统 - 项目架构分析报告

**文档版本**: v4.0
**生成时间**: 2026-01-28
**分析范围**: 完整项目架构、模块依赖、数据流

---

## 1. 项目概览

| 属性 | 说明 |
|------|------|
| **项目名称** | Eye-Trac-Transformer |
| **核心功能** | 基于舒尔特方格任务的眼动轨迹数据，通过深度学习预测个体注意力水平 |
| **技术定位** | 眼动追踪 + 注意力预测的深度学习研究项目 |
| **代码规模** | 约 8,500 行 Python 代码 |
| **架构模式** | 分层架构（配置层 → 数据层 → 模型层 → 训练层 → 实验层） |

---

## 2. 项目目录结构

```
eye-trac-transformer/
├── configs/                        # 配置文件目录
│   ├── default.json                # 默认配置（UnifiedConfig）
│   └── README.md                   # 配置说明
│
├── docs/                           # 项目文档
│   ├── project-analysis-report.md  # 本报告
│   ├── refactor-config-plan.md     # 配置重构计划
│   ├── refactor-models-plan.md     # 模型重构计划
│   └── ...
│
├── experiments/                    # 实验脚本
│   ├── dl_transformer_experiment.py  # 层级Transformer实验
│   ├── dl_cadt_experiment.py         # CADT域适应实验
│   └── segment_experiment.py         # 片段级实验
│
├── outputs/                        # 输出结果
│   ├── dl_models/                  # 深度学习模型输出
│   └── cadt_res/                   # CADT实验结果
│
├── scripts/                        # 工具脚本
│   └── preprocess_data.py          # 数据预处理
│
├── src/                            # 核心源代码
│   ├── config/                     # 配置系统
│   │   ├── __init__.py
│   │   └── config.py               # UnifiedConfig
│   │
│   ├── data/                       # 数据处理
│   │   ├── loader.py               # 数据加载器
│   │   ├── preprocessor.py         # 数据预处理
│   │   ├── schemas.py              # 数据结构定义
│   │   └── split_strategy.py       # 2×2数据划分
│   │
│   ├── models/                     # 模型模块
│   │   ├── base.py                 # BaseModel 抽象基类
│   │   ├── encoders.py             # 共享编码器
│   │   ├── heads.py                # 统一预测头
│   │   ├── attention.py            # 注意力机制
│   │   ├── dl_models.py            # 层级Transformer模型
│   │   ├── dl_cadt_models.py       # CADT域适应模型
│   │   ├── dl_dataset.py           # PyTorch数据集
│   │   ├── dl_trainer.py           # 训练器
│   │   ├── dl_cadt_trainer.py      # CADT训练器
│   │   ├── segment_model.py        # 片段级模型
│   │   └── segment_trainer.py      # 片段级训练器
│   │
│   ├── segmentation/               # 眼动分割
│   │   └── event_segmenter.py     # 搜索片段切分
│   │
│   └── utils/                      # 工具函数
│       └── geometry.py             # 几何计算
│
├── requirements.txt                # 依赖列表
├── README.md                       # 项目说明
└── .gitignore                      # Git忽略配置
```

---

## 3. 架构分层与依赖关系

### 3.1 模块依赖矩阵

| 模块 | schemas | loader | preprocessor | split_strategy | attention | encoders | heads | dl_models | dl_dataset | dl_trainer | config |
|------|---------|--------|--------------|----------------|------------|----------|-------|-----------|------------|------------|--------|
| **schemas.py** | - | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **loader.py** | ✅ | - | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **preprocessor.py** | ✅ | ❌ | - | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **split_strategy.py** | ❌ | ❌ | ❌ | - | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **event_segmenter.py** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **attention.py** | ❌ | ❌ | ❌ | ❌ | - | ❌ | ❌ | ❌ | ❌ | ❌ |
| **encoders.py** | ❌ | ❌ | ❌ | ❌ | ✅ | - | ❌ | ❌ | ❌ | ❌ | ❌ |
| **heads.py** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | - | ❌ | ❌ | ❌ | ❌ |
| **base.py** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **dl_models.py** | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | - | ❌ | ❌ | ❌ |
| **dl_dataset.py** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | - | ❌ | ❌ |
| **dl_trainer.py** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | - | ✅ |
| **dl_cadt_models.py** | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| **dl_cadt_trainer.py** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |
| **segment_model.py** | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **segment_trainer.py** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ |

**图例**: ✅ = 依赖该模块，❌ = 不依赖

---

## 4. 核心数据结构 (src/data/schemas.py)

### 4.1 数据类层次

```
dataclasses
     │
     ├── ClickPoint (点击事件)
     │    ├── timestamp: datetime
     │    ├── x, y: float
     │    └── target_number: int
     │
     ├── GazePoint (眼动轨迹点)
     │    ├── timestamp: datetime
     │    └── x, y: float
     │
     ├── SearchSegment (搜索片段)
     │    ├── segment_id: int
     │    ├── start_number, target_number: int
     │    ├── gaze_points: List[GazePoint]
     │    └── target_position: Tuple[float, float]
     │
     ├── TaskConfig (任务配置)
     │    ├── task_id: int
     │    ├── grid_size: int
     │    └── number_range: Tuple[int, int]
     │
     ├── TaskTrial (单个任务试次)
     │    ├── subject_id: str
     │    ├── task_id: int
     │    ├── config: TaskConfig
     │    ├── clicks: List[ClickPoint]
     │    ├── gaze_points: List[GazePoint]
     │    └── segments: List[SearchSegment]
     │
     └── SubjectData (被试数据)
          ├── subject_id: str
          ├── total_score: float
          ├── category: str
          └── trials: List[TaskTrial]
```

### 4.2 数据流转

```
Excel文件
   ↓ DataLoader.load_subject()
SubjectData
   ├── TaskTrial
   │   ↓ GazePreprocessor.preprocess_trial()
   │   ├── clicks: List[ClickPoint]
   │   ├── gaze_points: List[GazePoint]
   │   ↓ EventSegmenter.segment()
   │   └── segments: List[SearchSegment]
   │       ↓ SequenceFeatureExtractor
   │       └── 特征序列 (7维 × N点)
   ↓ HierarchicalGazeDataset / SegmentGazeDataset
PyTorch DataLoader
   ↓ DeepLearningTrainer / SegmentTrainer
训练 → 模型
```

---

## 5. 模型架构详解

### 5.1 共享编码器 (src/models/encoders.py)

```
HierarchicalEncoder (层级编码器)
    │
    ├─► GazeTransformerEncoder (片段编码器)
    │   │
    │   ├─► 输入: (batch, seq_len, 7)
    │   ├─► 输入投影 + 位置编码 + [CLS] token
    │   ├─► TransformerEncoder (多层)
    │   └─► 输出: (batch, d_model)
    │
    ├─► AttentionPooling (片段→任务聚合)
    │   └─► 加权注意力池化
    │
    ├─► TaskTransformerEncoder (任务编码器)
    │   │
    │   ├─► 输入: (batch, num_tasks, d_model)
    │   ├─► 位置编码
    │   ├─► TransformerEncoder
    │   └─► 输出: (batch, num_tasks, task_d_model)
    │
    └─► AttentionPooling (任务→被试聚合)
        └─► 加权注意力池化
```

### 5.2 层级 Transformer 网络 (HierarchicalTransformerNetwork)

```
输入: (batch, max_tasks, max_segments, max_seq_len, 7)
  ↓
[HierarchicalEncoder]
  ├─► GazeTransformerEncoder → 片段表示
  ├─► AttentionPooling → 任务表示
  ├─► TaskTransformerEncoder → 编码任务间依赖
  └─► AttentionPooling → 被试表示
  ↓
[PredictionHead]
  ├─► 分类: Linear → num_classes logits
  └─► 回归: Linear → scalar score
```

### 5.3 片段级模型 (SegmentEncoder)

```
输入: (batch, seq_len, 7)
  ↓
[GazeTransformerEncoder] → 片段表示 (batch, d_model)
  ↓
[PredictionHead] → 片段预测 (batch, num_classes)
  ↓
[推理时聚合] → 被试预测
  ├─► 分类: 多数投票 / 平均概率
  └─► 回归: 简单平均
```

### 5.4 CADT 域适应模型

```
基础编码器: HierarchicalEncoder
    │
    ├─► [任务分类器] Classifier
    │   └─► CE Loss + Label Smoothing
    │
    └─► [域适应模块]
        │
        ├─► [双辨别器]
        │   ├─► Discriminator1 (特征 + 距离)
        │   │   └─► GradientReversalLayer (GRL)
        │   └─► Discriminator2 (纯特征)
        │
        └─► [原型学习]
            ├─► 类别中心初始化
            └─► 原型聚类损失
```

---

## 6. 文件详细说明

### 6.1 配置模块 [src/config/]

| 文件 | 行数 | 导出类 |
|------|------|--------|
| [config.py](src/config/config.py) | ~200 | `UnifiedConfig`, `ExperimentConfig`, `ModelConfig`, `TrainingConfig`, `TaskConfig`, `DeviceConfig`, `OutputConfig`, `CADTConfig` |

### 6.2 模型模块 [src/models/]

| 文件 | 行数 | 导出类 | 说明 |
|------|------|--------|------|
| [base.py](src/models/base.py) | ~30 | `BaseModel` | 抽象基类，定义 `from_config()` 接口 |
| [encoders.py](src/models/encoders.py) | ~370 | `GazeTransformerEncoder`, `TaskTransformerEncoder`, `HierarchicalEncoder` | 共享编码器 |
| [heads.py](src/models/heads.py) | ~60 | `PredictionHead` | 统一预测头 |
| [attention.py](src/models/attention.py) | ~145 | `AttentionPooling`, `PositionalEncoding` | 注意力机制 |
| [dl_models.py](src/models/dl_models.py) | ~150 | `HierarchicalTransformerNetwork` | 层级模型（使用共享编码器）|
| [dl_cadt_models.py](src/models/dl_cadt_models.py) | ~490 | `CADTTransformerModel` | CADT域适应模型 |
| [dl_dataset.py](src/models/dl_dataset.py) | ~450 | `SequenceConfig`, `SequenceFeatureExtractor`, `HierarchicalGazeDataset`, `SegmentGazeDataset` | 数据集 |
| [dl_trainer.py](src/models/dl_trainer.py) | ~900 | `DeepLearningTrainer` | 标准训练器 |
| [dl_cadt_trainer.py](src/models/dl_cadt_trainer.py) | ~730 | `CADTTrainer` | CADT训练器 |
| [segment_model.py](src/models/segment_model.py) | ~130 | `SegmentEncoder` | 片段级模型 |
| [segment_trainer.py](src/models/segment_trainer.py) | ~430 | `SegmentTrainer` | 片段级训练器 |

### 6.3 实验脚本 [experiments/]

| 文件 | 行数 | 功能 |
|------|------|------|
| [dl_transformer_experiment.py](experiments/dl_transformer_experiment.py) | ~930 | 层级Transformer实验 |
| [dl_cadt_experiment.py](experiments/dl_cadt_experiment.py) | ~435 | CADT域适应实验 |
| [segment_experiment.py](experiments/segment_experiment.py) | ~320 | 片段级实验 |

### 6.4 工具脚本 [scripts/]

| 文件 | 行数 | 功能 |
|------|------|------|
| [preprocess_data.py](scripts/preprocess_data.py) | ~260 | 数据预处理（Excel→Numpy）|

---

## 7. 代码统计

| 模块 | 文件数 | 代码行数 | 主要功能 |
|------|--------|----------|----------|
| 配置系统 (src/config/) | 1 | ~200 | 统一配置管理 |
| 配置文件 (configs/) | 2 | ~300 | JSON配置、文档 |
| 数据处理 (src/data/) | 4 | ~1,330 | 数据加载、预处理、结构定义 |
| 分割模块 (src/segmentation/) | 1 | ~315 | 事件分割 |
| 模型基础 (src/models/base*) | 1 | ~30 | 抽象基类 |
| 共享组件 (src/models/encoders, heads, attention) | 3 | ~575 | 共享编码器、预测头、注意力 |
| 层级模型 (src/models/dl_*) | 5 | ~2,720 | Transformer、CADT、数据集、训练器 |
| 片段模型 (src/models/segment_*) | 2 | ~560 | 片段级模型和训练器 |
| 实验脚本 (experiments/) | 3 | ~1,685 | 实验脚本 |
| 工具脚本 (scripts/) | 1 | ~260 | 数据预处理 |
| 文档 (docs/) | 5 | ~80,000 | 项目文档 |
| **总计** | **26** | **~8,500** | |

---

## 8. 项目特色

1. **模块化设计**: 共享编码器、统一预测头、抽象基类
2. **两种输入方式**: 被试级（层级聚合）和片段级（推理聚合）
3. **域适应**: CADT 模型实现跨被试/跨任务泛化
4. **统一配置**: JSON 驱动的配置系统，支持实验可复现
5. **GPU 优化**: 混合精度训练、多 GPU 并行、梯度检查点
6. **完整流水线**: 从 Excel 到模型预测的全流程自动化

---

## 9. 使用流程

```bash
# 1. 数据预处理（必需）
python scripts/preprocess_data.py --data_dir data/gaze_trajectory_data --output_dir outputs

# 2. 层级Transformer实验
python experiments/dl_transformer_experiment.py

# 3. CADT域适应实验
python experiments/dl_cadt_experiment.py

# 4. 片段级实验
python experiments/segment_experiment.py
```

---

## 10. 变更历史

### 2026-01-28 - 模型架构重构 (v4.0)
- **新增**:
  - `src/models/base.py` - BaseModel 抽象基类
  - `src/models/encoders.py` - 共享编码器（HierarchicalEncoder）
  - `src/models/heads.py` - 统一预测头
  - `src/models/segment_model.py` - 片段级模型
  - `src/models/segment_trainer.py` - 片段级训练器
  - `experiments/segment_experiment.py` - 片段级实验脚本
  - `docs/` - 项目文档目录
- **重构**:
  - `src/models/dl_models.py` - 使用共享编码器，代码量 486→148 行
  - `src/models/dl_cadt_models.py` - 使用共享编码器，代码量 611→491 行
  - `src/models/dl_trainer.py` - 使用 `from_config()` 接口
  - `src/models/dl_cadt_trainer.py` - 使用 `from_config()` 接口
  - `src/models/attention.py` - 清理未使用代码，264→145 行
  - `src/models/dl_dataset.py` - 添加 `SegmentGazeDataset` 类
- **删除**:
  - `config.py` (根目录) - 旧配置文件
  - `scripts/diagnose_clicks.py` - 调试工具
- **更新**:
  - `README.md` - 重写为中文，反映新架构

### 2026-01-27 - 配置系统重构 (v3.0)
- 统一配置管理 (UnifiedConfig)
- 删除环境变量驱动
- 新增 `configs/default.json`

### 2026-01-27 - 数据结构分离
- 分离点击和眼动数据
- 新增 `ClickPoint` 类
- `TaskTrial` 分离为 `clicks` 和 `gaze_points`

---

## 11. 当前 Git 状态

- **当前分支**: `classification`
- **主分支**: `main`
- **工作区状态**: 干净
- **最近提交**:
  - `9ba6ad0`: res: ce 加权损失结果
  - `cd17a42`: feat: 计算类别权重并设置加权交叉熵损失函数
