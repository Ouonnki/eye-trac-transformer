# 眼动追踪注意力预测系统 (Eye-Trac-Transformer)

基于舒尔特方格任务的眼动轨迹数据，通过深度学习预测个体注意力水平的机器学习系统。

## 项目简介

本项目使用层级 Transformer 网络分析眼动轨迹数据，实现注意力水平的自动评估。系统支持两种输入方式：

- **被试级模型**：将每个被试的所有任务和片段作为整体输入（层级聚合）
- **片段级模型**：将每个搜索片段作为独立样本，推理时聚合为被试级预测

## 项目结构

```
eye-trac-transformer/
├── configs/                      # 配置文件目录
│   ├── default.json             # 默认配置（UnifiedConfig）
│   └── README.md                # 配置说明
├── docs/                        # 项目文档
│   ├── project-analysis-report.md    # 项目分析报告
│   ├── refactor-config-plan.md       # 配置重构计划
│   ├── refactor-models-plan.md       # 模型重构计划
│   └── ...
├── experiments/                 # 实验脚本
│   ├── dl_transformer_experiment.py  # 层级Transformer实验
│   ├── dl_cadt_experiment.py         # CADT域适应实验
│   └── segment_experiment.py         # 片段级实验
├── scripts/                     # 工具脚本
│   └── preprocess_data.py      # 数据预处理
├── outputs/                     # 输出结果
│   ├── dl_models/              # 深度学习模型输出
│   └── cadt_res/               # CADT实验结果
├── src/                         # 源代码
│   ├── config/                 # 配置系统
│   │   ├── __init__.py
│   │   └── config.py           # UnifiedConfig 统一配置
│   ├── data/                   # 数据处理
│   │   ├── loader.py           # 数据加载器
│   │   ├── preprocessor.py     # 数据预处理
│   │   ├── schemas.py          # 数据结构定义
│   │   └── split_strategy.py   # 2×2数据划分
│   ├── models/                 # 模型模块
│   │   ├── base.py             # BaseModel 抽象基类
│   │   ├── encoders.py         # 共享编码器
│   │   ├── heads.py            # 统一预测头
│   │   ├── attention.py        # 注意力机制
│   │   ├── dl_models.py        # 层级Transformer模型
│   │   ├── dl_cadt_models.py   # CADT域适应模型
│   │   ├── dl_dataset.py       # PyTorch数据集
│   │   ├── dl_trainer.py       # 训练器
│   │   ├── dl_cadt_trainer.py  # CADT训练器
│   │   ├── segment_model.py    # 片段级模型
│   │   └── segment_trainer.py  # 片段级训练器
│   ├── segmentation/           # 眼动分割
│   │   └── event_segmenter.py # 搜索片段切分
│   └── utils/                  # 工具函数
├── requirements.txt            # 依赖列表
└── README.md                   # 本文件
```

## 安装

```bash
# 克隆项目
git clone <repository-url>
cd eye-trac-transformer

# 创建虚拟环境
conda create -n eye-trac python=3.10
conda activate eye-trac

# 安装依赖
pip install -r requirements.txt

# 如需GPU加速（推荐）
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## 数据准备

### 原始数据结构

将眼动数据放置在 `data/gaze_trajectory_data/` 目录下：

```
data/gaze_trajectory_data/
├── 个人标签.xlsx          # 被试标签（总分、类别）
├── 题目信息.xlsx          # 任务配置
├── 202011030100/         # 被试文件夹
│   ├── 1.xlsx            # 任务1数据（sheet3:点击，sheet4:眼动）
│   ├── 2.xlsx            # 任务2数据
│   └── ...
└── ...
```

### 数据预处理

```bash
python scripts/preprocess_data.py --data_dir data/gaze_trajectory_data --output_dir outputs
```

预处理完成后会生成 `outputs/processed_data.pkl`，供后续训练使用。

## 使用方法

### 1. 层级Transformer模型（被试级）

```bash
python experiments/dl_transformer_experiment.py
```

**模型架构**：
```
输入: (batch, max_tasks, max_segments, max_seq_len, 7)
  ↓
[GazeTransformerEncoder] 片段序列 → 片段表示
  ↓
[AttentionPooling] 片段 → 任务表示
  ↓
[TaskTransformerEncoder] 任务序列 → 编码任务间依赖
  ↓
[AttentionPooling] 任务 → 被试表示
  ↓
[PredictionHead] 被试表示 → 注意力预测
```

### 2. CADT域适应模型

```bash
python experiments/dl_cadt_experiment.py
```

支持跨被试/跨任务域适应，通过修改 `configs/default.json` 中的 `cadt.target_domain` 设置目标域。

### 3. 片段级模型

```bash
python experiments/segment_experiment.py
```

将每个搜索片段作为独立样本，推理时聚合为被试级预测。

## 配置说明

所有配置通过 `configs/default.json` 管理：

| 配置项 | 说明 |
|--------|------|
| `experiment` | 实验配置（输出目录、随机种子等）|
| `model` | 模型架构参数（维度、层数、注意力头等）|
| `training` | 训练参数（学习率、batch size、早停等）|
| `task` | 任务配置（分类/回归、类别数）|
| `device` | 设备配置（GPU、混合精度等）|
| `cadt` | CADT域适应参数 |
| `output` | 输出配置（图表DPI等）|

详细说明参见 [configs/README.md](configs/README.md)

## 数据划分策略

### 2×2 矩阵划分

| 划分 | 被试范围 | 任务范围 | 说明 |
|------|---------|---------|------|
| train | 前100人 | 前20题 | 训练集 |
| test1 | 后49人 | 前20题 | 新人旧题（跨被试泛化）|
| test2 | 前100人 | 后10题 | 旧人新题（跨任务泛化）|
| test3 | 后49人 | 后10题 | 新人新题（双重泛化）|

## 服务器训练配置

针对单张 RTX 3090 (24GB) 优化：

| 配置项 | 值 | 说明 |
|--------|-----|------|
| `batch_size` | 16 | 可尝试24 |
| `use_amp` | true | FP16混合精度 |
| `use_multi_gpu` | false | 单GPU |
| `num_workers` | 8 | DataLoader工作进程 |

预期训练时间：约30-45分钟

## 输出结果

训练完成后，结果保存在输出目录：

- `model_fold{i}.pt`: 模型权重
- `cv_results.json`: 交叉验证指标
- `predictions.csv`: 预测结果
- `training_curves*.png`: 训练曲线

## 评估指标

### 分类任务
- Accuracy: 准确率
- F1 Score: F1分数（macro平均）
- Precision/Recall: 精确率/召回率

### 回归任务
- R²: 决定系数
- MAE: 平均绝对误差
- RMSE: 均方根误差

## 技术栈

- **深度学习框架**: PyTorch 2.0+
- **数据处理**: NumPy, Pandas
- **机器学习**: scikit-learn, XGBoost
- **可视化**: Matplotlib

## 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@software{eye_trac_transformer,
  title={Eye-Trac-Transformer: Eye Movement-Based Attention Prediction},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/eye-trac-transformer}
}
```

## 许可证

MIT License
