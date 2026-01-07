# 眼动追踪注意力预测系统

基于眼动轨迹数据预测个体注意力水平的机器学习系统。

## 项目结构

```
├── src/
│   ├── data/              # 数据加载与预处理
│   │   ├── loader.py      # 数据加载器
│   │   ├── preprocessor.py # 眼动数据预处理
│   │   └── schemas.py     # 数据结构定义
│   ├── features/          # 特征工程
│   │   └── extractor.py   # 手工特征提取
│   ├── models/            # 模型
│   │   ├── xgboost_model.py    # XGBoost基线模型
│   │   ├── dl_models.py        # Transformer深度学习模型
│   │   ├── dl_dataset.py       # 序列数据集
│   │   ├── dl_trainer.py       # 深度学习训练器
│   │   └── attention.py        # 注意力池化模块
│   └── segmentation/      # 眼动片段分割
│       └── event_segmenter.py
├── experiments/           # 实验脚本
│   ├── dl_transformer_experiment.py  # Transformer实验
│   └── ...
├── main.py               # XGBoost主程序
├── config.py             # 配置文件
└── requirements.txt      # 依赖列表
```

## 安装

```bash
# 克隆项目
git clone <repository-url>
cd boost/data

# 创建虚拟环境
conda create -n cadt python=3.10
conda activate cadt

# 安装依赖
pip install -r requirements.txt

# 如需GPU加速（推荐）
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## 数据准备

将眼动数据放置在 `data/gaze_trajectory_data/` 目录下，结构如下：

```
data/gaze_trajectory_data/
├── 个人标签.xlsx          # 被试标签（总分、类别）
├── 题目信息.xlsx          # 任务配置
├── 202011030100/         # 被试文件夹
│   ├── 1.xlsx            # 任务1数据
│   ├── 2.xlsx            # 任务2数据
│   └── ...
└── ...
```

## 使用方法

### 方法1: XGBoost基线模型

```bash
python main.py
```

### 方法2: Transformer深度学习模型

```bash
python experiments/dl_transformer_experiment.py
```

## 模型架构

### Hierarchical Transformer Network

```
输入: 7维眼动特征序列 (x, y, dt, velocity, acceleration, direction, direction_change)
  ↓
GazeTransformerEncoder: 片段序列 → 片段表示
  ↓
TaskAggregator: 注意力池化 → 任务表示
  ↓
TaskTransformerEncoder: 任务序列 → 编码任务间依赖
  ↓
SubjectAggregator: 注意力池化 → 被试表示
  ↓
输出: 注意力分数预测
```

## 服务器训练配置

针对单张 RTX 3090 (24GB) + 16核 Xeon + 31GB 内存优化：

| 配置项 | 值 | 说明 |
|--------|-----|------|
| batch_size | 16 | 单卡保守值（可尝试24） |
| use_multi_gpu | False | 单GPU |
| use_amp | True | FP16混合精度加速 |
| num_workers | 8 | 16核CPU的一半 |
| pin_memory | True | 锁页内存加速 |

**重要路径配置**：
- 数据目录：`/data/gaze_trajectory_data`
- 输出目录：`/data/outputs/dl_models`（避免使用 / 分区）

预期训练时间：约 30-45 分钟（5折交叉验证）

## 输出结果

训练完成后，结果保存在 `/data/outputs/dl_models/`：

- `model_fold{i}.pt`: 各折模型权重
- `cv_results.json`: 交叉验证指标
- `predictions.csv`: 预测结果

## 评估指标

- R²: 决定系数
- MAE: 平均绝对误差
- RMSE: 均方根误差
