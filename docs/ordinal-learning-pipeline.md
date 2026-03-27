# 任务级序数学习全流程说明

本文只覆盖仓库中的“任务级序数学习”主线，不讨论层级 Transformer、CADT、片段级模型，也不引用它们的训练脚本、配置或实验结果。本文的目标是同时服务两类场景：

- 代码接手：快速定位真实调用路径、关键模块和配置项。
- 方法复现：从原始 Excel 到训练、评估、阈值校准，按顺序复现一条完整实验链路。

当前主线推荐以 `configs/task_level_small_lr_ordinal_pw_manual11.json` 为示例配置，以 `outputs/task_level/task_level_small_lr_ordinal_pw_manual11_20260311_153000` 为示例结果快照。若历史输出目录中的配置快照与当前 `configs/` 中的配置不完全一致，以当前仓库源码和 `configs/` 为准。

## 1. 任务与目标

### 1.1 任务定义

当前主线要解决的问题是：给定某个被试在某一道舒尔特方格任务上的完整眼动搜索过程，预测该“被试-题目”样本所属的三分类注意力等级。

- 预测单位：一个被试的一道题。
- 标签类型：有序三分类。
- 标签取值：`0 / 1 / 2`。
- 有序含义：类别之间存在强顺序关系，因此训练时采用序数学习而不是普通三分类交叉熵主线。

### 1.2 仓库中的真实主线路径

任务级序数学习的真实调用链如下：

```text
原始 Excel
  -> src/data/loader.py
  -> src/data/preprocessor.py
  -> src/segmentation/event_segmenter.py
  -> scripts/preprocess_data.py
  -> data/processed/processed_data_subject_norm.pkl
  -> src/models/task_level_dataset.py
  -> scripts/train_task_level.py
  -> src/models/task_level_model.py
  -> src/models/task_level_trainer.py
  -> src/models/losses.py
  -> outputs/task_level/<run_dir>/
  -> scripts/calibrate_ordinal_thresholds.py
```

### 1.2.1 全流程图

下面这张图把“原始数据 -> 预处理 -> 训练 -> 评估 -> 校准”这条主线完整串起来了：

```text
┌──────────────────────────┐
│ data/gaze_trajectory_data│
│ 个人标签.xlsx           │
│ 题目信息.xlsx           │
│ 题号1到30_分类结果.xlsx │
│ <subject>/<task>.xlsx   │
└────────────┬─────────────┘
             │
             v
┌──────────────────────────┐
│ src/data/loader.py       │
│ load_labels/load_tasks   │
│ load_subject/load_trial  │
└────────────┬─────────────┘
             │
             v
┌──────────────────────────┐
│ src/data/preprocessor.py │
│ 点击表清洗               │
│ 眼动轨迹清洗             │
│ 时间戳解析与坐标裁剪     │
└────────────┬─────────────┘
             │
             v
┌──────────────────────────┐
│ src/segmentation/        │
│ event_segmenter.py       │
│ 相邻点击 -> 搜索片段     │
└────────────┬─────────────┘
             │
             v
┌──────────────────────────┐
│ scripts/preprocess_data.py│
│ 7维特征提取              │
│ 任务级标签 1/2/3 -> 0/1/2│
│ 被试内标准化             │
└────────────┬─────────────┘
             │
             v
┌────────────────────────────────────────────┐
│ data/processed/processed_data_subject_norm │
│ .pkl                                       │
└────────────┬───────────────────────────────┘
             │
             v
┌──────────────────────────┐
│ TaskLevelGazeDataset     │
│ 一个样本 = 一个被试一道题│
│ collate_fn 动态补齐 batch│
└────────────┬─────────────┘
             │
             v
┌──────────────────────────┐
│ scripts/train_task_level │
│ split_2x2                │
│ train / val / test1~3    │
└────────────┬─────────────┘
             │
             v
┌──────────────────────────┐
│ TaskLevelEncoder         │
│ OrdinalLoss              │
│ TaskLevelTrainer         │
└────────────┬─────────────┘
             │
             v
┌─────────────────────────────────────────┐
│ outputs/task_level/<run_dir>/           │
│ config.json / history.json              │
│ test_results.json / training_curves.png │
└────────────┬────────────────────────────┘
             │
             v
┌──────────────────────────┐
│ calibrate_ordinal_       │
│ thresholds.py            │
│ Val 网格搜索阈值         │
└──────────────────────────┘
```

### 1.3 当前主线涉及的核心文件

- `scripts/preprocess_data.py`：离线预处理总入口。
- `src/data/loader.py`：读取标签、题目信息、单题轨迹 Excel。
- `src/data/preprocessor.py`：时间戳解析、坐标裁剪、点击和眼动清洗。
- `src/segmentation/event_segmenter.py`：按相邻点击切分搜索片段。
- `src/models/task_level_dataset.py`：把预处理结果展开成任务级样本。
- `src/models/task_level_model.py`：任务级模型 `TaskLevelEncoder`。
- `src/models/task_embedding.py`：任务条件嵌入。
- `src/models/attention.py`：片段注意力聚合。
- `src/models/losses.py`：序数标签构造、序数损失、阈值解码。
- `src/models/task_level_trainer.py`：训练与评估逻辑。
- `scripts/train_task_level.py`：任务级训练脚本。
- `scripts/calibrate_ordinal_thresholds.py`：验证集阈值校准脚本。

## 2. 原始数据与标签来源

### 2.1 目录结构

原始数据默认位于 `data/gaze_trajectory_data/`，组织形式如下：

```text
data/gaze_trajectory_data/
  个人标签.xlsx
  题目信息.xlsx
  题号1到30_分类结果.xlsx
  202011030100/
    1.xlsx
    2.xlsx
    ...
    30.xlsx
  202011030101/
    1.xlsx
    ...
```

### 2.2 文件职责

- `个人标签.xlsx`
  - 由 `src/data/loader.py` 的 `load_labels()` 读取。
  - 主要字段是 `被试编号`、`总分`、类别列。
  - 其中 `category` 保持原始的 `1 / 2 / 3` 编码，主要用于被试级任务或辅助统计。

- `题目信息.xlsx`
  - 由 `load_tasks()` 读取。
  - 用于构建 `TaskConfig`，包含方格数量、数字范围、点击后是否消失、干扰项数量等任务条件。

- `题号1到30_分类结果.xlsx`
  - 由 `scripts/preprocess_data.py` 中的 `load_task_level_labels()` 读取。
  - 每个 sheet 对应一题，每行对应一个被试。
  - 关键字段是 `class_1_2_3`。
  - 在预处理阶段会执行 `label = int(row["class_1_2_3"]) - 1`，将原始 `1 / 2 / 3` 转为任务级训练使用的 `0 / 1 / 2`。

- 被试目录下的 `1.xlsx ~ 30.xlsx`
  - 由 `load_gaze_trajectory(subject_id, task_id)` 读取。
  - `sheet3`：点击数据。
  - `sheet4`：连续眼动轨迹。

### 2.3 标签体系要点

当前仓库里同时存在两套标签语义：

- 被试级 `category`
  - 来源：`个人标签.xlsx`
  - 编码：`1 / 2 / 3`
  - 当前任务级主线不直接拿它训练。

- 任务级 `task_label`
  - 来源：`题号1到30_分类结果.xlsx`
  - 编码：预处理后固定为 `0 / 1 / 2`
  - 当前任务级序数学习真正使用的是它。

这也是后文“常见坑”里要特别强调的 1-based / 0-based 区别。

## 3. 预处理流程

### 3.1 总入口

预处理入口是：

```bash
python scripts/preprocess_data.py --data_dir data/gaze_trajectory_data --output_dir data/processed --subject_normalize
```

脚本内部流程可以概括为：

```text
DataLoader 读取标签和题目元信息
  -> 逐被试 load_subject()
  -> GazePreprocessor.preprocess_trial()
  -> AdaptiveSegmenter.segment()
  -> extract_features() 生成 7 维序列
  -> 可选 normalize_subject_features() 做被试内标准化
  -> 保存为 pickle
```

### 3.2 DataLoader 负责什么

`src/data/loader.py` 的职责是把原始 Excel 组织成统一数据对象：

- `load_labels()`
  - 读取 `个人标签.xlsx`。
- `load_tasks()`
  - 读取 `题目信息.xlsx`。
  - 解析 `grid_size`、`number_range`、`click_disappear`、`has_distractor`、`grid_distractor_count`、`number_distractor_count`。
- `load_gaze_trajectory(subject_id, task_id)`
  - 从被试目录读取单题 Excel。
  - `sheet3` 作为点击表。
  - `sheet4` 作为连续眼动表。
- `load_subject(subject_id)`
  - 组装一个被试的 30 道题试次。
  - 每个试次先挂原始 DataFrame，等待后续预处理。

### 3.3 GazePreprocessor 的工作

`src/data/preprocessor.py` 的 `GazePreprocessor` 主要完成两类事情。

第一类是点击数据预处理：

- 自动匹配列名，如 `时间`、`坐标X`、`坐标Y`、`正确`。
- 解析两类时间戳格式：
  - `2025/10/2 18:58:11:842`
  - `2025-10-02 18:58:11.842`
- 对点击坐标做数值化和屏幕范围裁剪。
- 给每次点击补出 `target_number`。

第二类是连续眼动轨迹预处理：

- 自动匹配时间列和坐标列。
- 解析时间戳。
- 把坐标裁剪到 `[0, screen_width]` 和 `[0, screen_height]`。
- 把结果转成 `GazePoint(timestamp, x, y)`。

这里并不会做复杂的滤波或插值；`GazeDataCleaner` 提供了额外清洗工具，但当前主线的 `scripts/preprocess_data.py` 并没有主动调用。

### 3.4 搜索片段如何切分

`src/segmentation/event_segmenter.py` 的 `AdaptiveSegmenter` 基于相邻点击构造搜索片段。

核心定义是：

- 一个片段表示从 `Click(N-1)` 到 `Click(N)` 的搜索过程。
- 片段内容包含：
  - 起始点击位置
  - 两次点击之间的全部连续眼动点
  - 结束点击位置

这意味着：

- 如果一道题只有 1 次点击，无法形成搜索片段，会被跳过。
- 如果一道题有 `n` 次点击，就会得到 `n - 1` 个搜索片段。
- 每个片段都至少包含两个由点击转换而来的点，即便中间没有连续眼动点。

`AdaptiveSegmenter` 还会在网格布局不完整时，用点击坐标补齐目标数字到屏幕位置的映射。

### 3.5 7 维特征构造

`scripts/preprocess_data.py` 中的 `extract_features()` 会把一个片段的点序列转成 `(seq_len, 7)` 的特征矩阵，7 维分别是：

1. `x`：按屏幕宽度归一化后的横坐标。
2. `y`：按屏幕高度归一化后的纵坐标。
3. `dt`：当前点与前一时刻的时间差，单位毫秒，最小裁成 `1.0`。
4. `velocity`：位移 / 时间差。
5. `acceleration`：速度差 / 时间差。
6. `direction`：位移方向，经 `atan2(dy, dx) / pi` 归一化到 `[-1, 1]`。
7. `direction_change`：相邻方向变化量，经 `pi` 归一化到 `[0, 1]`。

### 3.6 首片段首点与非首片段首点的差异

这个细节非常关键，因为仓库专门对它做了区分。

- 如果当前片段是该题的第一个片段：
  - 第 0 个点代表任务刚开始。
  - 没有历史动态信息。
  - `dt/velocity/acceleration/direction/direction_change` 统一置 0。

- 如果当前片段不是第一个片段：
  - 第 0 个点代表从上一点击位置重新起步。
  - 会借助第 1 个点估计初始 `dt`、`velocity`、`direction`。
  - 这样非首片段的起点不再被误当成“静止的零状态”。

这个设计的目的是保留“上一目标刚刚点击完成，视线重新发起搜索”的动态痕迹。

### 3.7 被试内标准化

当命令行传入 `--subject_normalize` 时，脚本会调用 `normalize_subject_features()` 对每个被试内部的所有片段做标准化。

标准化规则不是对 7 维全部统一处理，而是只处理动态维度：

- 会标准化的维度：
  - `dt`
  - `velocity`
  - `acceleration`

- 保持原值的维度：
  - `x`
  - `y`
  - `direction`
  - `direction_change`

原因是：

- `x/y` 承载屏幕空间布局，过度标准化会破坏任务几何结构。
- `direction` 和 `direction_change` 已经是归一化角度量。
- `dt/velocity/acceleration` 更容易受个体搜索节奏影响，所以做被试内对齐更合理。

此外还有两条实现细节：

- 标准化后会对 `dt/velocity/acceleration` 做 `[-10, 10]` 的裁剪，防止极端异常值破坏训练。
- 对每个任务的第一个片段，其第 0 个点在标准化后会再次强制恢复为全零动态特征。

### 3.8 任务和被试何时会被跳过

预处理时并不是所有原始记录都会进入最终数据集。以下情况会被跳过：

- 该题没有点击数据。
- 相邻点击不足 2 次，无法构成片段。
- 在任务级标签表中找不到 `(subject_id, task_id)` 对应的 `task_label`。
- 所有片段提取后都为空。
- 整个被试最终没有任何有效任务。

### 3.9 预处理输出结构

输出文件有两个：

- `data/processed/processed_data.pkl`
  - 只做特征提取，不做被试内标准化。
- `data/processed/processed_data_subject_norm.pkl`
  - 在上面的基础上额外做被试内标准化。

当前任务级主线示例配置使用的是后者。

输出结构可以概括为：

```python
[
    {
        "subject_id": "202011030100",
        "label": 73.0,
        "category": 2,
        "tasks": [
            {
                "task_id": 1,
                "task_label": 0,
                "segments": [np.ndarray(shape=(seq_len, 7)), ...],
                "task_conditions": {
                    "grid_size": 25,
                    "number_range": (1, 25),
                    "click_disappear": False,
                    "has_distractor": True,
                    "distractor_count": 4,
                    "grid_distractor_count": 4,
                    "number_distractor_count": 0,
                },
            },
            ...
        ],
        "normalization_stats": {
            "dt_mean": ...,
            "dt_std": ...,
            "velocity_mean": ...,
            "velocity_std": ...,
            "acceleration_mean": ...,
            "acceleration_std": ...,
        },
    },
    ...
]
```

## 4. 任务级样本构造

### 4.1 一个样本是什么

`src/models/task_level_dataset.py` 中的 `TaskLevelGazeDataset` 会把预处理结果展开成任务级样本列表：

- 一个样本 = 一个被试的一道题。
- 数据集遍历 `processed_data` 中每个 `subject_data["tasks"]`。
- 只保留包含 `task_label` 的任务。

最终 `self.samples` 中每条记录大致包含：

- `subject_id`
- `task_id`
- `segments`
- `task_conditions`
- `label`

### 4.2 输入张量的语义

`__getitem__()` 返回的核心字段是：

- `segments`
  - 形状：`(S, 300, 7)`
  - 含义：该题所有搜索片段的时序特征。
- `segment_mask`
  - 形状：`(S,)`
  - 含义：哪些片段有效。
- `segment_seq_mask`
  - 形状：`(S, 300)`
  - 含义：每个片段内部哪些时间步有效。
- `task_conditions`
  - 形状：`(5,)`
  - 含义：该题的 5 维离散任务条件。
- `label`
  - 形状：标量
  - 含义：`0 / 1 / 2` 序数标签。

### 4.3 `max_seq_len=300` 和 `max_segments=25` 的真实作用

这里有一个容易误解的点。

#### `max_seq_len=300`

这是当前实现中的硬截断长度：

- 每个片段最多保留 300 个时间步。
- 少于 300 的部分用 0 填充。
- 有效时间步由 `segment_seq_mask` 标出来。

#### `max_segments=25`

在任务级数据集里，它更像“语义上预期的上界”，而不是当前实现中的硬截断阈值：

- `TaskLevelGazeDataset.__getitem__()` 直接按真实片段数 `num_segments = len(sample["segments"])` 创建张量。
- 它不会主动把片段数截断到 25。
- 真正的批量对齐发生在 `task_level_collate_fn()`：
  - 先找出当前 batch 内最大的片段数。
  - 再把所有样本 pad 到该 batch 的最大值。

因此，对任务级主线来说：

- `max_seq_len=300` 是硬截断。
- `max_segments=25` 目前主要是配置层语义和上游任务设计假设，不是硬截断实现。

### 4.4 `task_conditions` 的编码规则

任务条件最终被编码成 5 维整数向量：

1. `grid_scale`
   - `9 -> 1`
   - `16 -> 2`
   - `25 -> 3`
   - `36 -> 4`

2. `continuous_thinking`
   - 若 `number_range` 上界是 `99`，则为 `1`
   - 否则为 `0`

3. `click_disappear`
   - 点击后数字是否消失
   - `False -> 0`
   - `True -> 1`

4. `has_distractor`
   - 是否存在方格干扰项
   - 依据 `grid_distractor_count > 0`

5. `has_task_distractor`
   - 是否存在数字干扰项
   - 依据 `number_distractor_count > 0`

## 5. 数据划分与增强

### 5.1 训练脚本实际使用的划分逻辑

当前任务级训练脚本 `scripts/train_task_level.py` 没有调用 `src/data/split_strategy.py` 的 `TwoByTwoSplitter`，而是内置了一版自己的 `split_2x2()`。

这版逻辑的真实规则是：

1. 从 `dataset.samples` 中收集全部 `subject_id` 并排序。
2. 取前 `train_subjects` 个被试作为训练被试，默认是前 100 人。
3. 取任务 `1 ~ train_tasks` 作为训练题目，默认是前 20 题。
4. 构成 4 个大区块：
   - `train_pool`：前 100 人 × 前 20 题
   - `test1`：其余被试 × 前 20 题
   - `test2`：前 100 人 × 后 10 题
   - `test3`：其余被试 × 后 10 题
5. 再把 `train_pool` 里的被试随机打乱，并按 `train_val_split=0.9` 做 `train/val` 被试级划分。

也就是说，`val` 不是从样本级随机抽，而是从 `train_pool` 里的被试级切出来的同分布验证集。

### 5.2 四个评估分布的意义

- `Val (同分布)`
  - 与训练集使用同一批题目范围。
  - 只是在被试上做了留出。

- `Test1 (新被试+旧题)`
  - 训练题目不变。
  - 被试变成训练中没见过的人。
  - 用来看跨被试泛化。

- `Test2 (旧被试+新题)`
  - 被试仍是训练阶段见过的人。
  - 题目换成训练阶段没见过的题。
  - 用来看跨任务泛化。

- `Test3 (新被试+新题)`
  - 被试和题目都没有在训练阶段出现。
  - 是最难的双重泛化分布。

### 5.3 类别不平衡处理

`scripts/train_task_level.py` 提供了三类不平衡处理入口，但在当前序数主线里真正起主导作用的是后两类。

第一类是 `class_weights`：

- 通过 `compute_class_weights()` 或手动配置得到。
- 主要服务于分类版交叉熵。
- 对当前 `head_type=ordinal` 主线，`TaskLevelTrainer` 会直接走 `OrdinalLoss` 分支，因此 `class_weights` 不参与序数损失计算。

第二类是 `use_balanced_sampler`：

- 当前示例配置开启。
- 默认模式是 `effective_num`。
- 会构建 `WeightedRandomSampler`，在 DataLoader 层面对训练样本重采样。

第三类是 `ordinal_pos_weight`：

- 这是序数主线真正直接参与损失计算的类别不平衡校正方式。
- 详见第 7 章。

### 5.4 在线增强

当前任务级主线支持 5 种在线增强，定义在 `src/models/augmentation.py` 中。

1. `Gaussian Noise`
   - 对 `x/y` 加高斯噪声。
   - 仅作用于有效时间步。

2. `Temporal Scaling`
   - 对 `dt/velocity/acceleration` 乘一个随机缩放因子。

3. `Segment Dropout`
   - 随机丢弃整个片段。
   - 至少保留 1 个有效片段。

4. `Time Masking`
   - 在时间维随机遮蔽连续窗口。

5. `Feature Masking`
   - 随机把某些通道整段置零。

这些增强只有在满足两个条件时才会生效：

- 配置里显式给出 `augmentation` 参数。
- 当前样本索引属于训练集 `training_indices`。

验证集和测试集不会走增强。

## 6. 模型结构

### 6.1 主模型 `TaskLevelEncoder`

任务级主模型定义在 `src/models/task_level_model.py`，输入是：

- `segments`: `(B, S, 300, 7)`
- `segment_mask`: `(B, S)`
- `task_conditions`: `(B, 5)`
- `segment_seq_mask`: `(B, S, 300)`

主干结构如下：

```text
每个片段的 7 维时间序列
  -> 片段编码器
  -> 片段表示 (B, S, segment_d_model)
  -> AttentionPooling 做片段级聚合
  -> 任务表示 (B, segment_d_model)
  -> 任务条件嵌入
  -> 融合
  -> PredictionHead
  -> 分类 logits 或序数 logits
```

### 6.1.1 完整模型架构图

下面这张图按真实张量流向展开了 `TaskLevelEncoder` 的完整结构：

```text
segments (B, S, 300, 7)
segment_mask (B, S)
segment_seq_mask (B, S, 300)
task_conditions (B, 5)
        │
        ├──────────────────────────────────────────────────────────────┐
        │                                                              │
        v                                                              v
┌──────────────────────┐                                  ┌─────────────────────────┐
│ 片段展开             │                                  │ 任务条件编码            │
│ (B,S,300,7)          │                                  │ task_conditions (B,5)   │
│ -> (B*S,300,7)       │                                  │                         │
└──────────┬───────────┘                                  │ independent:            │
           │                                              │   5个子嵌入拼接         │
           v                                              │   输出 5*emb_dim        │
┌──────────────────────┐                                  │                         │
│ 片段编码器           │                                  │ mlp:                    │
│ transformer/cnn1d/   │                                  │   5维条件 -> MLP        │
│ rnn/lstm/gru/bilstm  │                                  │   输出 emb_dim          │
└──────────┬───────────┘                                  └──────────┬──────────────┘
           │                                                         │
           v                                                         │
┌──────────────────────┐                                             │
│ segment_reprs        │                                             │
│ (B, S, segment_d)    │                                             │
└──────────┬───────────┘                                             │
           │                                                         │
           ├──────────── use_conditional_pooling = false ─────────────┤
           │                                                         │
           └────── use_conditional_pooling = true -> condition -------┘
                                   │
                                   v
                         ┌──────────────────────┐
                         │ AttentionPooling     │
                         │ 打分 = W2(tanh(W1h + │
                         │ 可选 Wcond(c)))      │
                         └──────────┬───────────┘
                                    │
                                    v
                         ┌──────────────────────┐
                         │ task_repr            │
                         │ (B, segment_d)       │
                         └──────────┬───────────┘
                                    │
                                    ├── 若 use_task_embedding=true
                                    │   与 task_emb 拼接
                                    v
                         ┌──────────────────────┐
                         │ fused_repr           │
                         │ (B, segment_d+emb)   │
                         └──────────┬───────────┘
                                    │
                                    v
                         ┌──────────────────────┐
                         │ PredictionHead       │
                         │ Linear -> GELU ->    │
                         │ Dropout -> Linear    │
                         └──────────┬───────────┘
                                    │
               ┌────────────────────┴────────────────────┐
               │                                         │
               v                                         v
     classification logits (B, C)            ordinal logits (B, C-1)
                                                      │
                                                      v
                                      build_ordinal_targets / OrdinalLoss
                                                      │
                                                      v
                                   decode_ordinal_logits + thresholds
                                                      │
                                                      v
                                          最终类别 0 / 1 / 2
```

### 6.2 支持的片段编码器族

片段编码器由 `model.segment_encoder_type` 控制，当前支持 6 类：

- `transformer`
  - `GazeTransformerEncoder`
  - 结构包括输入投影、`[CLS] token`、位置编码、多层 `TransformerEncoderLayer`。

- `cnn1d`
  - `GazeCnnEncoder`
  - 把输入转成 `(batch, input_dim, seq_len)` 后做一维卷积，再池化。

- `rnn`
  - `GazeRnnEncoder` 的简单 RNN 版本。

- `lstm`
  - 单向 LSTM。

- `gru`
  - 单向 GRU。

- `bilstm`
  - 双向 LSTM。
  - 隐状态拼接后再投影到 `segment_d_model`。

### 6.3 片段注意力聚合

片段编码器输出的形状是 `(B, S, segment_d_model)`，随后交给 `AttentionPooling`。

该模块的逻辑是：

- 先做一层线性投影到 `attention_dim`。
- 经过 `tanh` 和 `dropout`。
- 再投影成每个片段的一个标量分数。
- 对分数做带掩码的 `softmax`。
- 得到片段权重，并对片段表示做加权求和。

如果 `use_conditional_pooling=True`，任务嵌入还会先经过 `W_cond` 投影，再加到注意力隐层上，形成条件化注意力。

### 6.4 任务嵌入与融合

当前任务级主线支持两种任务嵌入方式。

#### 独立嵌入 `task_embedding_type=independent`

定义在 `TaskEmbedding` 中：

- `grid_scale` 使用一个可学习向量乘以尺度值，属于连续嵌入。
- 其余 4 个二值条件使用 `nn.Embedding(2, task_embedding_dim)`。
- 最终把 5 个子嵌入拼起来。

输出维度固定是：

```text
task_emb_output_dim = 5 * task_embedding_dim
```

例如：

- `task_embedding_dim=2`
- 则最终任务嵌入维度是 `10`

#### MLP 嵌入 `task_embedding_type=mlp`

定义在 `MLPTaskEmbedding` 中：

- 先把 5 维条件转成浮点数。
- 把 `grid_scale` 从 `1~4` 归一化到 `0.25~1.0`。
- 再过两层 MLP。

输出维度固定是：

```text
task_emb_output_dim = task_embedding_dim
```

例如：

- `task_embedding_dim=16`
- 则 MLP 嵌入输出维度就是 `16`

### 6.5 融合方式

当前任务级模型有两种与任务条件相关的融合点：

1. 注意力阶段
   - 只有当 `use_conditional_pooling=True` 时，任务嵌入才会进入注意力打分。

2. 预测头前
   - 只要 `use_task_embedding=True`，聚合后的任务表示都会与任务嵌入拼接。

因此，即使没有开启条件化池化，任务嵌入也仍然会在预测头前参与融合。

### 6.6 预测头

`PredictionHead` 是一个两层 MLP：

```text
Linear(input_dim -> hidden_dim)
  -> GELU
  -> Dropout
  -> Linear(hidden_dim -> output_dim)
```

输出维度取决于 `head_type`：

- `classification`
  - 输出维度是 `num_classes`
- `ordinal`
  - 输出维度是 `num_classes - 1`

## 7. 序数学习机制

### 7.1 为什么不是普通三分类

当前任务标签虽然只有三类，但三类之间有顺序关系，普通交叉熵会把“把 0 预测成 2”和“把 0 预测成 1”视作同等级错误。序数学习则能显式建模这种顺序结构。

### 7.2 输出维度

对当前三分类任务，序数头输出的是两个累计 logits：

```text
num_classes = 3
output_dim = num_classes - 1 = 2
```

可以把它理解为在学习两个二值判断：

- 是否大于 0
- 是否大于 1

### 7.3 `build_ordinal_targets()` 如何构造标签

`src/models/losses.py` 中的 `build_ordinal_targets(labels, num_classes)` 会把离散标签转换成累计二值目标。

对于三分类：

```text
y = 0 -> [0, 0]
y = 1 -> [1, 0]
y = 2 -> [1, 1]
```

数学上等价于：

```text
target[k] = 1{ y > k }
```

### 7.4 `OrdinalLoss` 如何计算

`OrdinalLoss` 本质上是带 `pos_weight` 的 `binary_cross_entropy_with_logits`：

- 输入：序数头输出的 `(B, C-1)` logits
- 目标：累计二值标签 `(B, C-1)`
- 损失：对每个阈值判断分别做 BCE，然后取均值

它不像普通多分类那样直接对 3 个类做 softmax，而是让模型分别学习：

- 这个样本是不是超过第 0 个等级
- 这个样本是不是超过第 1 个等级

### 7.5 `ordinal_pos_weight_mode`

当前训练脚本支持 3 种模式。

#### `none`

- 不使用 `pos_weight`
- 所有阈值判断的正负样本权重一样

#### `manual`

- 直接从配置读取 `training.ordinal_pos_weight`
- 示例主线配置使用：

```json
"ordinal_pos_weight": [0.5, 1.1]
```

#### `auto`

- 调用 `compute_ordinal_pos_weight()`
- 对每个阈值分别统计：
  - `pos = (labels > k).sum()`
  - `neg = (labels <= k).sum()`
- 再用：

```text
w_k = clip(neg / pos, clip_min, clip_max)
```

作为该阈值的 `pos_weight`

### 7.6 `decode_ordinal_logits()` 如何解码

推理阶段不会直接对两个 logits 取 `argmax`，而是：

1. 对每个 logit 做 `sigmoid` 得到累计概率 `P(y > k)`。
2. 将它与 `ordinal_thresholds[k]` 比较。
3. 统计有多少个阈值被判为真。
4. 真值个数就是最终类别。

三分类时：

- 两个阈值都不过：预测 0
- 只过第一个：预测 1
- 两个都过：预测 2

### 7.7 `ordinal_probs_to_class_probs()` 如何恢复类别概率

评估和校准脚本需要类别概率，因此仓库又把累计概率恢复为类别概率分布。

设：

- `p0 = P(y > 0)`
- `p1 = P(y > 1)`

则：

```text
P(y = 0) = 1 - p0
P(y = 1) = p0 - p1
P(y = 2) = p1
```

实现里还做了两步保护：

- 先把累计概率投影成单调非增，防止数值异常。
- 再做归一化，保证三类概率和为 1。

## 8. 训练配置与优化

### 8.1 主线示例配置

当前推荐主线配置是 `configs/task_level_small_lr_ordinal_pw_manual11.json`，关键字段如下：

```json
{
  "model": {
    "segment_d_model": 32,
    "segment_nhead": 2,
    "segment_num_layers": 1,
    "attention_dim": 16,
    "task_embedding_dim": 2,
    "use_task_embedding": true,
    "dropout": 0.0,
    "num_classes": 3,
    "head_type": "ordinal",
    "segment_encoder_type": "transformer"
  },
  "training": {
    "batch_size": 16,
    "epochs": 100,
    "lr": 0.0003,
    "weight_decay": 0.0,
    "grad_clip": 1.0,
    "patience": 20,
    "early_stop_metric": "f1_macro",
    "train_val_split": 0.9,
    "use_balanced_sampler": true,
    "balanced_sampler_mode": "effective_num",
    "balanced_sampler_beta": 0.999,
    "ordinal_thresholds": [0.47, 0.5],
    "ordinal_pos_weight_mode": "manual",
    "ordinal_pos_weight": [0.5, 1.1]
  },
  "data": {
    "processed_data_path": "data/processed/processed_data_subject_norm.pkl"
  }
}
```

### 8.2 优化器与学习率调度

`TaskLevelTrainer` 中的默认优化设置是：

- 优化器：`AdamW`
- 学习率：来自配置中的 `training.lr`
- 权重衰减：`training.weight_decay`
- 调度器：`ReduceLROnPlateau`
  - 监控的是验证损失
  - `patience=10`
  - `factor=0.5`

### 8.3 梯度裁剪与早停

- 梯度裁剪：`training.grad_clip`
  - 当前示例配置为 `1.0`
- 早停监控指标：`training.early_stop_metric`
  - 当前示例配置为 `f1_macro`
- 早停耐心值：`training.patience`
  - 当前示例配置为 `20`

### 8.4 Focal Loss、Label Smoothing 与序数主线的关系

虽然训练配置里保留了 `use_focal_loss` 和 `label_smoothing` 字段，但对当前序数主线要区分看待：

- `head_type=ordinal` 时
  - 训练器直接走 `OrdinalLoss`
  - 不会使用分类版 `FocalLoss`
  - 也不会使用交叉熵分支的 `label_smoothing`

所以在当前序数主线里，真正直接影响损失的核心字段是：

- `ordinal_thresholds`
- `ordinal_pos_weight_mode`
- `ordinal_pos_weight`

### 8.5 训练目录产物

训练脚本会把每次训练写到一个带时间戳的新目录中，例如：

```text
outputs/task_level/task_level_small_lr_ordinal_pw_manual11_20260311_153000/
```

脚本逻辑会生成或尝试生成的核心产物包括：

- `best_model.pt`
- `config.json`
- `history.json`
- `test_results.json`
- `training_curves.png`

当前仓库中该示例目录已跟踪的文件是：

- `config.json`
- `history.json`
- `test_results.json`
- `threshold_calibration.json`
- `training_curves.png`

其中 `best_model.pt` 没有出现在仓库快照里，原因是 `.gitignore` 对 `*.pt` 和 `outputs/*.pt` 做了忽略。

### 8.6 示例训练快照

对示例目录 `task_level_small_lr_ordinal_pw_manual11_20260311_153000`：

- 训练历史记录长度：`51` 个 epoch
- `val_f1_macro` 最优值：`0.5983197353433288`
- 最优 epoch：`31`

这说明本次运行在 100 个最大 epoch 之前就因为早停结束了。

## 9. 评估与结果读取

### 9.1 评估脚本会输出什么

`scripts/train_task_level.py` 在加载最佳模型后，会对以下 4 个分布评估：

- `Val (同分布)`
- `Test1 (新被试+旧题)`
- `Test2 (旧被试+新题)`
- `Test3 (新被试+新题)`

每个分布都会输出：

- `loss`
- `accuracy`
- `f1_weighted`
- `f1_macro`
- `spearman`
- `spearman_p`
- `predictions`
- `labels`
- `probabilities`

### 9.2 指标如何理解

- `accuracy`
  - 标准分类准确率。

- `f1_weighted`
  - 按类别样本数加权的 F1。
  - 受大类影响更明显。

- `f1_macro`
  - 各类 F1 简单平均。
  - 对少数类更敏感。
  - 当前主线也用它作为早停指标。

- `spearman`
  - 把标签当有序等级后，计算预测类别与真实类别的等级相关。
  - 这是序数任务很有用的辅助指标。

### 9.3 示例结果快照

来自 `outputs/task_level/task_level_small_lr_ordinal_pw_manual11_20260311_153000/test_results.json` 的基线结果如下：

- `Val (同分布)`
  - `loss = 0.3746610547487552`
  - `accuracy = 0.675`
  - `f1_weighted = 0.6791857556285867`
  - `f1_macro = 0.5983197353433288`
  - `spearman = 0.5256044616040819`

- `Test1 (新被试+旧题)`
  - `loss = 0.37163570813006824`
  - `accuracy = 0.5771929824561404`
  - `f1_weighted = 0.6053086285501381`
  - `f1_macro = 0.5087712055890539`
  - `spearman = 0.4405717438552628`

- `Test2 (旧被试+新题)`
  - `loss = 0.3747178919258572`
  - `accuracy = 0.529`
  - `f1_weighted = 0.5551803758968028`
  - `f1_macro = 0.47368931112423146`
  - `spearman = 0.4336318428916295`

- `Test3 (新被试+新题)`
  - `loss = 0.3844498321413994`
  - `accuracy = 0.47368421052631576`
  - `f1_weighted = 0.4932531764280974`
  - `f1_macro = 0.4249741258013657`
  - `spearman = 0.350653080258356`

从这个快照可以看出：

- 同分布验证集表现最好。
- 跨被试、跨任务和双重泛化都会带来明显下降。
- `Test3` 是最困难的分布。

### 9.4 全面指标对比脚本 `build_task_embedding_report.py`

仓库里还有一个专门汇总和对比指标的脚本：

```bash
python scripts/build_task_embedding_report.py
```

它支持 3 类模式：

- 消融对比模式
  - `--baseline-results`
  - `--task-results`
  - 可选 `--random-results`
  - 用于比较 `Baseline / Baseline + task / 随机 / 差值`

- 单模型模式
  - `--single-results`
  - 可选 `--encoder-name`
  - 用于导出单个模型在各分布上的完整指标

- 多模型模式
  - `--multi-results`
  - 用于把多个运行目录汇总成编码器对比报表

当前脚本的统一指标列定义在 `METRIC_COLUMNS` 中，完整列表为：

- `ACC`
- `Precision Macro`
- `Precision Weighted`
- `Recall Macro`
- `Recall Weighted`
- `F1 Macro`
- `Spearman`
- `F1 Weight`
- `G-Mean`
- `MCC`
- `QWK`
- `AUC-PR`

### 9.5 指标含义与计算口径

这些指标并不是都来自 `test_results.json` 原始字段，有一部分是 `build_task_embedding_report.py` 二次计算出来的。当前脚本的计算口径如下：

- `ACC`
  - `accuracy_score(labels, preds)`
  - 普通准确率。

- `Precision Macro`
  - `precision_score(..., average="macro")`
  - 各类精确率简单平均。

- `Precision Weighted`
  - `precision_score(..., average="weighted")`
  - 按类别样本数加权的精确率。

- `Recall Macro`
  - `recall_score(..., average="macro")`
  - 各类召回率简单平均。

- `Recall Weighted`
  - `recall_score(..., average="weighted")`
  - 按类别样本数加权的召回率。

- `F1 Macro`
  - `f1_score(..., average="macro")`
  - 当前主线最重要的类均衡指标，也是训练脚本的早停指标。

- `Spearman`
  - `spearmanr(labels, preds)`
  - 直接用离散类别计算等级相关。

- `F1 Weight`
  - `f1_score(..., average="weighted")`
  - 脚本里沿用了报表中的命名 `F1 Weight`。

- `G-Mean`
  - 先计算各类别召回率，再取几何平均。
  - 当前实现里只要某一类召回率为 0，`G-Mean` 就会直接为 0。

- `MCC`
  - `matthews_corrcoef(labels, preds)`
  - 多分类 Matthews 相关系数。

- `QWK`
  - `cohen_kappa_score(labels, preds, weights="quadratic")`
  - 二次加权 Kappa，适合有序分类。

- `AUC-PR`
  - 先把真实标签做 one-vs-rest 二值化，再用 `average_precision_score(..., average="macro")`
  - 使用的是宏平均 PR-AUC。

### 9.6 文档中采用的全面指标结果来源

为了把“脚本中的全部指标”和“项目里已有的对应结果”一起放进文档，下面两个现成报表被用作附录 C 的数据来源：

- 任务嵌入对比报表：
  - `outputs/task_level/task_embedding_comparison_20260305_210442_pr_recall_macro_weighted.xlsx`

- 编码器对比报表：
  - `outputs/task_level/encoder_comparison_20260317_110456.xlsx`

附录 C 会把这两个报表中的完整指标结果逐条展开，便于直接查阅，不必再手动打开 Excel。

## 10. 阈值校准

### 10.1 为什么还需要校准

序数模型输出的是累计概率判断，而不是直接训练一个固定 softmax 类别边界。默认阈值常常设为 `0.5` 或手工给定值，但它未必是当前模型在验证集上的最佳决策边界。

因此仓库提供了 `scripts/calibrate_ordinal_thresholds.py`，用验证集搜索更优阈值。

### 10.2 输入与输出

脚本输入：

- 一个运行目录
- 或直接给 `test_results.json`

脚本会：

1. 读取 `Val` 集的 `labels` 和 `probabilities`。
2. 先把类别概率转回累计概率。
3. 在给定网格内枚举全部阈值组合。
4. 以 `f1_macro` 最大为首要目标。
5. 若 `f1_macro` 相同，再比较 `accuracy`。

可选输出：

- 控制台打印校准前后对比
- 报告 JSON
- 回写配置中的 `training.ordinal_thresholds`

### 10.3 示例命令

```bash
python scripts/calibrate_ordinal_thresholds.py --results outputs/task_level/task_level_small_lr_ordinal_pw_manual11_20260311_153000
```

### 10.4 示例校准结果

来自 `threshold_calibration.json` 的最优阈值是：

```text
[0.46000000000000024, 0.5700000000000003]
```

验证集最佳指标变成：

- `accuracy = 0.695`
- `f1_macro = 0.6105400838021159`
- `f1_weighted = 0.6976843878047085`

相对基线的变化：

- `Val`
  - `f1_macro: +0.012220348458787056`
  - `accuracy: +0.02`

- `Test1`
  - `f1_macro: +0.0048543459293922675`
  - `accuracy: +0.021929824561403466`

- `Test2`
  - `f1_macro: -0.028073092625678864`
  - `accuracy: +0.0010000000000000009`

- `Test3`
  - `f1_macro: -0.014273274310145356`
  - `accuracy: +0.00877192982456143`

### 10.5 为什么只在部分分布上提升

这是当前任务级序数主线里一个非常值得注意的现象。

阈值校准的优化目标是：

- 用 `Val` 集最大化 `f1_macro`

因此：

- 它通常会改善 `Val`
- 有时会顺带改善与 `Val` 更接近的 `Test1`
- 但不保证改善跨任务更明显的 `Test2`
- 也不保证改善最难的 `Test3`

换句话说，阈值校准本身也是一种“针对验证分布调边界”的过程，而不是对所有泛化分布都单调增益的万能步骤。

## 11. 本地复现与验证

### 11.1 推荐执行顺序

```bash
python scripts/preprocess_data.py --data_dir data/gaze_trajectory_data --output_dir data/processed --subject_normalize
python scripts/train_task_level.py --config configs/task_level_small_lr_ordinal_pw_manual11.json
python scripts/calibrate_ordinal_thresholds.py --results outputs/task_level/task_level_small_lr_ordinal_pw_manual11_20260311_153000
python -m unittest discover -s tests -p "test_*.py" -v
```

### 11.2 当前仓库中已验证的轻量步骤

本次文档整理过程中已本地验证：

- `python scripts/preprocess_data.py --help`
- `python scripts/train_task_level.py --help`
- `python scripts/calibrate_ordinal_thresholds.py --results outputs/task_level/task_level_small_lr_ordinal_pw_manual11_20260311_153000`
- `python -m unittest discover -s tests -p "test_*.py" -v`

其中：

- `calibrate_ordinal_thresholds.py` 能复现现有 `threshold_calibration.json` 的阈值与指标变化趋势。
- `unittest discover` 当前通过 3 个测试。

### 11.3 为什么这里不把全量训练结果重新跑一遍

本文是文档交付，不是重新产出一套实验结果。当前仓库已经存在：

- 对应配置文件
- 对应训练历史
- 对应评估结果
- 对应阈值校准结果

因此正文中的数值以现有产物为准，重点是把“真实链路和真实实现”讲清楚，而不是再次执行一轮长时间训练。

## 附录 A：任务级序数模型内部变体

### A.1 `manual11` 主线

代表配置：

- `configs/task_level_small_lr_ordinal_pw_manual11.json`
- `configs/task_level_ordinal_manual11_bilstm.json`
- 以及对应的 `cnn1d/rnn/lstm/gru` 版本

特点：

- `use_task_embedding = true`
- 默认独立任务嵌入
- 是否条件化池化由配置决定，标准 `manual11` 主线通常不开启
- 重点在“固定任务嵌入策略 + 不同片段编码器”的横向对比

### A.2 `indep_cond`

代表配置：

- `configs/task_level_ordinal_indep_cond_bilstm.json`
- 以及 `transformer/cnn1d/rnn/lstm/gru` 对应版本

特点：

- `task_embedding_type = "independent"`
- `use_conditional_pooling = true`
- 任务嵌入既会在预测头前拼接，也会参与注意力聚合打分

批量运行脚本：

- `scripts/run_indep_cond_ablation.sh`

### A.3 `mlp_only`

代表配置：

- `configs/task_level_ordinal_mlp_only.json`

特点：

- `task_embedding_type = "mlp"`
- `use_conditional_pooling = false`
- 任务条件先通过 MLP 编成一个低维向量，再在预测头前与主干表示拼接

### A.4 `mlp_cond`

代表配置：

- `configs/task_level_ordinal_mlp_cond.json`

特点：

- `task_embedding_type = "mlp"`
- `use_conditional_pooling = true`
- MLP 任务嵌入同时进入注意力聚合和预测头融合

### A.5 编码器消融

任务级主线还保留了编码器消融脚本：

- `scripts/run_encoder_ablation.sh`

它会对以下编码器做横向对比：

- `transformer`
- `cnn1d`
- `rnn`
- `lstm`
- `gru`
- `bilstm`

对应配置则是：

- `configs/task_level_small_lr_ordinal_pw_manual11.json`
- `configs/task_level_ordinal_manual11_cnn1d.json`
- `configs/task_level_ordinal_manual11_rnn.json`
- `configs/task_level_ordinal_manual11_lstm.json`
- `configs/task_level_ordinal_manual11_gru.json`
- `configs/task_level_ordinal_manual11_bilstm.json`

## 附录 B：常见坑与注意事项

### B.1 任务级标签是 0-based，被试级类别是 1-based

这是最常见也最容易写错的地方：

- `category`：`1 / 2 / 3`
- `task_label`：`0 / 1 / 2`

当前任务级序数训练必须使用后者。

### B.2 训练脚本内置了自己的 `split_2x2`

仓库里同时存在：

- `src/data/split_strategy.py`
- `scripts/train_task_level.py` 内部的 `split_2x2`

当前任务级主线实际走的是后者，不能混淆。

### B.3 `python -m unittest` 默认不会发现当前测试

直接运行：

```bash
python -m unittest
```

在当前仓库里会得到 `Ran 0 tests`。需要显式使用：

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

### B.4 历史输出目录里的配置快照可能落后于当前代码

例如示例目录 `task_level_small_lr_ordinal_pw_manual11_20260311_153000/config.json` 中没有记录当前代码已强制要求的 `segment_encoder_type` 字段，但当前仓库中的正式配置文件已经补齐了它。

所以：

- 用历史目录看指标和产物没问题。
- 用历史目录里的 `config.json` 反推当前代码结构时要格外小心。
- 结构解释以 `configs/` 和 `src/` 里的当前实现为准。

### B.5 示例目录里看不到 `best_model.pt` 不代表训练脚本不保存它

当前 `.gitignore` 忽略了 `*.pt` 和 `outputs/*.pt`，因此仓库快照通常只保留文本产物和图片，不保留大模型权重文件。

### B.6 `max_segments=25` 在任务级数据集里不是硬截断

很多人会自然地认为配置里写了 `max_segments=25`，数据集就会把片段数裁成 25。当前任务级实现并不是这样：

- `max_seq_len=300` 才是明确的时间维硬截断。
- 片段数目前按真实值保留。
- batch 级对齐由 `task_level_collate_fn()` 动态完成。

### B.7 `--workers` 的帮助说明和真实默认行为要以代码为准

`scripts/preprocess_data.py` 中 `workers` 参数的真实默认值是 `0`，代码会把它解释为“自动使用 `multiprocessing.cpu_count()`”。阅读帮助文本时若看到旧描述，应以源码逻辑为准。

## 附录 C：全面指标对比脚本的完整结果快照

### C.1 指标缩写对照

为了让完整结果部分更紧凑，下面统一使用这些缩写：

- `ACC`：准确率
- `P-Macro`：宏平均精确率
- `P-Weight`：加权精确率
- `R-Macro`：宏平均召回率
- `R-Weight`：加权召回率
- `F1-Macro`：宏平均 F1
- `Spearman`：等级相关
- `F1-Weight`：加权 F1
- `G-Mean`：召回率几何平均
- `MCC`：Matthews 相关系数
- `QWK`：二次加权 Kappa
- `AUC-PR`：宏平均 PR-AUC

### C.2 任务嵌入对比完整结果

结果来源：

- `outputs/task_level/task_embedding_comparison_20260305_210442_pr_recall_macro_weighted.xlsx`

- 同分布 (Val)
  - 随机: ACC 0.315000，P-Macro 0.352780，P-Weight 0.470104，R-Macro 0.382736，R-Weight 0.315000，F1-Macro 0.308578，Spearman 0.098482，F1-Weight 0.328606，G-Mean 0.365040，MCC 0.039212，QWK 0.090751，AUC-PR 0.342808
  - Baseline: ACC 0.635000，P-Macro 0.476923，P-Weight 0.499077，R-Macro 0.380028，R-Weight 0.635000，F1-Macro 0.340387，Spearman 0.229505，F1-Weight 0.511869，G-Mean 0.000000，MCC 0.167592，QWK 0.110840，AUC-PR 0.410633
  - Baseline + task: ACC 0.415000，P-Macro 0.358010，P-Weight 0.461534，R-Macro 0.393926，R-Weight 0.415000，F1-Macro 0.358410，Spearman 0.165233，F1-Weight 0.427814，G-Mean 0.371879，MCC 0.027127，QWK 0.149338，AUC-PR 0.412166
  - 差值/比例: ACC -34.65%，P-Macro -24.93%，P-Weight -7.52%，R-Macro +3.66%，R-Weight -34.65%，F1-Macro +5.30%，Spearman -28.00%，F1-Weight -16.42%，G-Mean nan，MCC -83.81%，QWK +34.73%，AUC-PR +0.37%

- 不同分布1 (Test1)
  - 随机: ACC 0.353509，P-Macro 0.336858，P-Weight 0.575334，R-Macro 0.335509，R-Weight 0.353509，F1-Macro 0.289764，Spearman 0.023819，F1-Weight 0.409593，G-Mean 0.333655，MCC 0.005445，QWK 0.019960，AUC-PR 0.334884
  - Baseline: ACC 0.735088，P-Macro 0.450618，P-Weight 0.641995，R-Macro 0.345372，R-Weight 0.735088，F1-Macro 0.308298，Spearman 0.116945，F1-Weight 0.632849，G-Mean 0.000000，MCC 0.092733，QWK 0.045178，AUC-PR 0.384154
  - Baseline + task: ACC 0.492105，P-Macro 0.408415，P-Weight 0.626022，R-Macro 0.446915，R-Weight 0.492105，F1-Macro 0.397031，Spearman 0.205904，F1-Weight 0.535402，G-Mean 0.443744，MCC 0.116469，QWK 0.191562，AUC-PR 0.400024
  - 差值/比例: ACC -33.05%，P-Macro -9.37%，P-Weight -2.49%，R-Macro +29.40%，R-Weight -33.05%，F1-Macro +28.78%，Spearman +76.07%，F1-Weight -15.40%，G-Mean nan，MCC +25.60%，QWK +324.01%，AUC-PR +4.13%

- 不同分布2 (Test2)
  - 随机: ACC 0.294000，P-Macro 0.297036，P-Weight 0.512188，R-Macro 0.270474，R-Weight 0.294000，F1-Macro 0.242323，Spearman -0.088236，F1-Weight 0.347107，G-Mean 0.268625，MCC -0.065963，QWK -0.080801，AUC-PR 0.323605
  - Baseline: ACC 0.691000，P-Macro 0.330877，P-Weight 0.545881，R-Macro 0.350727，R-Weight 0.691000，F1-Macro 0.320847，Spearman 0.106193，F1-Weight 0.600219，G-Mean 0.000000，MCC 0.041230，QWK 0.073257，AUC-PR 0.402423
  - Baseline + task: ACC 0.366000，P-Macro 0.383536，P-Weight 0.545154，R-Macro 0.376991，R-Weight 0.366000，F1-Macro 0.306771，Spearman 0.182970，F1-Weight 0.398397，G-Mean 0.317554，MCC 0.011234，QWK 0.134465，AUC-PR 0.422827
  - 差值/比例: ACC -47.03%，P-Macro +15.92%，P-Weight -0.13%，R-Macro +7.49%，R-Weight -47.03%，F1-Macro -4.39%，Spearman +72.30%，F1-Weight -33.62%，G-Mean nan，MCC -72.75%，QWK +83.55%，AUC-PR +5.07%

- 不同分布3 (Test3)
  - 随机: ACC 0.356140，P-Macro 0.356753，P-Weight 0.530860，R-Macro 0.393692，R-Weight 0.356140，F1-Macro 0.324975，Spearman 0.151331，F1-Weight 0.387916，G-Mean 0.390216，MCC 0.041435，QWK 0.138494，AUC-PR 0.344262
  - Baseline: ACC 0.712281，P-Macro 0.468514，P-Weight 0.619397，R-Macro 0.385423，R-Weight 0.712281，F1-Macro 0.368927，Spearman 0.257991，F1-Weight 0.623856，G-Mean 0.000000，MCC 0.203168，QWK 0.168623，AUC-PR 0.427998
  - Baseline + task: ACC 0.417544，P-Macro 0.377233，P-Weight 0.554600，R-Macro 0.413968，R-Weight 0.417544，F1-Macro 0.339784，Spearman 0.237688，F1-Weight 0.433729，G-Mean 0.328644，MCC 0.083751，QWK 0.180392，AUC-PR 0.442719
  - 差值/比例: ACC -41.38%，P-Macro -19.48%，P-Weight -10.46%，R-Macro +7.41%，R-Weight -41.38%，F1-Macro -7.90%，Spearman -7.87%，F1-Weight -30.48%，G-Mean nan，MCC -58.78%，QWK +6.98%，AUC-PR +3.44%

### C.3 编码器对比完整结果

结果来源：

- `outputs/task_level/encoder_comparison_20260317_110456.xlsx`

- 同分布 (Val)
  - transformer: ACC 0.540000，P-Macro 0.494429，P-Weight 0.573998，R-Macro 0.498161，R-Weight 0.540000，F1-Macro 0.481755，Spearman 0.432121，F1-Weight 0.549359，G-Mean 0.491682，MCC 0.184422，QWK 0.413073，AUC-PR 0.458022
  - cnn1d: ACC 0.495000，P-Macro 0.447746，P-Weight 0.534779，R-Macro 0.413507，R-Weight 0.495000，F1-Macro 0.396055，Spearman 0.285662，F1-Weight 0.492993，G-Mean 0.373969，MCC 0.074701，QWK 0.253923，AUC-PR 0.454311
  - rnn: ACC 0.535000，P-Macro 0.477837，P-Weight 0.562129，R-Macro 0.541074，R-Weight 0.535000，F1-Macro 0.492661，Spearman 0.358499，F1-Weight 0.539638，G-Mean 0.518991，MCC 0.209480，QWK 0.353373，AUC-PR 0.474491
  - lstm: ACC 0.600000，P-Macro 0.528669，P-Weight 0.612295，R-Macro 0.500018，R-Weight 0.600000，F1-Macro 0.506838，Spearman 0.414748，F1-Weight 0.600809，G-Mean 0.475362，MCC 0.241879，QWK 0.402778，AUC-PR 0.493990
  - gru: ACC 0.645000，P-Macro 0.562832，P-Weight 0.657323，R-Macro 0.579129，R-Weight 0.645000，F1-Macro 0.569705，Spearman 0.500048，F1-Weight 0.650195，G-Mean 0.557199，MCC 0.361434，QWK 0.485861，AUC-PR 0.575100
  - bilstm: ACC 0.560000，P-Macro 0.519242，P-Weight 0.612398，R-Macro 0.533601，R-Weight 0.560000，F1-Macro 0.511775，Spearman 0.435862，F1-Weight 0.577996，G-Mean 0.531805，MCC 0.253543，QWK 0.419280，AUC-PR 0.519701

- 不同分布1 (Test1)
  - transformer: ACC 0.528947，P-Macro 0.433696，P-Weight 0.645998，R-Macro 0.496872，R-Weight 0.528947，F1-Macro 0.431268，Spearman 0.319146，F1-Weight 0.556703，G-Mean 0.465013，MCC 0.181340，QWK 0.290405，AUC-PR 0.431249
  - cnn1d: ACC 0.493860，P-Macro 0.369889，P-Weight 0.595443，R-Macro 0.403391，R-Weight 0.493860，F1-Macro 0.361260，Spearman 0.197722，F1-Weight 0.522230，G-Mean 0.330293，MCC 0.074016，QWK 0.175506，AUC-PR 0.377180
  - rnn: ACC 0.485965，P-Macro 0.408394，P-Weight 0.626695，R-Macro 0.466347，R-Weight 0.485965，F1-Macro 0.401939，Spearman 0.262585，F1-Weight 0.523389，G-Mean 0.455645，MCC 0.135311，QWK 0.242820，AUC-PR 0.400752
  - lstm: ACC 0.572807，P-Macro 0.448149，P-Weight 0.655937，R-Macro 0.504302，R-Weight 0.572807，F1-Macro 0.458464，Spearman 0.310010，F1-Weight 0.600104，G-Mean 0.490244，MCC 0.196390，QWK 0.297144，AUC-PR 0.438779
  - gru: ACC 0.534211，P-Macro 0.420651，P-Weight 0.628779，R-Macro 0.473096，R-Weight 0.534211，F1-Macro 0.426369，Spearman 0.309039，F1-Weight 0.564101，G-Mean 0.453583，MCC 0.145520，QWK 0.292543，AUC-PR 0.422538
  - bilstm: ACC 0.515789，P-Macro 0.428926，P-Weight 0.638259，R-Macro 0.490973，R-Weight 0.515789，F1-Macro 0.426897，Spearman 0.296017，F1-Weight 0.546462，G-Mean 0.474486，MCC 0.163084，QWK 0.275068，AUC-PR 0.420452

- 不同分布2 (Test2)
  - transformer: ACC 0.320000，P-Macro 0.501779，P-Weight 0.578223，R-Macro 0.404954，R-Weight 0.320000，F1-Macro 0.260460，Spearman 0.348740，F1-Weight 0.321748，G-Mean 0.241635，MCC 0.027120，QWK 0.184901，AUC-PR 0.459598
  - cnn1d: ACC 0.458000，P-Macro 0.418827，P-Weight 0.569925，R-Macro 0.381982，R-Weight 0.458000，F1-Macro 0.317905，Spearman 0.216138，F1-Weight 0.471767，G-Mean 0.241843，MCC 0.018534，QWK 0.163594，AUC-PR 0.389328
  - rnn: ACC 0.442000，P-Macro 0.447287，P-Weight 0.591511，R-Macro 0.416734，R-Weight 0.442000，F1-Macro 0.350825，Spearman 0.249534，F1-Weight 0.468395，G-Mean 0.339153，MCC 0.063833，QWK 0.191722，AUC-PR 0.424759
  - lstm: ACC 0.415000，P-Macro 0.628741，P-Weight 0.670002，R-Macro 0.411538，R-Weight 0.415000，F1-Macro 0.307227，Spearman 0.267903，F1-Weight 0.430317，G-Mean 0.247716，MCC 0.058277，QWK 0.175476，AUC-PR 0.454506
  - gru: ACC 0.378000，P-Macro 0.481237，P-Weight 0.592809，R-Macro 0.443949，R-Weight 0.378000，F1-Macro 0.328225，Spearman 0.369279，F1-Weight 0.394664，G-Mean 0.338163，MCC 0.078438，QWK 0.236627，AUC-PR 0.466790
  - bilstm: ACC 0.363000，P-Macro 0.465122，P-Weight 0.584635，R-Macro 0.413785，R-Weight 0.363000，F1-Macro 0.283412，Spearman 0.312370，F1-Weight 0.373090，G-Mean 0.247772，MCC 0.050946，QWK 0.182067，AUC-PR 0.448215

- 不同分布3 (Test3)
  - transformer: ACC 0.352632，P-Macro 0.369335，P-Weight 0.532746，R-Macro 0.413176，R-Weight 0.352632，F1-Macro 0.259661，Spearman 0.376972，F1-Weight 0.328504，G-Mean 0.150930，MCC 0.086324，QWK 0.198105，AUC-PR 0.454670
  - cnn1d: ACC 0.454386，P-Macro 0.367811，P-Weight 0.559898，R-Macro 0.422113，R-Weight 0.454386，F1-Macro 0.329621，Spearman 0.281272，F1-Weight 0.457363，G-Mean 0.214026，MCC 0.109833，QWK 0.199595，AUC-PR 0.409829
  - rnn: ACC 0.470175，P-Macro 0.421693，P-Weight 0.582317，R-Macro 0.427560，R-Weight 0.470175，F1-Macro 0.350027，Spearman 0.269794，F1-Weight 0.477076，G-Mean 0.270555，MCC 0.115695，QWK 0.203197，AUC-PR 0.421364
  - lstm: ACC 0.449123，P-Macro 0.439058，P-Weight 0.591663，R-Macro 0.430357，R-Weight 0.449123，F1-Macro 0.319956，Spearman 0.307954，F1-Weight 0.446326，G-Mean 0.171291，MCC 0.126333，QWK 0.203442，AUC-PR 0.451869
  - gru: ACC 0.408772，P-Macro 0.460147，P-Weight 0.599280，R-Macro 0.451773，R-Weight 0.408772，F1-Macro 0.331788，Spearman 0.367668，F1-Weight 0.405064，G-Mean 0.296043，MCC 0.143866，QWK 0.232689，AUC-PR 0.482478
  - bilstm: ACC 0.401754，P-Macro 0.542957，P-Weight 0.624417，R-Macro 0.430828，R-Weight 0.401754，F1-Macro 0.300095，Spearman 0.346883，F1-Weight 0.392666，G-Mean 0.205069，MCC 0.121762，QWK 0.204400，AUC-PR 0.464834
