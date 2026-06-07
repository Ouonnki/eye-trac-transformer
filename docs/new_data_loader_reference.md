# 新数据推理 Loader 参考说明（不含舒尔特）

本文档用于后续编写 `新数据` 目录的只推理 loader。范围只包含三类任务：

- 复杂问题解决任务
- 情景意识任务
- 找不同任务

舒尔特方格任务不纳入本次测试推理。loader 的目标是生成高中低三分类预测及概率，不使用真实标签，不计算评估指标，也不通过伪造标签绕过现有数据集限制。

## 1. 总体约定

### 1.1 数据目录

新数据根目录：

```text
新数据/
├── 复杂问题解决任务（4，0，0，0，0）/
├── 情景意识任务（4，1，1，0，1）/
├── 找不同任务（3，1，0，1，0）/
└── 舒尔特方格任务/              # 本次排除
```

本次 loader 只扫描前三个目录。目录名中的五元组对应模型输入的 `task_conditions`。

### 1.2 推理目标

- 输出类别：高中低三分类。
- 输出内容：类别、中文标签、三类概率、样本元数据和 warning。
- 不读取或生成 `task_label`。
- 不使用情景意识结果表、找不同行为结果、舒尔特总分表作为评估标签。
- 不计算 accuracy、F1、confusion matrix 等评估指标。

### 1.3 采样率

原训练数据按 60Hz 采集；本批新数据按 120Hz 采集。为减少推理分布偏移，loader 默认将 120Hz 显式降采样到 60Hz，再提取 7 维眼动特征。

CSV 中 `Recording timestamp` 的相邻有效 gaze 点差值约为 `8333`，应按微秒理解。降采样目标间隔为约 `16667` 微秒。

## 2. 任务说明

| 任务 | 数据结构 | 缺失情况 | 切片规则 | `task_id` | `task_conditions` |
|---|---|---|---|---:|---|
| 复杂问题解决任务 | 41 个眼动 CSV | 缺 `0429322` | 每个 `TaskStartN` 到其后的第一个 `MouseEvent` | 101 | `[4, 0, 0, 0, 0]` |
| 情景意识任务 | 41 个眼动 CSV，另有结果/常模 CSV | 缺 `0429332` | 每个 `TaskStartN` 到连续 `KeyboardEvent=Enter` 的最后一个 | 102 | `[4, 1, 1, 0, 1]` |
| 找不同任务 | 41 个眼动 CSV + 41 个行为 XLSX | 缺 `0429332` | 每个 `TaskStartN` 到该 trial 内最后一个有效 `MouseEvent`；`TaskEndN` 只作校验边界 | 103 | `[3, 1, 0, 1, 0]` |

三类任务的被试并集为 42 人，三类都完整的交集为 40 人。loader 默认允许部分任务缺失，并在输出中记录 `missing_tasks`。

## 3. 数据格式

### 3.1 眼动 CSV

三类任务的眼动 CSV 表头一致，关键列如下：

| 列名 | 用途 |
|---|---|
| `Recording timestamp` | 主时间戳，单位按微秒处理 |
| `Event` | 事件类型，如 `TaskStartN`、`TaskEndN`、`MouseEvent`、`KeyboardEvent` |
| `Event value` | 事件值，如 `Down, Left`、`Up, Left`、`Enter` |
| `Gaze point X` | gaze X 坐标 |
| `Gaze point Y` | gaze Y 坐标 |

读取规则：

- 编码使用 `utf-8-sig`。
- 事件行只用于确定片段边界。
- 事件行通常没有 gaze 坐标，不能作为眼动点。
- 眼动点必须同时具备有效 `Gaze point X` 和 `Gaze point Y`。
- 坐标异常处理应显式记录 warning，不做静默吞错。

### 3.2 找不同行为 XLSX

找不同目录下包含行为 XLSX，结构包括题目基本信息、被试完成信息、鼠标点击信息。该文件只作为辅助元数据来源，不作为高中低真实标签，也不用于评估推理结果。

默认推理 loader 不依赖行为 XLSX。若后续需要校验找不同 CSV 的鼠标事件，可读取行为 XLSX 中的点击数量、trialID 和结束原因作为一致性检查。

## 4. 事件切片规则

### 4.1 通用边界处理

每个 segment 由一个开始时间和一个结束时间定义：

- 开始时间来自 `TaskStartN`。
- 结束时间按任务专属关键事件规则确定。
- 只取 `[start_time, end_time]` 内的有效 gaze 点。
- 如果窗口内有效 gaze 点少于 2 个，该 segment 应跳过并记录 warning。
- `TaskEndN` 可用于校验，但不替代已确认的关键事件规则。

### 4.2 复杂问题解决任务

规则：每个 `TaskStartN` 到其后的第一个 `MouseEvent`。

实现要点：

- 对每个 `TaskStartN`，在时间上向后查找第一个 `Event == "MouseEvent"`。
- 不要求该 `MouseEvent` 的 `Event value` 必须是 `Down, Left` 或 `Up, Left`，但应记录实际值。
- 若在下一个 `TaskStart` 或文件结束前找不到 `MouseEvent`，该 trial 跳过并记录 warning。

### 4.3 情景意识任务

规则：每个 `TaskStartN` 到连续 `KeyboardEvent=Enter` 的最后一个。

实现要点：

- 对每个 `TaskStartN`，在该 trial 内查找 `Event == "KeyboardEvent"` 且 `Event value == "Enter"` 的事件序列。
- 连续 Enter 的定义需要实现时明确，默认按相邻 Enter 间隔不超过一个小阈值识别为同一连续段。
- 结束时间取第一个连续 Enter 段的最后一个 Enter。
- 若没有 Enter，跳过该 trial 并记录 warning。

### 4.4 找不同任务

规则：每个 `TaskStartN` 到该 trial 内最后一个有效 `MouseEvent`；`TaskEndN` 只作校验边界。

实现要点：

- trial 范围优先由 `TaskStartN` 到同编号 `TaskEndN` 确定。
- 在该范围内查找 `Event == "MouseEvent"`。
- 结束时间取该 trial 内最后一个 `MouseEvent`。
- 如果没有 `TaskEndN`，可退化为到下一个 `TaskStart` 前，但必须记录 warning。
- 如果 trial 内没有 `MouseEvent`，跳过该 trial 并记录 warning。

## 5. 降采样与特征

### 5.1 120Hz 到 60Hz

默认降采样策略：

1. 对每个 segment 内有效 gaze 点按 `Recording timestamp` 升序排序。
2. 以 segment 第一个有效 gaze 时间戳为锚点。
3. 每约 `16667` 微秒划分一个时间桶。
4. 每个桶选择最接近桶中心或桶起点的一个有效 gaze 点。
5. 保持输出时间戳单调递增。

降采样后，中位 `dt` 应接近 `16.67ms`。如果中位值明显偏离，应停止并暴露错误或 warning，不能静默继续。

### 5.2 7 维特征

每个 segment 输出 `np.ndarray`，形状为 `(seq_len, 7)`：

| 维度 | 名称 | 说明 |
|---:|---|---|
| 0 | `x` | X 坐标归一化到屏幕宽度 |
| 1 | `y` | Y 坐标归一化到屏幕高度 |
| 2 | `dt` | 与前一点的时间差，单位毫秒 |
| 3 | `velocity` | 瞬时速度 |
| 4 | `acceleration` | 瞬时加速度 |
| 5 | `direction` | 移动方向，归一化到 `[-1, 1]` |
| 6 | `direction_change` | 方向变化量，归一化到 `[0, 1]` |

特征计算应复用现有 `scripts/preprocess_data.py` 中的 `extract_features` 思路，避免新旧推理分布不一致。

### 5.3 标准化

如果当前 checkpoint 对应 `processed_data_subject_norm.pkl`，新数据推理也应做被试内标准化：

- 只标准化 `dt`、`velocity`、`acceleration`。
- `x`、`y`、`direction`、`direction_change` 保持原策略。
- 每个被试使用该被试所有可用任务和 segment 统计均值与标准差。
- 不使用训练集统计量，除非后续明确保存并加载训练时 normalizer。

## 6. Loader 输出接口

### 6.1 样本字段

loader 输出的单个任务样本建议包含：

```python
{
    "subject_id": "0429324",
    "task_id": 101,
    "task_name": "复杂问题解决任务",
    "task_conditions": [4, 0, 0, 0, 0],
    "segments": [np.ndarray],
    "source_files": [".../0429324_复杂能力测试1.csv"],
    "missing_tasks": [],
    "warnings": [],
}
```

### 6.2 Inference dataset 字段

无标签 inference dataset 返回 batch 时应包含：

- `segments`
- `segment_mask`
- `segment_seq_mask`
- `task_conditions`
- `subject_id`
- `task_id`
- `task_name`

不得依赖 `task_label`。现有 `TaskLevelGazeDataset` 会跳过没有 `task_label` 的样本，因此本次推理需要单独实现 inference dataset 或对现有 dataset 做显式无标签模式扩展。

### 6.3 预测结果字段

预测输出建议包含：

```text
subject_id
task_id
task_name
pred_class
pred_label
prob_low
prob_mid
prob_high
warnings
```

`pred_class` 使用模型输出索引。`pred_label` 的中文映射建议为：

| `pred_class` | `pred_label` |
|---:|---|
| 0 | 低 |
| 1 | 中 |
| 2 | 高 |

## 7. 待确认问题与方案选择

| 问题 | 默认方案 | 备选方案 | 风险 |
|---|---|---|---|
| 120Hz 如何对齐旧 60Hz | 显式降采样到 60Hz | 保留 120Hz；或配置开关做对照 | 保留 120Hz 会改变 `dt`、速度和序列长度分布 |
| 被试任务缺失 | 允许部分任务推理并记录 `missing_tasks` | 只保留完整 40 人；或缺失即报错 | 只保留完整人群会丢弃可推理数据 |
| 新任务 ID | 使用 `101`、`102`、`103` | 复用旧题号；或不传 `task_id` | 复用旧题号会引入不可靠语义假设 |
| 情景意识连续 Enter 阈值 | 设置显式时间阈值并记录 | 取 trial 内最后一个 Enter；或取第一个 Enter | 不同规则会改变 segment 长度 |
| 找不同 MouseEvent 结束点 | 取 trial 内最后一个 MouseEvent | 取第一个 MouseEvent；或用行为 XLSX 点击表对齐 | 结束点不同会改变搜索过程覆盖范围 |
| 坐标屏幕尺寸 | 默认沿用 `1920x1080` | 从配置传入实际屏幕尺寸 | 错误屏幕尺寸会影响 `x/y` 归一化 |

实现时，除已确认默认方案外，其他方案必须通过显式配置选择。不要添加静默 fallback。

## 8. 主要陷阱

- 不能直接运行旧 `scripts/preprocess_data.py`。旧 loader 强依赖旧目录、旧 XLSX sheet 和标签表。
- 不能塞假标签绕过 dataset。推理路径必须原生支持无标签样本。
- 不能把 `Recording timestamp` 的约 `8333` 间隔当毫秒，应按微秒理解。
- 不能把事件行当眼动点。事件行通常没有 gaze 坐标。
- 不能把找不同行为 XLSX 当高中低真实标签。
- 不能在缺失任务、缺失关键事件、空 segment 时静默跳过；必须记录 warning 或显式失败。
- 不能新增“为了能跑”的边界规则、最大轮数、模拟成功路径或隐式降级逻辑。

## 9. 验证清单

后续实现 loader 时，至少覆盖以下检查：

- Manifest 测试：只纳入三类任务；42 人并集、40 人完整交集、缺失任务记录正确。
- Schema 测试：三类 CSV 表头一致，且关键列存在。
- Event 测试：三类任务按指定关键事件生成非空 segment。
- Sampling 测试：降采样后中位 `dt` 接近 `16.67ms`。
- Feature 测试：输出 shape 为 `(seq_len, 7)`，无 `NaN` 或 `Inf`。
- Inference 测试：无 `task_label` 时仍能构建 batch 并调用模型。
- Warning 测试：缺失任务、缺失关键事件、空 segment 都能在结果中显式体现。
