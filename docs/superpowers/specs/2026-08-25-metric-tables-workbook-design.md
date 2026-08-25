# 指标分表工作簿设计

## 目标

生成一个 Excel 工作簿，将四种方法在五个分类指标上的 Val、Test1、Test2、Test3 和 OOD 平均结果整理为论文三线表风格，便于核验、引用和截图。

## 数据范围

方法固定为：

1. Baseline
2. Baseline + Task Embedding
3. Ordinal
4. Ordinal + Task Embedding

指标固定为 ACC、Macro F1、G-Mean、AUC-PR 和 QWK。每个指标从对应完整实验的 `test_results.json` 中读取 Val、Test1、Test2、Test3；截图版本的完整 TAOA 数据使用 `outputs/task_level/task_level_ordinal_manual11_bilstm_20260312_193853/test_results.json`，不使用后来替换的 BiLSTM 结果。

## 工作簿结构

工作簿包含六个工作表：

- `ACC`
- `Macro F1`
- `G-Mean`
- `AUC-PR`
- `QWK`
- `Sources`

五个指标表采用相同结构：`Method | Val | Test1 | Test2 | Test3 | OOD Avg`。`OOD Avg` 由 Excel 公式 `AVERAGE(Test1:Test3)` 计算，显示三位小数。每张表采用黑白论文三线表视觉：仅保留顶部、表头底部和表格底部边框，不使用彩色填充或逐单元格网格线。

`Sources` 工作表记录方法名称、实验角色、结果文件路径、配置文件路径、编码器、预测头类型和任务嵌入开关，作为审计依据。

## 数据流

1. 从四个完整实验结果 JSON 读取标签、预测和概率。
2. 使用项目已有指标定义重新计算五项指标，避免依赖 JSON 中可能缺失的派生字段。
3. 将每个 split 的结果写入对应指标工作表。
4. 使用 Excel 公式生成 OOD Avg，并保留完整精度，单元格显示为三位小数。
5. 将原始来源和配置写入 `Sources` 工作表。

## 错误处理

任何来源缺少 Val、Test1、Test2、Test3，或标签、预测、概率数量不一致时，构建过程直接失败并报告具体来源；不生成占位值，也不静默跳过数据。

## 验证

- 核对 Macro F1 工作表与参考截图的全部显示值一致。
- 抽查每个指标至少一行的 OOD Avg，确认等于 Test1–Test3 的算术平均。
- 扫描工作簿公式错误。
- 渲染并检查六个工作表，确保标题、数值、来源路径和边框完整可见。

## 输出

最终仅交付一个 `.xlsx` 文件，保存到项目 `outputs` 下的独立输出目录，不修改原始实验结果或现有汇总工作簿。
