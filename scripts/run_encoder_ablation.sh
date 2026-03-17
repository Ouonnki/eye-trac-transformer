#!/usr/bin/env bash
set -euo pipefail

cd /home/sysadmin/eye-trac-transformer
RUNROOT="${1:-outputs/task_level/encoder_ablation_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUNROOT"

export PYTHONUNBUFFERED=1

echo "[$(date '+%F %T')] 编码器对比流水线启动" | tee -a "$RUNROOT/pipeline.log"

ENCODERS=("transformer" "cnn1d" "rnn" "lstm" "gru" "bilstm")
CONFIGS=(
  "configs/task_level_small_lr_ordinal_pw_manual11.json"
  "configs/task_level_ordinal_manual11_cnn1d.json"
  "configs/task_level_ordinal_manual11_rnn.json"
  "configs/task_level_ordinal_manual11_lstm.json"
  "configs/task_level_ordinal_manual11_gru.json"
  "configs/task_level_ordinal_manual11_bilstm.json"
)
EXPERIMENTS=(
  "task_level_ordinal_manual11_transformer"
  "task_level_ordinal_manual11_cnn1d"
  "task_level_ordinal_manual11_rnn"
  "task_level_ordinal_manual11_lstm"
  "task_level_ordinal_manual11_gru"
  "task_level_ordinal_manual11_bilstm"
)

RUN_DIRS=()

for i in "${!ENCODERS[@]}"; do
  enc="${ENCODERS[$i]}"
  cfg="${CONFIGS[$i]}"
  exp="${EXPERIMENTS[$i]}"

  echo "[$(date '+%F %T')] 训练开始：${enc}" | tee -a "$RUNROOT/pipeline.log"
  conda run -n cadt python -u scripts/train_task_level.py \
    --config "$cfg" \
    --experiment-name "$exp" \
    --use-task-embedding false \
    --output-dir outputs/task_level \
    2>&1 | tee "$RUNROOT/01_train_${enc}.log"

  run_dir=$(ls -dt outputs/task_level/${exp}_* | head -n1)
  if [[ -z "$run_dir" ]]; then
    echo "[$(date '+%F %T')] 未找到 ${enc} 的输出目录，停止执行" | tee -a "$RUNROOT/pipeline.log"
    exit 1
  fi
  RUN_DIRS+=("$run_dir")

  report_path="outputs/task_level/encoder_${enc}_report_$(date +%Y%m%d_%H%M%S).xlsx"
  echo "[$(date '+%F %T')] 生成单模型报表：${enc}" | tee -a "$RUNROOT/pipeline.log"
  conda run -n cadt python -u scripts/build_task_embedding_report.py \
    --single-results "$run_dir" \
    --encoder-name "$enc" \
    --output-xlsx "$report_path" \
    2>&1 | tee "$RUNROOT/02_report_${enc}.log"

done

multi_list=$(IFS=,; echo "${RUN_DIRS[*]}")
all_report_path="outputs/task_level/encoder_comparison_$(date +%Y%m%d_%H%M%S).xlsx"

echo "[$(date '+%F %T')] 生成全编码器对比报表" | tee -a "$RUNROOT/pipeline.log"
conda run -n cadt python -u scripts/build_task_embedding_report.py \
  --multi-results "$multi_list" \
  --output-xlsx "$all_report_path" \
  2>&1 | tee "$RUNROOT/03_report_all.log"

echo "[$(date '+%F %T')] 流水线完成" | tee -a "$RUNROOT/pipeline.log"
