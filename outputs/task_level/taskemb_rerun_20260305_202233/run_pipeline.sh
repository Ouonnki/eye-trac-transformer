#!/usr/bin/env bash
set -euo pipefail

cd /home/sysadmin/eye-trac-transformer
RUNROOT="$1"

export PYTHONUNBUFFERED=1

echo "[$(date '+%F %T')] Pipeline started" | tee -a "$RUNROOT/pipeline.log"

echo "[$(date '+%F %T')] Step 1/4: Baseline (no task embedding)" | tee -a "$RUNROOT/pipeline.log"
conda run -n cadt python -u scripts/train_task_level.py \
  --config configs/task_level.json \
  --use-task-embedding false \
  --experiment-name task_level_no_taskemb \
  --output-dir outputs/task_level \
  2>&1 | tee "$RUNROOT/01_train_no_taskemb.log"

echo "[$(date '+%F %T')] Step 2/4: Baseline + task embedding" | tee -a "$RUNROOT/pipeline.log"
conda run -n cadt python -u scripts/train_task_level.py \
  --config configs/task_level.json \
  --use-task-embedding true \
  --experiment-name task_level_with_taskemb \
  --output-dir outputs/task_level \
  2>&1 | tee "$RUNROOT/02_train_with_taskemb.log"

echo "[$(date '+%F %T')] Step 3/4: Uniform random baseline" | tee -a "$RUNROOT/pipeline.log"
conda run -n cadt python -u scripts/random_baseline_task_level.py \
  --config configs/task_level.json \
  2>&1 | tee "$RUNROOT/03_random_baseline.log"

BASE=$(ls -dt outputs/task_level/task_level_no_taskemb_* | head -n1)
TASK=$(ls -dt outputs/task_level/task_level_with_taskemb_* | head -n1)
RAND=$(ls -dt outputs/task_level/random_baseline_* | head -n1)
REPORT_PATH="outputs/task_level/task_embedding_comparison_$(date +%Y%m%d_%H%M%S).xlsx"

echo "BASE=$BASE" | tee -a "$RUNROOT/pipeline.log"
echo "TASK=$TASK" | tee -a "$RUNROOT/pipeline.log"
echo "RAND=$RAND" | tee -a "$RUNROOT/pipeline.log"
echo "REPORT=$REPORT_PATH" | tee -a "$RUNROOT/pipeline.log"

echo "[$(date '+%F %T')] Step 4/4: Build Excel report" | tee -a "$RUNROOT/pipeline.log"
conda run -n cadt python -u scripts/build_task_embedding_report.py \
  --baseline-results "$BASE/test_results.json" \
  --task-results "$TASK/test_results.json" \
  --random-results "$RAND/random_baseline_results.json" \
  --random-kind uniform_random \
  --output-xlsx "$REPORT_PATH" \
  2>&1 | tee "$RUNROOT/04_build_report.log"

echo "[$(date '+%F %T')] Pipeline finished" | tee -a "$RUNROOT/pipeline.log"
