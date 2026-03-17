#!/bin/bash
# Independent + Conditional Pooling 消融实验
# 对比基线: Independent (无条件化注意力)
# 变量: use_conditional_pooling=true

cd "$(dirname "$0")/.."

CONFIGS=(
    "configs/task_level_ordinal_indep_cond_transformer.json"
    "configs/task_level_ordinal_indep_cond_cnn1d.json"
    "configs/task_level_ordinal_indep_cond_rnn.json"
    "configs/task_level_ordinal_indep_cond_lstm.json"
    "configs/task_level_ordinal_indep_cond_gru.json"
    "configs/task_level_ordinal_indep_cond_bilstm.json"
)

for cfg in "${CONFIGS[@]}"; do
    enc=$(basename "$cfg" .json | sed 's/task_level_ordinal_indep_cond_//')
    echo "===== [$enc] Start: $(date '+%Y-%m-%d %H:%M:%S') ====="
    python scripts/train_task_level.py --config "$cfg"
    echo "===== [$enc] Done:  $(date '+%Y-%m-%d %H:%M:%S') ====="
    echo
done

echo "All done."
