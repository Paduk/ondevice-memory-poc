#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 RUN_ID GPU" >&2
  exit 2
fi

run_id=$1
gpu=$2
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
run_dir=$workspace/runs/$run_id
log=$run_dir/validation-v2.log

source /mnt/data/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/data/hj153lee/conda-envs/palmclaw-memory-sft
cd /home/hj153lee/PalmClaw

export CUDA_VISIBLE_DEVICES="$gpu"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

for epoch in 1 2 3 4; do
  label=$(printf '%02d' "$epoch")
  checkpoint=$run_dir/checkpoints/epoch-$label
  output=$run_dir/validation-v2-epoch-$label.json
  if [[ -s "$output" ]]; then
    echo "Epoch $epoch V2 Validation already exists; restoring." | tee -a "$log"
    continue
  fi
  echo "Epoch $epoch V2 Validation started." | tee -a "$log"
  python -m memory_training.validate_hf_checkpoint_v2 \
    --run-id "$run_id" \
    --checkpoint "$checkpoint" \
    --output "$output" \
    --scenario-batch-size 2 \
    --quiz-batch-size 16 \
    --teacher-batch-size 4 \
    2>&1 | tee -a "$log"
done

jq -s '{
  schema_version:"palmclaw-checkpoint-validation-v2-summary",
  results:map({
    epoch,
    checkpoint,
    teacher_forced_loss:.teacher_forced.loss,
    one_step_update_f1:.one_step.update_f1,
    closed_loop_update_f1:.closed_loop.update_f1,
    closed_loop_final_state_f1:.closed_loop.final_state_f1,
    closed_loop_quiz_esm:.closed_loop_quiz.esm,
    closed_loop_quiz_tool_f1:.closed_loop_quiz.tool_f1,
    closed_loop_quiz_arg_exact:.closed_loop_quiz.arg_exact,
    gold_quiz_esm:.quiz.esm
  })
}' "$run_dir"/validation-v2-epoch-*.json > "$run_dir/validation-v2-summary.json.tmp"
mv "$run_dir/validation-v2-summary.json.tmp" "$run_dir/validation-v2-summary.json"

jq --arg now "$(date -u +%Y-%m-%dT%H:%M:%S%:z)" \
  '.updated_at=$now | .state="COMPLETED" | .epoch=4 |
   .message="Training complete; checkpoint V2 Validation complete" |
   del(.evaluation)' \
  "$run_dir/status.json" > "$run_dir/status.json.tmp"
mv "$run_dir/status.json.tmp" "$run_dir/status.json"
echo "All checkpoint V2 Validations completed." | tee -a "$log"
