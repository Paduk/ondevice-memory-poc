#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?GPU index is required}"
method="${2:?Method is required}"
epochs="${3:-4}"
training_seed="${4:-45}"
run_tag="${5:-r1}"
model_key="${6:-llama3.2-1b}"

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
python_bin=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
data_parent=/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training

case "${method}" in
  patch)
    train_name=grouped-s1-s100-plus-temporal-t1-t20-v2
    eval_name=grouped-v2-v1-10-eval-fixed-noop5-seed45-v1
    batch_size=8
    eval_batch_size=8
    gradient_accumulation=2
    train_max_length=2048
    method_args=()
    ;;
  summary)
    train_name=grouped-s1-s100-plus-temporal-t1-t20-v2
    eval_name=grouped-v2-v1-10-eval-fixed-noop5-seed45-v1
    batch_size=4
    eval_batch_size=4
    gradient_accumulation=4
    train_max_length=2048
    method_args=()
    ;;
  delta_v3_compact_k2|delta_v3_compact_k5|delta_v3_compact_k10)
    interval="${method##*k}"
    train_name=grouped-s1-s100-plus-temporal-t1-t20-v2-delta-v3-compact-k${interval}-v1
    eval_name=grouped-v2-v1-10-eval-fixed-noop5-seed45-v1-delta-v3-compact-k${interval}-v1
    batch_size=2
    eval_batch_size=2
    gradient_accumulation=8
    train_max_length=4096
    pending_weights=()
    for ((depth = 0; depth < interval; depth++)); do
      pending_weights+=(1)
    done
    method_args=(--delta-v3-noop-pending-weights "${pending_weights[@]}")
    ;;
  *)
    echo "Unsupported method: ${method}; expected summary, patch, or delta_v3_compact_k{2,5,10}" >&2
    exit 2
    ;;
esac

validation_scenario_batch_size=3
validation_quiz_batch_size=16
test_scenario_batch_size=16
test_quiz_batch_size=16
train_quiz_batch_size=4
case "${model_key}" in
  llama3.2-1b) ;;
  llama3.2-3b)
    validation_scenario_batch_size=2
    validation_quiz_batch_size=8
    test_scenario_batch_size=8
    test_quiz_batch_size=8
    if [[ "${method}" == patch ]]; then
      batch_size=4
      eval_batch_size=4
      gradient_accumulation=4
      train_quiz_batch_size=4
    elif [[ "${method}" == summary ]]; then
      batch_size=2
      eval_batch_size=2
      gradient_accumulation=8
      train_quiz_batch_size=2
    else
      batch_size=1
      eval_batch_size=1
      gradient_accumulation=16
      train_quiz_batch_size=2
    fi
    ;;
  *)
    echo "Unsupported Llama model: ${model_key}" >&2
    exit 2
    ;;
esac

train_root="${data_parent}/${train_name}"
eval_root="${data_parent}/${eval_name}"
run_id="${model_key}-${method}-multitask-noop5-e${epochs}-b${batch_size}-trainseed${training_seed}-evalfixed-noop5-${run_tag}"
run_dir="${workspace}/runs/${run_id}"
log_path="${run_dir}/pipeline.log"

for root in "${train_root}" "${eval_root}"; do
  if [[ ! -f "${root}/catalog.sqlite" ]]; then
    echo "Prepared data root is incomplete: ${root}" >&2
    exit 2
  fi
done

cd "${repo_root}"
export CUDA_VISIBLE_DEVICES="${gpu}"
export HF_HOME="${workspace}/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH="${repo_root}:${repo_root}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${run_dir}"
exec > >(tee -a "${log_path}") 2>&1

train_scenarios=(
  $(seq 21 80)
  $(seq 101 110)
)
validation_scenarios=(81 82 83 84 85 111)
test_scenarios=(
  $(seq 86 100)
  $(seq 112 120)
)

final_checkpoint="${run_dir}/checkpoints/epoch-$(printf '%02d' "${epochs}")"
if [[ ! -d "${final_checkpoint}/adapter" ]]; then
  echo "[$(date -Is)] Starting ${model_key} ${method} training on GPU ${gpu}."
  train_command=(
    "${python_bin}" -m memory_training.train
    --model "${model_key}" \
    --method "${method}" \
    --run-id "${run_id}" \
    --workspace "${workspace}" \
    --data-root "${train_root}" \
    --catalog-path "${train_root}/catalog.sqlite" \
    --epochs "${epochs}" \
    --batch-size "${batch_size}" \
    --eval-batch-size "${eval_batch_size}" \
    --gradient-accumulation "${gradient_accumulation}" \
    --learning-rate 0.0002 \
    --weight-decay 0.01 \
    --warmup-ratio 0.03 \
    --max-length "${train_max_length}" \
    --max-new-tokens 768 \
    --multitask \
    --quiz-total-passes 2 \
    --quiz-batch-size "${train_quiz_batch_size}" \
    --quiz-validation-rows-per-scenario 4 \
    --gold-quiz-validation-rows 0 \
    --quiz-validation-seed 42 \
    --quiz-validation-max-new-tokens 256 \
    --quiz-validation-batch-size "${eval_batch_size}" \
    --skip-generation-validation \
    --skip-quiz-validation \
    --lora-rank 16 \
    --lora-alpha 32 \
    --lora-dropout 0.05 \
    --gradient-checkpointing \
    --seed "${training_seed}" \
    --training-seed "${training_seed}" \
    --noop-per-update 5 \
    --adjacent-noop-fraction 0.30 \
    --trajectory-fraction 0.20 \
    --train-scenarios "${train_scenarios[@]}" \
    --validation-scenarios "${validation_scenarios[@]}" \
    --num-workers 4 \
    --prefetch-factor 2 \
    --pin-memory \
    --log-steps 5 \
    --save-steps 500 \
    --eval-steps 0 \
    --eval-max-rows 100 \
    --eval-max-batches 1 \
    --eval-adjacent-noop-fraction 0.50 \
    --false-update-threshold 0.20
  )
  train_command+=("${method_args[@]}")
  "${train_command[@]}"
else
  echo "[$(date -Is)] Training already completed; reusing checkpoints."
fi

validate_epoch() {
  local epoch_number=$1
  local epoch checkpoint output epoch_log
  epoch=$(printf '%02d' "${epoch_number}")
  checkpoint="${run_dir}/checkpoints/epoch-${epoch}"
  output="${run_dir}/eval-fixed-validation-epoch-${epoch}.json"
  epoch_log="${run_dir}/eval-fixed-validation-epoch-${epoch}.log"
  if [[ -s "${output}" ]]; then
    echo "[$(date -Is)] Validation epoch ${epoch} already completed; skipping."
    return 0
  fi
  echo "[$(date -Is)] Starting fixed validation epoch ${epoch} on GPU ${gpu}."
  "${python_bin}" -m memory_training.validate_hf_checkpoint_v2 \
    --run-id "${run_id}" \
    --checkpoint "${checkpoint}" \
    --output "${output}" \
    --workspace "${workspace}" \
    --data-root "${eval_root}" \
    --catalog-path "${eval_root}/catalog.sqlite" \
    --validation-scenarios "${validation_scenarios[@]}" \
    --full-scenarios "${validation_scenarios[@]}" \
    --closed-loop-only \
    --scenario-batch-size "${validation_scenario_batch_size}" \
    --quiz-batch-size "${validation_quiz_batch_size}" \
    --no-status-update \
    >"${epoch_log}" 2>&1
}

echo "[$(date -Is)] Starting fixed validation for epochs 1-${epochs}."
declare -A running=()
next_epoch=1
failed=0
while (( next_epoch <= epochs || ${#running[@]} > 0 )); do
  while (( next_epoch <= epochs && ${#running[@]} < 2 )); do
    validate_epoch "${next_epoch}" &
    running[$!]="${next_epoch}"
    next_epoch=$((next_epoch + 1))
  done
  reaped=0
  for pid in "${!running[@]}"; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      epoch_number="${running[$pid]}"
      if ! wait "${pid}"; then
        echo "[$(date -Is)] Validation epoch ${epoch_number} failed." >&2
        failed=1
      fi
      unset 'running[$pid]'
      reaped=1
    fi
  done
  if (( failed )); then
    for pid in "${!running[@]}"; do kill "${pid}" 2>/dev/null || true; done
    exit 1
  fi
  if (( reaped == 0 )); then sleep 2; fi
done

selection="${run_dir}/eval-fixed-best-checkpoint.json"
"${python_bin}" - "${run_dir}" "${selection}" "${epochs}" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
output = Path(sys.argv[2])
epochs = int(sys.argv[3])
rows = []
for epoch in range(1, epochs + 1):
    path = run_dir / f"eval-fixed-validation-epoch-{epoch:02d}.json"
    result = json.loads(path.read_text())
    row = {
        "epoch": epoch,
        "checkpoint": result["checkpoint"],
        "closed_loop_update_f1": result["closed_loop"]["update_f1"],
        "closed_loop_final_state_f1": result["closed_loop"]["final_state_f1"],
        "closed_loop_quiz_esm": result["closed_loop_quiz"]["esm"],
        "output": str(path),
    }
    row["composite_score"] = (
        0.60 * row["closed_loop_quiz_esm"]
        + 0.25 * row["closed_loop_final_state_f1"]
        + 0.15 * row["closed_loop_update_f1"]
    )
    rows.append(row)
winner = max(
    rows,
    key=lambda row: (
        row["composite_score"],
        row["closed_loop_quiz_esm"],
        row["closed_loop_final_state_f1"],
        row["closed_loop_update_f1"],
        -row["epoch"],
    ),
)
output.write_text(
    json.dumps(
        {"selection_metric": "composite_60_25_15", "results": rows, "winner": winner},
        indent=2,
    )
    + "\n"
)
PY

winner=$(jq -r '.winner.checkpoint' "${selection}")
winner_epoch=$(basename "${winner}")
test_output="${run_dir}/eval-fixed-test-best-${winner_epoch}"

if [[ ! -s "${test_output}/summary.json" ]]; then
  echo "[$(date -Is)] Starting fixed 24-scenario Test with ${winner_epoch} on GPU ${gpu}."
  "${python_bin}" -m memory_training.evaluate_hf_closed_loop \
    --model "${model_key}" \
    --method "${method}" \
    --checkpoint "${winner}" \
    --output-dir "${test_output}" \
    --workspace "${workspace}" \
    --data-root "${eval_root}" \
    --catalog-path "${eval_root}/catalog.sqlite" \
    --scenarios "${test_scenarios[@]}" \
    --max-length 2048 \
    --max-new-tokens 768 \
    --quiz-max-new-tokens 256 \
    --scenario-batch-size "${test_scenario_batch_size}" \
    --quiz-batch-size "${test_quiz_batch_size}"
else
  echo "[$(date -Is)] Fixed Test already completed; skipping."
fi

"${python_bin}" - "${run_dir}" "${winner}" "${selection}" "${test_output}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

run_dir, winner, selection, test_output = map(Path, sys.argv[1:])
status = {
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "state": "COMPLETED",
    "run_id": run_dir.name,
    "final_state": "COMPLETED",
    "best_checkpoint": str(winner),
    "best_metric": "eval_fixed.composite.quiz_esm_60.final_state_f1_25.update_f1_15",
    "external_validation": selection.name,
    "external_test": str(test_output / "summary.json"),
}
temporary = run_dir / "status.json.tmp"
temporary.write_text(json.dumps(status, indent=2) + "\n")
os.replace(temporary, run_dir / "status.json")
PY

echo "[$(date -Is)] Completed ${model_key} ${method} training, validation, and Test."
