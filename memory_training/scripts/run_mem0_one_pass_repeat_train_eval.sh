#!/usr/bin/env bash
set -euo pipefail

gpu=${1:?GPU index required}
model=${2:?Model required: granite4-350m or qwen3.5-0.8b}
training_seed=${3:-46}
repeat_tag=${4:-r2}

case "${model}" in
  granite4-350m)
    model_slug=granite4-350m
    batch_size=8
    accumulation=2
    scenario_batch_size=16
    ;;
  qwen3.5-0.8b)
    model_slug=qwen35-0.8b
    batch_size=8
    accumulation=2
    scenario_batch_size=16
    ;;
  *)
    echo "Unsupported model: ${model}" >&2
    exit 2
    ;;
esac

repo=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
python=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
train_root=/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/mem0-style-one-pass-training-s21t10-no-v1-v2
eval_root=/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/mem0-style-one-pass-eval-fixed-noop5-seed45-v1
run_id=${model_slug}-mem0-one-pass-multitask-noop5-e4-b${batch_size}-trainseed${training_seed}-valcomposite-${repeat_tag}
run_dir=${workspace}/runs/${run_id}

export CUDA_VISIBLE_DEVICES="${gpu}"
export HF_HOME="${workspace}/cache/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PALMCLAW_MEM0_RETRIEVAL_DEVICE=cuda
export PYTHONPATH="${repo}:${repo}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"

train_scenarios=(
  $(seq 21 80) $(seq 101 110)
)
validation_scenarios=(81 82 83 84 85 111)
test_scenarios=($(seq 86 100) $(seq 112 120))

mkdir -p "${run_dir}"
exec > >(tee -a "${run_dir}/pipeline.log") 2>&1
cd "${repo}"

echo "[$(date -Is)] Pipeline start: ${model}, GPU ${gpu}, seed ${training_seed}."
final_checkpoint=${run_dir}/checkpoints/epoch-04
if [[ ! -d "${final_checkpoint}/adapter" ]]; then
  "${python}" -m memory_training.train \
    --model "${model}" --method mem0_one_pass --run-id "${run_id}" \
    --workspace "${workspace}" --data-root "${train_root}" \
    --catalog-path "${train_root}/catalog.sqlite" \
    --epochs 4 --batch-size "${batch_size}" --eval-batch-size 8 \
    --gradient-accumulation "${accumulation}" --learning-rate 2e-4 \
    --weight-decay 0.01 --warmup-ratio 0.03 \
    --max-length 2048 --max-new-tokens 768 \
    --multitask --quiz-total-passes 2 --quiz-batch-size 8 \
    --skip-quiz-validation --quiz-validation-rows-per-scenario 4 \
    --gold-quiz-validation-rows 0 --quiz-validation-seed 42 \
    --quiz-validation-max-new-tokens 256 --quiz-validation-batch-size 8 \
    --lora-rank 16 --lora-alpha 32 --lora-dropout 0.05 \
    --gradient-checkpointing --seed "${training_seed}" \
    --training-seed "${training_seed}" \
    --noop-per-update 5 --adjacent-noop-fraction 0.30 \
    --trajectory-fraction 0.20 \
    --train-scenarios "${train_scenarios[@]}" \
    --validation-scenarios "${validation_scenarios[@]}" \
    --num-workers 4 --prefetch-factor 2 --pin-memory \
    --log-steps 5 --save-steps 0 --eval-steps 0 \
    --eval-max-rows 100 --eval-max-batches 1 \
    --eval-adjacent-noop-fraction 0.50 --false-update-threshold 0.20 \
    --skip-generation-validation
else
  echo "[$(date -Is)] Epoch 04 exists; resuming post-training pipeline."
fi

for epoch_number in 1 2 3 4; do
  epoch=$(printf '%02d' "${epoch_number}")
  checkpoint=${run_dir}/checkpoints/epoch-${epoch}
  output=${run_dir}/eval-fixed-v2-validation-epoch-${epoch}.json
  if [[ ! -d "${checkpoint}/adapter" ]]; then
    echo "Missing checkpoint adapter: ${checkpoint}" >&2
    exit 1
  fi
  if [[ ! -s "${output}" ]]; then
    echo "[$(date -Is)] Validation epoch ${epoch} started."
    "${python}" -m memory_training.validate_hf_checkpoint_v2 \
      --run-id "${run_id}" --checkpoint "${checkpoint}" \
      --output "${output}" --workspace "${workspace}" \
      --data-root "${eval_root}" --catalog-path "${eval_root}/catalog.sqlite" \
      --validation-scenarios "${validation_scenarios[@]}" \
      --full-scenarios "${validation_scenarios[@]}" --closed-loop-only \
      --scenario-batch-size "${scenario_batch_size}" --quiz-batch-size 16 \
      --no-status-update \
      >"${run_dir}/eval-fixed-v2-validation-epoch-${epoch}.log" 2>&1
  fi
done

"${python}" - "${run_dir}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

run_dir = Path(sys.argv[1])
rows = []
for epoch in range(1, 5):
    path = run_dir / f"eval-fixed-v2-validation-epoch-{epoch:02d}.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    row = {
        "epoch": epoch,
        "checkpoint": value["checkpoint"],
        "closed_loop_quiz_esm": value["closed_loop_quiz"]["esm"],
        "closed_loop_final_state_f1": value["closed_loop"]["final_state_f1"],
        "closed_loop_update_f1": value["closed_loop"]["update_f1"],
        "false_update_rate": value["closed_loop"]["false_update_rate"],
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
summary = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "best_metric": "composite.quiz_esm_60.final_state_f1_25.update_f1_15",
    "results": rows,
    "winner": winner,
}
target = run_dir / "best-fixed-validation-checkpoint.json"
temporary = target.with_suffix(".json.tmp")
temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, target)
PY

checkpoint=$("${python}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["winner"]["checkpoint"])' "${run_dir}/best-fixed-validation-checkpoint.json")
epoch=$(basename "${checkpoint}")
test_output=${run_dir}/eval-fixed-v2-test-best-${epoch}
if [[ $(jq -r '.complete // false' "${test_output}/summary.json" 2>/dev/null || true) != true ]]; then
  echo "[$(date -Is)] Fixed Test started for ${epoch}."
  "${python}" -m memory_training.evaluate_hf_closed_loop \
    --model "${model}" --method mem0_one_pass --checkpoint "${checkpoint}" \
    --output-dir "${test_output}" --workspace "${workspace}" \
    --data-root "${eval_root}" --catalog-path "${eval_root}/catalog.sqlite" \
    --scenarios "${test_scenarios[@]}" --max-length 2048 \
    --max-new-tokens 768 --quiz-max-new-tokens 256 \
    --scenario-batch-size "${scenario_batch_size}" --quiz-batch-size 16 \
    >"${run_dir}/eval-fixed-v2-test-best-${epoch}.log" 2>&1
fi

echo "[$(date -Is)] COMPLETED ${model}: train, 4-epoch Validation selection, Test."
