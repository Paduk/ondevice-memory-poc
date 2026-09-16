#!/usr/bin/env bash
set -euo pipefail

gpu="${1:-5}"
repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
python_bin=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
data_root=/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/grouped-s1-s100-plus-temporal-t1-t20-plus-v1-10-v2-delta-v3-compact-k10-v1
run_id=llama3.2-1b-delta_v3_compact_k10-multitask-noop5-e4-b2-trainseed45-evalfixed-noop5-r1
run_dir="${workspace}/runs/${run_id}"
resume_checkpoint="${run_dir}/checkpoints/step-0001500"
target_checkpoint="${run_dir}/checkpoints/epoch-02"
expected_loss=0.2360149323940277

if [[ -d "${target_checkpoint}/adapter" ]]; then
  echo "epoch-02 adapter already exists: ${target_checkpoint}/adapter"
  exit 0
fi
if [[ ! -d "${resume_checkpoint}/adapter" || ! -f "${resume_checkpoint}/training_state.pt" ]]; then
  echo "Exact resume checkpoint is incomplete: ${resume_checkpoint}" >&2
  exit 2
fi

train_scenarios=(
  $(seq 15 80)
  $(seq 101 110)
  202 205 206 214 217 223 231 232 233 236
)

cd "${repo_root}"
export CUDA_VISIBLE_DEVICES="${gpu}"
export HF_HOME="${workspace}/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH="${repo_root}:${repo_root}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"

"${python_bin}" -m memory_training.train \
  --model llama3.2-1b \
  --method delta_v3_compact_k10 \
  --run-id "${run_id}" \
  --resume "${resume_checkpoint}" \
  --stop-after-epoch 2 \
  --workspace "${workspace}" \
  --data-root "${data_root}" \
  --catalog-path "${data_root}/catalog.sqlite" \
  --epochs 4 \
  --batch-size 2 \
  --eval-batch-size 2 \
  --gradient-accumulation 8 \
  --learning-rate 0.0002 \
  --weight-decay 0.01 \
  --warmup-ratio 0.03 \
  --max-length 4096 \
  --max-new-tokens 768 \
  --multitask \
  --quiz-total-passes 2 \
  --quiz-batch-size 4 \
  --vehiclemembench-root /home/hj153lee/VehicleMemBench \
  --quiz-validation-rows-per-scenario 4 \
  --gold-quiz-validation-rows 0 \
  --quiz-validation-seed 42 \
  --quiz-validation-max-new-tokens 256 \
  --quiz-validation-batch-size 2 \
  --skip-generation-validation \
  --skip-quiz-validation \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --gradient-checkpointing \
  --seed 45 \
  --training-seed 45 \
  --noop-per-update 5 \
  --delta-v3-noop-pending-weights 1 1 1 1 1 1 1 1 1 1 \
  --delta-v3-append-max-turns 32 \
  --adjacent-noop-fraction 0.30 \
  --trajectory-fraction 0.20 \
  --train-scenarios "${train_scenarios[@]}" \
  --validation-scenarios 81 82 83 84 85 111 \
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

actual_loss=$(jq -r '.teacher_forced.loss' "${run_dir}/validation-epoch-02.json")
"${python_bin}" - "${actual_loss}" "${expected_loss}" <<'PY'
import math
import sys

actual = float(sys.argv[1])
expected = float(sys.argv[2])
if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9):
    raise SystemExit(
        f"Recovered epoch-02 validation loss mismatch: {actual} != {expected}"
    )
print(f"Recovered epoch-02 validation loss verified: {actual:.16f}")
PY

test -d "${target_checkpoint}/adapter"
echo "Exact epoch-02 adapter recovery completed: ${target_checkpoint}/adapter"
