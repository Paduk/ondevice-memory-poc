#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
data_root=/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/grouped-s1-s100-plus-temporal-t1-t20-plus-v1-10-v2
python_bin=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
validation_runner="${repo_root}/memory_training/scripts/run_checkpoint_6full_validation_pairwise.sh"
run_id=granite4-350m-patch-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed47-r1
run_dir="${workspace}/runs/${run_id}"
old_session=granite4-350m-seed47-pairval-gpu3
gpu=3

cd "${repo_root}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="${repo_root}:${repo_root}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"

epoch1_pid=$(pgrep -f "memory_training.validate_hf_checkpoint_v2.*${run_id}.*checkpoints/epoch-01" | head -1)
if [[ -z "${epoch1_pid}" ]]; then
  echo "[$(date -Is)] Existing Epoch 01 validator was not found." >&2
  exit 1
fi

validate_epoch() {
  local epoch=$1
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" \
    -m memory_training.validate_hf_checkpoint_v2 \
    --run-id "${run_id}" \
    --checkpoint "${run_dir}/checkpoints/epoch-${epoch}" \
    --output "${run_dir}/validation-6full-pairwise-epoch-${epoch}.json" \
    --workspace "${workspace}" \
    --data-root "${data_root}" \
    --validation-scenarios 81 82 83 84 85 111 \
    --full-scenarios 81 82 83 84 85 111 \
    --closed-loop-only \
    --scenario-batch-size 3 \
    --quiz-batch-size 16 \
    >"${run_dir}/validation-6full-pairwise-epoch-${epoch}.log" 2>&1
}

echo "[$(date -Is)] Filling the free slot with Epoch 03; Epoch 01 PID is ${epoch1_pid}."
validate_epoch 03 &
epoch3_pid=$!

while kill -0 "${epoch1_pid}" 2>/dev/null && kill -0 "${epoch3_pid}" 2>/dev/null; do
  sleep 5
done

echo "[$(date -Is)] One slot is free; starting Epoch 04."
validate_epoch 04 &
epoch4_pid=$!

wait "${epoch3_pid}"
wait "${epoch4_pid}"
while kill -0 "${epoch1_pid}" 2>/dev/null; do
  sleep 5
done

for epoch in 01 02 03 04; do
  test -s "${run_dir}/validation-6full-pairwise-epoch-${epoch}.json"
done

tmux kill-session -t "${old_session}" 2>/dev/null || true
exec bash "${validation_runner}" "${gpu}" "${run_id}" 1 4
