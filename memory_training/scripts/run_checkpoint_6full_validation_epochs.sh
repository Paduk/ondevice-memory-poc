#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?GPU index is required}"
run_id="${2:?Run ID is required}"
first_epoch="${3:-1}"
last_epoch="${4:-5}"

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
data_root=/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/grouped-s1-s100-plus-temporal-t1-t20-plus-v1-10-v2
python_bin=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
run_dir="${workspace}/runs/${run_id}"
log_path="${run_dir}/validation-6full-batch6.log"

cd "${repo_root}"
export CUDA_VISIBLE_DEVICES="${gpu}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="${repo_root}:${repo_root}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"

exec > >(tee -a "${log_path}") 2>&1
echo "[$(date -Is)] Starting 6-full batch-6 checkpoint validation: ${run_id}, epochs ${first_epoch}-${last_epoch}."

for epoch_number in $(seq "${first_epoch}" "${last_epoch}"); do
  epoch=$(printf '%02d' "${epoch_number}")
  checkpoint="${run_dir}/checkpoints/epoch-${epoch}"
  output="${run_dir}/validation-6full-batch6-epoch-${epoch}.json"

  if [[ ! -d "${checkpoint}/adapter" ]]; then
    echo "[$(date -Is)] Missing checkpoint adapter: ${checkpoint}" >&2
    exit 1
  fi

  if [[ -s "${output}" ]] && jq -e \
      '.protocol.closed_loop_only == true and
       .protocol.validation_scenarios == [81,82,83,84,85,111] and
       .protocol.scenario_batch_size == 6' \
      "${output}" >/dev/null 2>&1; then
    echo "[$(date -Is)] Epoch ${epoch} already completed; skipping."
    continue
  fi

  echo "[$(date -Is)] Validating epoch ${epoch}."
  "${python_bin}" -m memory_training.validate_hf_checkpoint_v2 \
    --run-id "${run_id}" \
    --checkpoint "${checkpoint}" \
    --output "${output}" \
    --workspace "${workspace}" \
    --data-root "${data_root}" \
    --validation-scenarios 81 82 83 84 85 111 \
    --full-scenarios 81 82 83 84 85 111 \
    --closed-loop-only \
    --scenario-batch-size 6 \
    --quiz-batch-size 16 \
    --finalize-run
done

echo "[$(date -Is)] Completed 6-full batch-6 validation for epochs ${first_epoch}-${last_epoch}."
