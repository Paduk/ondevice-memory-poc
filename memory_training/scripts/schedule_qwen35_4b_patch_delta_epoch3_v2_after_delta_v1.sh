#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
python_bin=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
data_root=/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/grouped-s1-s100-plus-temporal-t1-t20-plus-v1-10-v2
gpu=0
delta_run_id=qwen35-4b-delta-v3-multitask-noop5-pending4030151005-grouped-v2-v1-10-e5-r1
delta_v1_progress="${workspace}/runs/${delta_run_id}/test-hf-v1-s1-s50-best-epoch-02/progress.json"
log_path="${workspace}/qwen35-4b-patch-delta-epoch3-v2-after-delta-v1-gpu0.log"

exec > >(tee -a "${log_path}") 2>&1
echo "[$(date -Is)] Waiting for Delta v3 V1 Test to complete."

while true; do
  state=MISSING
  if [[ -f "${delta_v1_progress}" ]]; then
    state="$(${python_bin} -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status", "UNKNOWN"))' "${delta_v1_progress}")"
  fi
  case "${state}" in
    COMPLETED)
      while pgrep -f '[m]emory_training.evaluate_hf_v1.*qwen35-4b-delta-v3' >/dev/null; do
        sleep 10
      done
      break
      ;;
    FAILED)
      echo "[$(date -Is)] Delta v3 V1 Test failed; Epoch 3 V2 Tests will not start." >&2
      exit 1
      ;;
  esac
  sleep 30
done

cd "${repo_root}"
export CUDA_VISIBLE_DEVICES="${gpu}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="${repo_root}:${repo_root}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"

run_v2_test() {
  local label="$1"
  local method="$2"
  local run_id="$3"
  local run_root="${workspace}/runs/${run_id}"
  local checkpoint="${run_root}/checkpoints/epoch-03"
  local output_dir="${run_root}/test-hf-closed-loop-s86-s100-t12-t20-epoch03"

  if [[ ! -d "${checkpoint}/adapter" ]]; then
    echo "[$(date -Is)] Missing ${label} checkpoint: ${checkpoint}/adapter" >&2
    exit 1
  fi
  if [[ -f "${output_dir}/progress.json" ]] && \
     [[ "$(${python_bin} -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status", ""))' "${output_dir}/progress.json")" == COMPLETED ]]; then
    echo "[$(date -Is)] ${label} Epoch 3 V2 Test already completed."
    return
  fi

  echo "[$(date -Is)] Starting ${label} Epoch 3 V2 Test on GPU ${gpu}."
  "${python_bin}" -m memory_training.evaluate_hf_closed_loop \
    --model qwen3.5-4b \
    --method "${method}" \
    --checkpoint "${checkpoint}" \
    --output-dir "${output_dir}" \
    --workspace "${workspace}" \
    --data-root "${data_root}" \
    --catalog-path "${data_root}/catalog.sqlite" \
    --scenarios 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 112 113 114 115 116 117 118 119 120 \
    --max-length 2048 \
    --max-new-tokens 768 \
    --quiz-max-new-tokens 256 \
    --scenario-batch-size 8 \
    --quiz-batch-size 16
  echo "[$(date -Is)] Completed ${label} Epoch 3 V2 Test."
}

run_v2_test Patch patch qwen35-4b-patch-multitask-noop5-grouped-v2-v1-10-r1
run_v2_test Delta-v3 delta_v3 "${delta_run_id}"

echo "[$(date -Is)] Patch and Delta v3 Epoch 3 V2 Tests completed."
