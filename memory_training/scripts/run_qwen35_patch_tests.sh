#!/usr/bin/env bash
set -euo pipefail

model="${1:?model id is required}"
run_id="${2:?run id is required}"
gpu="${3:?GPU index is required}"
method="${4:-patch}"

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
python_bin=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
data_root=/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/grouped-s1-s100-plus-temporal-t1-t20-plus-v1-10-v2
run_root="${workspace}/runs/${run_id}"
status_path="${run_root}/status.json"

if [[ ! -f "${status_path}" ]]; then
  echo "Missing training status: ${status_path}" >&2
  exit 1
fi

checkpoint_label="$(${python_bin} - "${status_path}" <<'PY'
import json
import pathlib
import sys

status = json.loads(pathlib.Path(sys.argv[1]).read_text())
best = status.get("best_checkpoint")
print(pathlib.Path(best).name if best else "")
PY
)"
if [[ -z "${checkpoint_label}" ]]; then
  checkpoint_label="$(find "${run_root}/checkpoints" -mindepth 1 -maxdepth 1 -type d -name 'epoch-*' -printf '%f\n' | sort | tail -1)"
fi
checkpoint="${run_root}/checkpoints/${checkpoint_label}"
if [[ ! -d "${checkpoint}/adapter" ]]; then
  echo "Missing checkpoint adapter: ${checkpoint}/adapter" >&2
  exit 1
fi

case "${model}" in
  qwen3.5-9b)
    scenario_batch_size=4
    quiz_batch_size=8
    ;;
  *)
    scenario_batch_size=8
    quiz_batch_size=16
    ;;
esac

v2_output="${run_root}/test-hf-closed-loop-s86-s100-t12-t20-best-${checkpoint_label}"
v1_output="${run_root}/test-hf-v1-s1-s50-best-${checkpoint_label}"

is_completed() {
  local progress="$1/progress.json"
  [[ -f "${progress}" ]] && [[ "$(${python_bin} -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status", ""))' "${progress}")" == COMPLETED ]]
}

cd "${repo_root}"
export CUDA_VISIBLE_DEVICES="${gpu}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="${repo_root}:${repo_root}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"

if is_completed "${v2_output}"; then
  echo "[$(date -Is)] V2 Test already completed: ${v2_output}"
else
  echo "[$(date -Is)] Starting ${model} V2 closed-loop Test with ${checkpoint_label} on GPU ${gpu}."
  "${python_bin}" -m memory_training.evaluate_hf_closed_loop \
    --model "${model}" \
    --method "${method}" \
    --checkpoint "${checkpoint}" \
    --output-dir "${v2_output}" \
    --workspace "${workspace}" \
    --data-root "${data_root}" \
    --catalog-path "${data_root}/catalog.sqlite" \
    --scenarios 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 112 113 114 115 116 117 118 119 120 \
    --max-length 2048 \
    --max-new-tokens 768 \
    --quiz-max-new-tokens 256 \
    --scenario-batch-size "${scenario_batch_size}" \
    --quiz-batch-size "${quiz_batch_size}"
fi

if is_completed "${v1_output}"; then
  echo "[$(date -Is)] V1 Test already completed: ${v1_output}"
else
  echo "[$(date -Is)] Starting ${model} V1 S1-S50 Test with ${checkpoint_label} on GPU ${gpu}."
  "${python_bin}" -m memory_training.evaluate_hf_v1 \
    --model "${model}" \
    --method "${method}" \
    --checkpoint "${checkpoint}" \
    --output-dir "${v1_output}" \
    --workspace "${workspace}" \
    --data-root "${data_root}" \
    --vehiclemembench-root /home/hj153lee/VehicleMemBench \
    --scenarios {1..50} \
    --max-length 2048 \
    --max-new-tokens 768 \
    --quiz-max-new-tokens 256 \
    --scenario-batch-size "${scenario_batch_size}" \
    --quiz-batch-size "${quiz_batch_size}" \
    --checkpoint-interval 50
fi

echo "[$(date -Is)] All ${model} ${method} Tests completed."
