#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/hj153lee/PalmClaw"
runs_root="/mnt/data/hj153lee/PalmClaw/on-device-memory-training/runs"
python_bin="/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python"
run_id="qwen35-4b-patch-multitask-noop5-r1"
run_root="${runs_root}/${run_id}"
checkpoint="${run_root}/checkpoints/epoch-04"
current_output="${run_root}/test-hf-closed-loop-s86-s100-epoch04"
current_progress="${current_output}/progress.json"
output_dir="${run_root}/test-hf-v1-s1-s50-epoch04"
watch_log="${run_root}/v1-after-s86-s100-test.log"

exec > >(tee -a "${watch_log}") 2>&1
echo "[$(date -Is)] Waiting for the GPU 5 S86-S100 Patch test."

missing_process_checks=0
while true; do
  status="$(${python_bin} - "${current_progress}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    print("MISSING")
else:
    try:
        print(json.loads(path.read_text(encoding="utf-8")).get("status", "UNKNOWN"))
    except (OSError, json.JSONDecodeError):
        print("UNREADABLE")
PY
)"
  if [[ "${status}" == "COMPLETED" ]]; then
    break
  fi
  if [[ "${status}" == "FAILED" ]]; then
    echo "[$(date -Is)] Current S86-S100 test failed; V1 test was not started."
    exit 1
  fi
  if pgrep -f '[m]emory_training.evaluate_hf_closed_loop.*test-hf-closed-loop-s86-s100-epoch04' >/dev/null; then
    missing_process_checks=0
  else
    missing_process_checks=$((missing_process_checks + 1))
    if (( missing_process_checks >= 5 )); then
      echo "[$(date -Is)] Current test exited without COMPLETED status; V1 test was not started."
      exit 1
    fi
  fi
  echo "[$(date -Is)] Current status=${status}; checking again in 60 seconds."
  sleep 60
done

echo "[$(date -Is)] S86-S100 test completed; waiting for its GPU process to exit."
while pgrep -f '[m]emory_training.evaluate_hf_closed_loop.*test-hf-closed-loop-s86-s100-epoch04' >/dev/null; do
  sleep 5
done

while nvidia-smi -i 5 --query-compute-apps=pid --format=csv,noheader,nounits \
  2>/dev/null | grep -Eq '[0-9]'; do
  echo "[$(date -Is)] GPU 5 is occupied by another process; waiting 60 seconds."
  sleep 60
done

if [[ ! -d "${checkpoint}/adapter" ]]; then
  echo "[$(date -Is)] Missing checkpoint adapter: ${checkpoint}/adapter"
  exit 1
fi

mkdir -p "${output_dir}"
cd "${repo_root}"
export CUDA_VISIBLE_DEVICES=5
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${repo_root}:${repo_root}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "[$(date -Is)] Starting V1 S1-S50: Gold Quiz -> Memory Replay -> Final-memory Quiz."
exec "${python_bin}" -m memory_training.evaluate_hf_v1 \
  --model qwen3.5-4b \
  --method patch \
  --checkpoint "${checkpoint}" \
  --output-dir "${output_dir}" \
  --vehiclemembench-root /home/hj153lee/VehicleMemBench \
  --scenarios {1..50} \
  --max-length 2048 \
  --max-new-tokens 768 \
  --quiz-max-new-tokens 256 \
  --scenario-batch-size 8 \
  --quiz-batch-size 16 \
  --checkpoint-interval 50
