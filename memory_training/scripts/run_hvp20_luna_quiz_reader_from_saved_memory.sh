#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
python_bin=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
data_root=/mnt/data/hj153lee/PalmClaw/evaluation/human-authored-vehicle-memory/hvp01-hvp20-multiparty-quiz-expansion-v1-eval
output_root="${workspace}/evaluations/hvp01-hvp20-multiparty-quiz-expansion-v1/gpt-5.6-luna-patch-writer-granite4-1b-reader"
memory_root="${output_root}/cloud-memory"
log_root="${output_root}/logs"
scenarios=(901 902 903 904 905 906 907 908 909 910 911 912 913 914 915 916 917 918 919 920)
scenario_parallelism=4
quiz_workers=4

mkdir -p "$log_root"
exec 9>"${output_root}/luna-reader-scheduler.lock"
if ! flock -n 9; then
  echo "[$(date -Is)] another HVP Luna reader evaluation is running" >&2
  exit 3
fi
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[$(date -Is)] OPENAI_API_KEY is required" >&2
  exit 2
fi

cd "$repo_root"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${repo_root}:${repo_root}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"

run_scenario() {
  local scenario=$1 scenario_output scenario_log
  scenario_output="${memory_root}/s${scenario}"
  scenario_log="${log_root}/luna-reader-s${scenario}.log"
  [[ -d "${scenario_output}/memory_steps" ]] || {
    echo "[$(date -Is)] saved memory missing: ${scenario_output}" >&2
    return 2
  }
  echo "[$(date -Is)] start Luna reader S${scenario}" | tee -a "$scenario_log"
  "$python_bin" -m memory_training.evaluate_cloud_closed_loop \
    --method patch \
    --scenario "$scenario" \
    --model gpt-5.6-luna \
    --output-dir "$scenario_output" \
    --data-root "$data_root" \
    --workspace "$workspace" \
    --catalog-path "${data_root}/catalog.sqlite" \
    --vehicle-tools-path "${data_root}/vehicle_tools.json" \
    --quiz-sft-path "${data_root}/quiz_sft.jsonl" \
    --vehiclemembench-root /home/hj153lee/VehicleMemBench \
    --memory-max-output-tokens 768 \
    --quiz-max-output-tokens 256 \
    --quiz-workers "$quiz_workers" \
    --reasoning-effort low \
    >>"$scenario_log" 2>&1
  echo "[$(date -Is)] completed Luna reader S${scenario}" | tee -a "$scenario_log"
}

echo "[$(date -Is)] start HVP Luna Quiz reader from saved memory scenario_parallelism=${scenario_parallelism} quiz_workers=${quiz_workers}" | tee -a "${output_root}/luna-reader-scheduler.log"
for ((offset=0; offset<${#scenarios[@]}; offset+=scenario_parallelism)); do
  pids=()
  batch=("${scenarios[@]:offset:scenario_parallelism}")
  for scenario in "${batch[@]}"; do
    run_scenario "$scenario" &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" -ne 0 ]]; then
    echo "[$(date -Is)] HVP Luna reader batch failed: ${batch[*]}" | tee -a "${output_root}/luna-reader-scheduler.log" >&2
    exit 1
  fi
done
echo "[$(date -Is)] all HVP Luna Quiz reader scenarios completed" | tee -a "${output_root}/luna-reader-scheduler.log"
