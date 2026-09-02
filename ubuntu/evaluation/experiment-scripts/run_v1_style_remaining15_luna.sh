#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/hj153lee/PalmClaw"
python_bin="${repo_root}/.conda/ubuntu-agent/bin/python"
runner="${repo_root}/ubuntu/evaluation/experiment-scripts/run_vehiclemembench_v1_style_clone.py"
batch_root="/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v1-style-augmentation-v1"
monitor_log="${batch_root}/remaining15-luna-monitor.log"
scenarios=(9 11 12 15 16 18 21 25 26 27 28 37 38 46 49)
targets=(306 307 308 309 310 311 312 313 314 315 316 317 318 319 320)
max_parallel=5
next_index=0
tick=0
declare -A retries

start_job() {
  local scenario="$1"
  local target="$2"
  local session="v1-style15-luna-s${scenario}"
  local output="${batch_root}/s${scenario}-style-v2"
  local log="${batch_root}/s${scenario}-style-v2.log"
  tmux new-session -d -s "${session}" \
    "cd '${repo_root}' && exec '${python_bin}' '${runner}' --source-scenario '${scenario}' --target-scenario '${target}' --output-root '${output}' --model gpt-5.6-luna --workers 4 --max-attempts 4 2>&1 | tee -a '${log}'"
}

while true; do
  active=0
  complete=0
  failed=0
  for index in $(seq 0 $((next_index - 1)) 2>/dev/null || true); do
    scenario="${scenarios[$index]}"
    target="${targets[$index]}"
    session="v1-style15-luna-s${scenario}"
    output="${batch_root}/s${scenario}-style-v2"
    if [[ -f "${output}/COMPLETED" ]]; then
      complete=$((complete + 1))
    elif tmux has-session -t "${session}" 2>/dev/null; then
      active=$((active + 1))
    else
      attempt=$(( ${retries[$scenario]:-0} + 1 ))
      retries[$scenario]="${attempt}"
      if [[ "${attempt}" -le 5 ]]; then
        start_job "${scenario}" "${target}"
        active=$((active + 1))
      else
        failed=$((failed + 1))
      fi
    fi
  done

  while [[ "${active}" -lt "${max_parallel}" && "${next_index}" -lt "${#scenarios[@]}" ]]; do
    start_job "${scenarios[$next_index]}" "${targets[$next_index]}"
    next_index=$((next_index + 1))
    active=$((active + 1))
  done

  if [[ "${tick}" -eq 0 || $((tick % 10)) -eq 0 ]]; then
    line="$(date -u +%FT%TZ) queued=${next_index}/${#scenarios[@]} active=${active} complete=${complete} failed=${failed}"
    for index in $(seq 0 $((next_index - 1)) 2>/dev/null || true); do
      scenario="${scenarios[$index]}"
      output="${batch_root}/s${scenario}-style-v2"
      checkpoints=0
      expected="?"
      if [[ -d "${output}/dialogues" ]]; then
        checkpoints="$(find "${output}/dialogues" -maxdepth 1 -type f -name 'session-*.json' | wc -l)"
      fi
      if [[ -f "${output}/source-template.json" ]]; then
        expected="$(${python_bin} -c 'import json,sys; print(json.load(open(sys.argv[1]))["session_count"])' "${output}/source-template.json")"
      fi
      line+=" S${scenario}=${checkpoints}/${expected}"
    done
    echo "${line}" | tee -a "${monitor_log}"
  fi

  if [[ "${complete}" -eq "${#scenarios[@]}" ]]; then
    exit 0
  fi
  if [[ "${failed}" -gt 0 && "${active}" -eq 0 ]]; then
    exit 1
  fi
  tick=$((tick + 1))
  sleep 30
done
