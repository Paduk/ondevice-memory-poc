#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/hj153lee/PalmClaw"
python_bin="${repo_root}/.conda/ubuntu-agent/bin/python"
runner="${repo_root}/ubuntu/evaluation/experiment-scripts/run_vehiclemembench_v1_style_clone.py"
batch_root="/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v1-style-augmentation-v1"
monitor_log="${batch_root}/first4-luna-monitor.log"
scenarios=(3 4 7 8)
targets=(302 303 304 305)

start_job() {
  local scenario="$1"
  local target="$2"
  local session="v1-style-luna-s${scenario}"
  local output="${batch_root}/s${scenario}-style-v2"
  local log="${batch_root}/s${scenario}-style-v2.log"
  if tmux has-session -t "${session}" 2>/dev/null; then
    return
  fi
  tmux new-session -d -s "${session}" \
    "cd '${repo_root}' && exec '${python_bin}' '${runner}' --source-scenario '${scenario}' --target-scenario '${target}' --output-root '${output}' --model gpt-5.6-luna --workers 4 --max-attempts 4 2>&1 | tee -a '${log}'"
}

for index in "${!scenarios[@]}"; do
  start_job "${scenarios[$index]}" "${targets[$index]}"
done

while true; do
  complete=0
  line="$(date -u +%FT%TZ)"
  for index in "${!scenarios[@]}"; do
    scenario="${scenarios[$index]}"
    target="${targets[$index]}"
    session="v1-style-luna-s${scenario}"
    output="${batch_root}/s${scenario}-style-v2"
    if [[ -d "${output}/dialogues" ]]; then
      checkpoints="$(find "${output}/dialogues" -maxdepth 1 -type f -name 'session-*.json' | wc -l)"
    else
      checkpoints=0
    fi
    expected="?"
    if [[ -f "${output}/source-template.json" ]]; then
      expected="$(${python_bin} -c 'import json,sys; print(json.load(open(sys.argv[1]))["session_count"])' "${output}/source-template.json")"
    fi
    if [[ -f "${output}/COMPLETED" ]]; then
      state="COMPLETED"
      complete=$((complete + 1))
    elif tmux has-session -t "${session}" 2>/dev/null; then
      state="RUNNING"
    else
      state="RESTARTED"
      start_job "${scenario}" "${target}"
    fi
    line+=" S${scenario}=${state}:${checkpoints}/${expected}"
  done
  echo "${line}" | tee -a "${monitor_log}"
  if [[ "${complete}" -eq "${#scenarios[@]}" ]]; then
    break
  fi
  sleep 300
done
