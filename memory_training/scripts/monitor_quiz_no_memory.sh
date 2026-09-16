#!/usr/bin/env bash
set -euo pipefail

workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
root=${workspace}/evaluations/quiz-only-e3-baselines
log=${root}/monitor-no_memory.log
interval=${QUIZ_BASELINE_MONITOR_INTERVAL_SECONDS:-1800}
models=(llama3.2-3b qwen3.5-2b granite4-1b granite4-350m qwen3.5-0.8b llama3.2-1b)

mkdir -p "${root}"
while true; do
  now=$(date -u +%FT%TZ)
  complete=0
  {
    echo "${now} SAMPLE"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
    for model in "${models[@]}"; do
      progress=${root}/${model}/no_memory/progress.json
      summary=${root}/${model}/no_memory/summary.json
      if [[ -f "${summary}" ]]; then
        complete=$((complete + 1))
        status=$(/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python -c \
          'import json,sys; x=json.load(open(sys.argv[1])); print(f"COMPLETE tasks={x[\"tasks\"]} esm={x[\"esm\"]:.4f} f1={x[\"tool_f1\"]:.4f}")' \
          "${summary}")
      elif [[ -f "${progress}" ]]; then
        status=$(/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python -c \
          'import json,sys; x=json.load(open(sys.argv[1])); print(f"{x[\"status\"]} quizzes={x[\"completed_quizzes\"]}/{x[\"total_quizzes\"]}")' \
          "${progress}")
      else
        status=QUEUED
      fi
      echo "${model} ${status}"
    done
  } >>"${log}" 2>&1
  if [[ ${complete} -eq ${#models[@]} ]]; then
    echo "${now} ALL_COMPLETE" >>"${log}"
    exit 0
  fi
  sleep "${interval}"
done
