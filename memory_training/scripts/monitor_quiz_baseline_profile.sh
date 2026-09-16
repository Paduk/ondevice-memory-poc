#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 PROFILE [INTERVAL_SECONDS]" >&2
  exit 2
fi

profile=$1
interval=${2:-300}
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
root=${workspace}/evaluations/quiz-only-e3-baselines
python=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
log=${root}/monitor-${profile}.log
models=(llama3.2-3b qwen3.5-2b granite4-1b qwen3.5-0.8b llama3.2-1b granite4-350m)

while true; do
  now=$(date -u +%FT%TZ)
  complete=0
  snapshot="${now} SAMPLE profile=${profile}"
  for model in "${models[@]}"; do
    progress=${root}/${model}/${profile}/progress.json
    summary=${root}/${model}/${profile}/summary.json
    if [[ -f ${summary} ]]; then
      status=$(${python} -c \
        'import json,sys; x=json.load(open(sys.argv[1])); print("COMPLETE {0}/{0} ESM={1:.4f}".format(x["tasks"],x["esm"]))' \
        "${summary}")
      complete=$((complete + 1))
    elif [[ -f ${progress} ]]; then
      status=$(${python} -c \
        'import json,sys; x=json.load(open(sys.argv[1])); print("{} {}/{}".format(x["status"],x["completed_quizzes"],x["total_quizzes"]))' \
        "${progress}")
    else
      status=QUEUED
    fi
    snapshot+=$'\n'"${model} ${status}"
  done
  snapshot+=$'\n'"$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)"
  printf '%s\n' "${snapshot}" | tee -a "${log}"
  if [[ ${complete} -eq ${#models[@]} ]]; then
    echo "${now} ALL_COMPLETE profile=${profile}" | tee -a "${log}"
    exit 0
  fi
  if ! pgrep -u "$(id -u)" -f "evaluate_quiz_baseline.*--profile ${profile}" >/dev/null; then
    echo "${now} STALLED profile=${profile}: incomplete with no evaluator" | tee -a "${log}"
    exit 1
  fi
  sleep "${interval}"
done
