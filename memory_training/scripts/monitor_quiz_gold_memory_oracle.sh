#!/usr/bin/env bash
set -euo pipefail

duration_seconds=${1:-7200}
interval_seconds=${2:-600}
if [[ ! ${duration_seconds} =~ ^[0-9]+$ || ! ${interval_seconds} =~ ^[1-9][0-9]*$ ]]; then
  echo "usage: $0 [DURATION_SECONDS] [INTERVAL_SECONDS]" >&2
  exit 2
fi

repo=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
output_root=${workspace}/evaluations/quiz-only-e3-baselines
runner=${repo}/memory_training/scripts/run_quiz_gold_memory_oracle.sh
state_dir=${output_root}/gold-memory-oracle-scheduler
monitor_log=${state_dir}/monitor.log

# Start larger models first to reduce the parallel schedule's tail latency.
models=(
  llama3.2-3b
  qwen3.5-2b
  llama3.2-1b
  granite4-1b
  qwen3.5-0.8b
  granite4-350m
)

mkdir -p "${state_dir}"
exec 8>"${state_dir}/monitor.lock"
if ! flock -n 8; then
  echo "A Gold-memory Oracle monitor is already running." >&2
  exit 3
fi

declare -A job_pid=()
declare -A job_gpu=()

timestamp() {
  date -u +%FT%TZ
}

log() {
  echo "$(timestamp) $*" | tee -a "${monitor_log}"
}

is_complete() {
  local model=$1
  local summary=${output_root}/${model}/gold_memory/summary.json
  [[ -f ${summary} ]] && jq -e \
    '.tasks == 960 and .profile == "gold_memory" and (.scenarios | length) == 24' \
    "${summary}" >/dev/null 2>&1
}

reap_jobs() {
  local model pid rc
  for model in "${!job_pid[@]}"; do
    pid=${job_pid[${model}]}
    if kill -0 "${pid}" 2>/dev/null; then
      continue
    fi
    if wait "${pid}"; then
      rc=0
    else
      rc=$?
    fi
    log "REAP model=${model} gpu=${job_gpu[${model}]} pid=${pid} rc=${rc}"
    unset 'job_pid['"${model}"']'
    unset 'job_gpu['"${model}"']'
  done
}

pending_models() {
  local model
  for model in "${models[@]}"; do
    if ! is_complete "${model}" && [[ -z ${job_pid[${model}]+x} ]]; then
      echo "${model}"
    fi
  done
}

empty_gpus() {
  local rows active_uuids gpu uuid memory
  rows=$(nvidia-smi \
    --query-gpu=index,uuid,memory.used \
    --format=csv,noheader,nounits)
  active_uuids=$(nvidia-smi \
    --query-compute-apps=gpu_uuid \
    --format=csv,noheader,nounits 2>/dev/null || true)
  while IFS=, read -r gpu uuid memory; do
    gpu=${gpu//[[:space:]]/}
    uuid=${uuid//[[:space:]]/}
    memory=${memory//[[:space:]]/}
    if [[ ${memory} =~ ^[0-9]+$ ]] \
      && (( memory <= 100 )) \
      && ! grep -Fxq "${uuid}" <<<"${active_uuids}"; then
      echo "${gpu}"
    fi
  done <<<"${rows}"
}

start_time=$(date +%s)
deadline=$((start_time + duration_seconds))
log "MONITOR_START duration=${duration_seconds}s interval=${interval_seconds}s deadline=$(date -u -d "@${deadline}" +%FT%TZ)"

while (( $(date +%s) <= deadline )); do
  reap_jobs

  completed=0
  for model in "${models[@]}"; do
    if is_complete "${model}"; then
      completed=$((completed + 1))
    fi
  done
  if (( completed == ${#models[@]} )); then
    log "ALL_COMPLETE models=${completed}"
    exit 0
  fi

  mapfile -t pending < <(pending_models)
  mapfile -t available < <(empty_gpus)
  log "POLL completed=${completed}/${#models[@]} running=${#job_pid[@]} pending=${#pending[@]} empty_gpus=${available[*]:-none}"

  launches=${#pending[@]}
  if (( ${#available[@]} < launches )); then
    launches=${#available[@]}
  fi
  for ((i = 0; i < launches; i++)); do
    model=${pending[${i}]}
    gpu=${available[${i}]}
    queue_log=${state_dir}/${model}-gpu${gpu}.queue.log
    nohup "${runner}" "${gpu}" "${model}" >>"${queue_log}" 2>&1 </dev/null &
    pid=$!
    job_pid[${model}]=${pid}
    job_gpu[${model}]=${gpu}
    log "LAUNCH model=${model} gpu=${gpu} pid=${pid}"
  done

  remaining=$((deadline - $(date +%s)))
  if (( remaining <= 0 )); then
    break
  fi
  sleep_for=${interval_seconds}
  if (( remaining < sleep_for )); then
    sleep_for=${remaining}
  fi
  sleep "${sleep_for}"
done

reap_jobs
log "MONITOR_STOP deadline_reached=true running_jobs=${#job_pid[@]}; launched jobs are left running"
