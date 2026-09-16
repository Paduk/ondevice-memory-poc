#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/hj153lee/PalmClaw
pipeline="${repo_root}/memory_training/scripts/run_llama32_1b_patch_delta_k5_pipeline.sh"
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
monitor_dir="${workspace}/reports/schedulers"
monitor_log="${monitor_dir}/llama32-1b-k2-k10-20260907.log"
poll_seconds=600
duration_seconds=14400
started_at=$(date +%s)
deadline=$((started_at + duration_seconds))

mkdir -p "${monitor_dir}"
exec > >(tee -a "${monitor_log}") 2>&1

queue=(2 10)
queue_index=0
declare -A launched_gpu=()
declare -A launched_session=()
mapfile -t gpu_uuids < <(
  nvidia-smi --query-gpu=uuid --format=csv,noheader,nounits
)
compute_uuids=""

tmux_session_exists() {
  tmux has-session -t "$1" 2>/dev/null
}

gpu_is_reserved() {
  local gpu=$1 k session
  if [[ "${gpu}" == 0 ]] && tmux_session_exists llama32_1b_patch_r2_20260907; then
    return 0
  fi
  if [[ "${gpu}" == 5 ]] && tmux_session_exists llama32_1b_k5_r2_20260907; then
    return 0
  fi
  for k in 2 10; do
    session="${launched_session[$k]:-}"
    if [[ "${launched_gpu[$k]:-}" == "${gpu}" && -n "${session}" ]] \
      && tmux_session_exists "${session}"; then
      return 0
    fi
  done
  return 1
}

gpu_has_compute_process() {
  local gpu=$1
  grep -Fqx "${gpu_uuids[$gpu]}" <<<"${compute_uuids}"
}

refresh_compute_processes() {
  compute_uuids=$(
    nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits
  )
}

gpu_is_free() {
  local gpu=$1 used
  gpu_is_reserved "${gpu}" && return 1
  gpu_has_compute_process "${gpu}" && return 1
  used=$(nvidia-smi -i "${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits)
  (( used < 512 ))
}

find_free_gpu() {
  local gpu
  for gpu in $(seq 0 7); do
    if gpu_is_free "${gpu}"; then
      sleep 5
      refresh_compute_processes
      if gpu_is_free "${gpu}"; then
        echo "${gpu}"
        return 0
      fi
    fi
  done
  return 1
}

launch_next() {
  local gpu k session run_dir
  gpu=$1
  k="${queue[$queue_index]}"
  session="llama32_1b_k${k}_auto_20260907_gpu${gpu}"
  run_dir="${workspace}/runs/llama3.2-1b-delta_v3_compact_k${k}-multitask-noop5-e4-b2-trainseed45-evalfixed-noop5-r1"
  if [[ -s "${run_dir}/status.json" ]] \
    && [[ $(jq -r '.state // empty' "${run_dir}/status.json") == COMPLETED ]]; then
    echo "[$(date -Is)] k=${k} is already completed; advancing queue."
  else
    echo "[$(date -Is)] GPU ${gpu} confirmed free; launching k=${k} as ${session}."
    tmux new-session -d -s "${session}" \
      "exec ${pipeline} ${gpu} delta_v3_compact_k${k} 4 45 r1"
    launched_gpu[$k]="${gpu}"
    launched_session[$k]="${session}"
    sleep 30
    if ! tmux_session_exists "${session}"; then
      echo "[$(date -Is)] k=${k} exited during startup; retaining it at the queue head." >&2
      return 1
    fi
    echo "[$(date -Is)] k=${k} startup survived 30s on GPU ${gpu}."
  fi
  queue_index=$((queue_index + 1))
}

echo "[$(date -Is)] Monitoring started: interval=10m, window=4h, queue=k2,k10."
while (( $(date +%s) <= deadline )); do
  echo "[$(date -Is)] Poll $((($(date +%s) - started_at) / poll_seconds + 1)); queue_index=${queue_index}."
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits
  refresh_compute_processes
  while (( queue_index < ${#queue[@]} )); do
    if gpu=$(find_free_gpu); then
      if launch_next "${gpu}"; then
        # Admit at most one new run per poll so its first step is observable
        # before the next queued method is assigned.
        break
      fi
      break
    else
      echo "[$(date -Is)] No unreserved process-free GPU is available."
      break
    fi
  done
  if (( queue_index >= ${#queue[@]} )); then
    echo "[$(date -Is)] Both k=2 and k=10 have been scheduled; monitor completed."
    exit 0
  fi
  remaining=$((deadline - $(date +%s)))
  (( remaining > 0 )) || break
  if (( remaining < poll_seconds )); then
    sleep "${remaining}"
  else
    sleep "${poll_seconds}"
  fi
done

echo "[$(date -Is)] Four-hour window ended with unscheduled queue: ${queue[*]:queue_index}."
exit 1
