#!/usr/bin/env bash
set -euo pipefail

run_tag="${1:-20260915-v1}"
duration_minutes="${2:-300}"
interval_minutes="${3:-20}"

workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
campaign_root="${workspace}/benchmarks/test24-stress-composite-campaign-${run_tag}"
monitor_root="${campaign_root}/monitor"
log_path="${monitor_root}/snapshots.log"
latest_path="${monitor_root}/latest.txt"
if ((duration_minutes > 0)); then
  iterations=$((duration_minutes / interval_minutes))
else
  iterations=-1
fi

mkdir -p "${monitor_root}"

snapshot() {
  local temporary="${latest_path}.tmp"
  {
    echo "timestamp=$(date -Is)"
    echo "gpu_status"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader | awk -F, '$1==3 || $1==5'
    echo "sessions"
    tmux list-sessions 2>/dev/null | \
      rg '^(granite1b-test24-stress-composite|test24-stress-composite-queue):' || true
    echo "active_evaluations"
    pgrep -af '[m]emory_training.evaluate_hf_closed_loop.*test24-update-stress' || true
    echo "job_states"
    for state in "${campaign_root}"/state/*.state; do
      [[ -e "${state}" ]] || continue
      printf '%s\t' "$(basename "${state}")"
      cat "${state}"
    done
    echo "recent_progress"
    find "${workspace}/benchmarks" -path \
      "*test24-update-stress-${run_tag}/composite/update*/progress.json" \
      -mmin -30 -print0 2>/dev/null | sort -z | while IFS= read -r -d '' path; do
        printf '%s\t' "${path}"
        jq -c '{status,phase,progress,completed:(.completed_scenarios|length),current_scenarios,elapsed_seconds,eta_seconds,esm,update_f1}' "${path}" 2>/dev/null || true
      done
    echo "recent_errors"
    rg -n -i 'traceback|cuda out of memory|error:|failed GPU=' \
      "${campaign_root}/logs" \
      "${workspace}/benchmarks/granite4-1b-test24-update-stress-${run_tag}/logs" \
      2>/dev/null | tail -n 20 || true
  } >"${temporary}"
  mv "${temporary}" "${latest_path}"
  cat "${latest_path}" >>"${log_path}"
  echo >>"${log_path}"
}

campaign_finished() {
  local completion_state="${state_root:-${campaign_root}/state}/campaign-complete.state"
  [[ -s "${completion_state}" ]] || return 1
  [[ "$(cut -f1 "${completion_state}")" == 0 ]] || return 1
  ! pgrep -f '[m]emory_training.evaluate_hf_closed_loop.*test24-update-stress' >/dev/null
}

iteration=0
while true; do
  snapshot
  if ((duration_minutes == 0)); then
    if campaign_finished; then
      echo "campaign_monitoring_complete=$(date -Is)" >>"${log_path}"
      break
    fi
  elif ((iteration >= iterations)); then
    echo "monitoring_window_complete=$(date -Is)" >>"${log_path}"
    break
  fi

  iteration=$((iteration + 1))
  sleep "$((interval_minutes * 60))"
done
