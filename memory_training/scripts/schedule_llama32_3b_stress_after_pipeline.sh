#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?GPU index is required}"
method="${2:?Method is required}"
pipeline_session="${3:?Pipeline tmux session is required}"
extra_blocking_session="${4:-}"

runner=/home/hj153lee/PalmClaw/memory_training/scripts/run_llama32_3b_stress_test_once.sh

while tmux has-session -t "${pipeline_session}" 2>/dev/null; do
  sleep 30
done
if [[ -n "${extra_blocking_session}" ]]; then
  while tmux has-session -t "${extra_blocking_session}" 2>/dev/null; do
    sleep 30
  done
fi

echo "[$(date -Is)] prerequisites released; starting GPU=${gpu} method=${method}"
exec "${runner}" "${gpu}" "${method}"
