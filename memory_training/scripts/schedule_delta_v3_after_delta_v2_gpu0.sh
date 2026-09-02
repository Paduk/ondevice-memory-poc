#!/usr/bin/env bash
set -euo pipefail

workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
source_run=qwen35-4b-delta-v2-multitask-noop5-grouped-v2-v1-10-r1
status_path="${workspace}/runs/${source_run}/status.json"
runner=/home/hj153lee/PalmClaw/memory_training/scripts/run_qwen35_4b_delta_v3.sh
log_path="${workspace}/delta-v3-e5-after-delta-v2-gpu0-scheduler.log"

exec > >(tee -a "${log_path}") 2>&1
echo "[$(date -Is)] Waiting for Delta-v2 Epoch 4 Validation on GPU 0."
while true; do
  state=MISSING
  if [[ -f "${status_path}" ]]; then
    state=$(jq -r '.state // "UNKNOWN"' "${status_path}")
  fi
  case "${state}" in
    COMPLETED) break ;;
    FAILED)
      echo "[$(date -Is)] Delta-v2 Validation failed; Delta-v3 will not start." >&2
      exit 1
      ;;
  esac
  sleep 30
done

while pgrep -f "memory_training.validate_hf_checkpoint_v2.*${source_run}" >/dev/null; do
  sleep 10
done

echo "[$(date -Is)] Delta-v2 Validation complete; starting Delta-v3 E5 on GPU 0."
exec bash "${runner}" 0 5
