#!/usr/bin/env bash
set -euo pipefail

gpu=${1:?GPU index required}
shift
if (( $# == 0 )); then
  echo "Command required" >&2
  exit 2
fi

confirmations=0
echo "[$(date -Is)] Waiting for GPU ${gpu} to become truly idle."
while (( confirmations < 2 )); do
  snapshot=$(nvidia-smi -i "${gpu}" \
    --query-gpu=memory.used,utilization.gpu \
    --format=csv,noheader,nounits)
  memory=$(awk -F',' '{gsub(/ /, "", $1); print $1}' <<<"${snapshot}")
  utilization=$(awk -F',' '{gsub(/ /, "", $2); print $2}' <<<"${snapshot}")
  pids=$(nvidia-smi -i "${gpu}" --query-compute-apps=pid \
    --format=csv,noheader,nounits | awk 'NF' || true)
  if [[ -z "${pids}" ]] && (( memory < 512 && utilization < 5 )); then
    confirmations=$((confirmations + 1))
  else
    confirmations=0
  fi
  echo "[$(date -Is)] GPU ${gpu}: memory=${memory}MiB util=${utilization}% pids=${pids:-none} confirmation=${confirmations}/2"
  if (( confirmations < 2 )); then
    sleep 60
  fi
done

echo "[$(date -Is)] GPU ${gpu} confirmed idle; launching command."
exec "$@"
