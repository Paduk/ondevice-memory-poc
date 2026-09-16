#!/usr/bin/env bash
set -euo pipefail

gpu_a="${1:?First idle GPU index is required}"
gpu_b="${2:?Second idle GPU index is required}"
run_tag="${3:-r1}"

if [[ "${gpu_a}" == "${gpu_b}" ]]; then
  echo "Two distinct GPU indices are required." >&2
  exit 2
fi

repo_root=/home/hj153lee/PalmClaw
runner="${repo_root}/memory_training/scripts/run_granite4_1b_pending_sampling_ablation.sh"

for gpu in "${gpu_a}" "${gpu_b}"; do
  used_memory=$(
    nvidia-smi -i "${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits
  )
  if (( used_memory > 1024 )); then
    echo "GPU ${gpu} is no longer idle: ${used_memory} MiB is in use." >&2
    exit 2
  fi
done

cd "${repo_root}"
echo "[$(date -Is)] Starting paired pending-sampling queue on GPUs ${gpu_a}, ${gpu_b}."

(
  "${runner}" "${gpu_a}" unstratified 45 "${run_tag}"
  "${runner}" "${gpu_a}" depth-weighted 46 "${run_tag}"
  "${runner}" "${gpu_a}" unstratified 47 "${run_tag}"
) &
queue_a_pid=$!

(
  "${runner}" "${gpu_b}" depth-weighted 45 "${run_tag}"
  "${runner}" "${gpu_b}" unstratified 46 "${run_tag}"
  "${runner}" "${gpu_b}" depth-weighted 47 "${run_tag}"
) &
queue_b_pid=$!

set +e
wait "${queue_a_pid}"
queue_a_status=$?
wait "${queue_b_pid}"
queue_b_status=$?
set -e

if (( queue_a_status != 0 || queue_b_status != 0 )); then
  echo "[$(date -Is)] Queue failed: GPU ${gpu_a}=${queue_a_status}, GPU ${gpu_b}=${queue_b_status}." >&2
  exit 1
fi

echo "[$(date -Is)] All six pending-sampling runs completed."
