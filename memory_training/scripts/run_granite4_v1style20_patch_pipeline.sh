#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?GPU index is required}"
model="${2:?Model key is required}"
epochs="${3:-4}"
training_seed="${4:-45}"
run_tag="${5:-r1}"
case "${model}" in
  granite4-350m) batch_size=16 ;;
  granite4-1b) batch_size=12 ;;
  *) echo "Unsupported model: ${model}" >&2; exit 2 ;;
esac
run_id="${model}-patch-multitask-noop5-v1style20-grouped-v2-v1-10-e${epochs}-b${batch_size}-trainseed${training_seed}-${run_tag}"

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
controller_log="${workspace}/controllers/${run_id}.log"
mkdir -p "$(dirname "${controller_log}")"
exec > >(tee -a "${controller_log}") 2>&1

echo "[$(date -Is)] Pipeline started: ${run_id}, GPU ${gpu}."
bash "${repo_root}/memory_training/scripts/run_granite4_v1style20_patch_train.sh" \
  "${gpu}" "${model}" "${epochs}" "${training_seed}" "${run_tag}"
bash "${repo_root}/memory_training/scripts/run_granite4_v1style20_patch_post.sh" \
  "${gpu}" "${model}" "${run_id}" "${epochs}"
echo "[$(date -Is)] Pipeline completed: ${run_id}."
