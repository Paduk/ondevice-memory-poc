#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
python_bin=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
runner="${repo_root}/memory_training/scripts/run_granite4_350m_patch_full6_validation.sh"
gpu=7
first_run=granite4-350m-patch-multitask-noop5-full6val-grouped-v2-v1-10-e4-b8-trainseed45-r1
status_path="${workspace}/runs/${first_run}/status.json"

echo "[$(date -Is)] Waiting for ${first_run} on GPU ${gpu}."
while true; do
  state=MISSING
  if [[ -f "${status_path}" ]]; then
    state="$(${python_bin} -c 'import json,sys; print(json.load(open(sys.argv[1])).get("state", "UNKNOWN"))' "${status_path}")"
  fi
  case "${state}" in
    COMPLETED)
      while pgrep -f '[m]emory_training.train.*granite4-350m-patch-multitask-noop5-full6val.*trainseed45-r1' >/dev/null; do
        sleep 10
      done
      echo "[$(date -Is)] Starting the same Full-6 profile with seed 46 on GPU ${gpu}."
      cd "${repo_root}"
      exec bash "${runner}" "${gpu}" 4 46 r1
      ;;
    FAILED)
      echo "[$(date -Is)] Seed 45 failed; seed 46 will not start." >&2
      exit 1
      ;;
  esac
  sleep 30
done
