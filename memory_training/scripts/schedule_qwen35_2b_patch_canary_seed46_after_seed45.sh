#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
python_bin=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
runner="${repo_root}/memory_training/scripts/run_qwen35_canary_variance.sh"
first_run=qwen35-2b-patch-canary12pct-opmatch-e4-trainseed45-r1
status_path="${workspace}/runs/${first_run}/status.json"
gpu=0

echo "[$(date -Is)] Waiting for ${first_run} on GPU ${gpu}."
while true; do
  state=MISSING
  if [[ -f "${status_path}" ]]; then
    state="$(${python_bin} -c 'import json,sys; print(json.load(open(sys.argv[1])).get("state", "UNKNOWN"))' "${status_path}")"
  fi
  case "${state}" in
    COMPLETED)
      while pgrep -f '[m]emory_training.train.*qwen35-2b-patch-canary12pct-opmatch-e4-trainseed45-r1' >/dev/null; do
        sleep 10
      done
      echo "[$(date -Is)] Starting seed 46 on GPU ${gpu}."
      cd "${repo_root}"
      exec bash "${runner}" "${gpu}" qwen3.5-2b patch 46 4 r1
      ;;
    FAILED)
      echo "[$(date -Is)] Seed 45 failed; seed 46 will not start." >&2
      exit 1
      ;;
  esac
  sleep 30
done
