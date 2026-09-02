#!/usr/bin/env bash
set -euo pipefail

model="${1:?model id is required}"
run_id="${2:?run id is required}"
gpu="${3:?GPU index is required}"
method="${4:-patch}"

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
python_bin=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
status_path="${workspace}/runs/${run_id}/status.json"
test_script="${repo_root}/memory_training/scripts/run_qwen35_patch_tests.sh"
log_path="${workspace}/${run_id}-tests-gpu${gpu}-scheduler.log"

exec > >(tee -a "${log_path}") 2>&1
echo "[$(date -Is)] Waiting for ${run_id} to complete."

while true; do
  state=MISSING
  if [[ -f "${status_path}" ]]; then
    state="$(${python_bin} -c 'import json,sys; print(json.load(open(sys.argv[1])).get("state", "UNKNOWN"))' "${status_path}")"
  fi
  case "${state}" in
    COMPLETED)
      while pgrep -f "[m]emory_training.train.*--run-id ${run_id}" >/dev/null; do
        sleep 10
      done
      echo "[$(date -Is)] Training completed; starting Tests on GPU ${gpu}."
      exec bash "${test_script}" "${model}" "${run_id}" "${gpu}" "${method}"
      ;;
    FAILED)
      echo "[$(date -Is)] Training failed; Tests will not start." >&2
      exit 1
      ;;
  esac
  sleep 30
done
