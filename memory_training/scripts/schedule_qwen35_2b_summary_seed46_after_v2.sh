#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
python_bin=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
source_run=qwen35-2b-summary-multitask-noop5-grouped-v2-v1-10-e4-b2-r1
v2_progress="${workspace}/runs/${source_run}/test-hf-closed-loop-s86-s100-t12-t20-best-epoch-04/progress.json"
test_tmux=qwen35-2b-summary-tests-after-train-gpu3
runner="${repo_root}/memory_training/scripts/run_qwen35_2b_summary.sh"
gpu=3

echo "[$(date -Is)] Waiting for the 2B Summary V2 Test to complete."
while true; do
  state=MISSING
  if [[ -f "${v2_progress}" ]]; then
    state="$(${python_bin} -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status", "UNKNOWN"))' "${v2_progress}")"
  fi
  case "${state}" in
    COMPLETED)
      # The old test runner would immediately continue into V1. Stop that chain;
      # if V1 has already begun loading, terminating the tmux pane stops it too.
      tmux kill-session -t "${test_tmux}" 2>/dev/null || true
      while pgrep -f '[m]emory_training.evaluate_hf_.*qwen35-2b-summary-multitask-noop5-grouped-v2-v1-10-e4-b2-r1' >/dev/null; do
        sleep 2
      done
      echo "[$(date -Is)] V1 Test cancelled; starting Summary seed 46 on GPU ${gpu}."
      cd "${repo_root}"
      exec bash "${runner}" "${gpu}" 4 46 r2
      ;;
    FAILED)
      echo "[$(date -Is)] V2 Test failed; seed-46 training will not start." >&2
      exit 1
      ;;
  esac
  sleep 2
done
