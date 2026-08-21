#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly WORKER="${SCRIPT_DIR}/run_turnwise_combined_r2_sol_quiz_single.sh"
readonly SESSION_PREFIX="tw-combined-sol-quiz"
readonly EVAL_ROOT="/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench"
readonly CONTROLLER_ROOT="${EVAL_ROOT}/turnwise-combined-r2-memory-sol-quiz-2x-controller-20260818"

mkdir -p "${CONTROLLER_ROOT}"
exec > >(tee -a "${CONTROLLER_ROOT}/controller.log") 2>&1
echo "[$(date --iso-8601=seconds)] launching 20 frozen-memory SOL Quiz jobs"

for repetition in 1 2; do
  for scenario in $(seq 1 10); do
    session="${SESSION_PREFIX}-r${repetition}-s${scenario}"
    if tmux has-session -t "${session}" 2>/dev/null; then
      echo "Session already exists: ${session}" >&2
      exit 1
    fi
    tmux new-session -d -s "${session}" \
      "bash '${WORKER}' '${scenario}' '${repetition}'"
    echo "launched ${session}"
  done
done

touch "${CONTROLLER_ROOT}/ALL_STARTED"
echo "[$(date --iso-8601=seconds)] all jobs launched"
