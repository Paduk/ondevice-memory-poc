#!/usr/bin/env bash
set -euo pipefail

METHOD="${1:?usage: run_cloud_t12_luna_smoke.sh METHOD}"
PROFILE="${2:-filtered}"
case "$METHOD" in
  summary|patch|delta_v3) ;;
  *) echo "unknown method: $METHOD" >&2; exit 2 ;;
esac

REPO=/home/hj153lee/PalmClaw
PYTHON=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
OUTPUT_ROOT=/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-cloud-ondevice-prompt-t12-luna-none-smoke-20260831
OUTPUT_NAME="$METHOD"
EXTRA_ARGS=()
if [[ "$METHOD" == "summary" ]]; then
  OUTPUT_NAME=summary_compact600_filtered
  EXTRA_ARGS+=(--summary-memory-token-budget 600 --vehicle-memory-filter)
elif [[ "$METHOD" == "patch" ]]; then
  OUTPUT_NAME=patch_filtered
  EXTRA_ARGS+=(--vehicle-memory-filter)
elif [[ "$METHOD" == "delta_v3" ]]; then
  OUTPUT_NAME=delta_v3_filtered
  EXTRA_ARGS+=(--vehicle-memory-filter)
fi
if [[ "$PROFILE" == "semantic5" ]]; then
  if [[ "$METHOD" == "summary" ]]; then
    echo "semantic5 supports patch and delta_v3 only" >&2
    exit 2
  fi
  OUTPUT_NAME="${OUTPUT_NAME}_semantic5"
  EXTRA_ARGS+=(--semantic-compaction-every 5 --semantic-compaction-token-budget 600)
elif [[ "$PROFILE" != "filtered" ]]; then
  echo "unknown profile: $PROFILE" >&2
  exit 2
fi

cd "$REPO"
exec "$PYTHON" -m memory_training.evaluate_cloud_closed_loop \
  --method "$METHOD" \
  --scenario 112 \
  --model gpt-5.6-luna \
  --timeout-seconds 60 \
  --reasoning-effort none \
  --output-dir "$OUTPUT_ROOT/$OUTPUT_NAME" \
  "${EXTRA_ARGS[@]}"
