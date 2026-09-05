#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?GPU index is required}"
interval="${2:?Compaction interval is required}"
epochs="${3:-4}"
training_seed="${4:-45}"
run_tag="${5:-r1}"
resume_checkpoint="${6:-}"

/home/hj153lee/PalmClaw/memory_training/scripts/run_granite4_1b_delta_v3_compact_k_train.sh \
  "${gpu}" "${interval}" "${epochs}" "${training_seed}" "${run_tag}" "${resume_checkpoint}"

exec /home/hj153lee/PalmClaw/memory_training/scripts/evaluate_granite4_1b_delta_v3_compact_k.sh \
  "${gpu}" "${interval}" "${epochs}" "${training_seed}" "${run_tag}"
