#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-stats}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python}"
PALMCLAW_ROOT="${PALMCLAW_ROOT:-/home/hj153lee/PalmClaw}"
SOURCE_ROOT="${SOURCE_ROOT:-/mnt/data/hj153lee/PalmClaw/evaluation/longmemeval-source}"
DATA_ROOT="${DATA_ROOT:-/mnt/data/hj153lee/PalmClaw/evaluation/longmemeval-patch-pilot-v1}"
WORKSPACE="${WORKSPACE:-/mnt/data/hj153lee/PalmClaw/evaluation/longmemeval-patch-workspace}"
MODEL="${MODEL:-granite4-1b}"
RUN_ID="${RUN_ID:-longmemeval-general-patch-${MODEL}-e3-alltrain}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-/mnt/data/hj153lee/PalmClaw/on-device-memory-training/runs/granite4-1b-patch-multitask-noop5-v1style20-grouped-v2-v1-10-e4-b12-trainseed45-r1/checkpoints/epoch-03}"

cd "$PALMCLAW_ROOT"

case "$ACTION" in
  prepare)
    "$PYTHON_BIN" -m memory_training.prepare_longmemeval_patch \
      --oracle "$SOURCE_ROOT/longmemeval_oracle.json" \
      --question-metadata "$SOURCE_ROOT/0822_all_500_questions_final_v2.json" \
      --full-data "$SOURCE_ROOT/longmemeval_s_cleaned.json" \
      --output "$DATA_ROOT" \
      --force
    ;;
  stats)
    DATA_ROOT="$DATA_ROOT" "$PYTHON_BIN" -c 'import json,os; p=os.environ["DATA_ROOT"]+"/manifest.json"; print(json.dumps(json.load(open(p))["statistics"], indent=2, ensure_ascii=False))'
    ;;
  train)
    if [[ ! -d "$INIT_CHECKPOINT/adapter" ]]; then
      echo "missing initialization adapter: $INIT_CHECKPOINT/adapter" >&2
      exit 2
    fi
    "$PYTHON_BIN" -m memory_training.train \
      --model "$MODEL" \
      --method general_patch \
      --init-checkpoint "$INIT_CHECKPOINT" \
      --workspace "$WORKSPACE" \
      --data-root "$DATA_ROOT" \
      --catalog-path "$DATA_ROOT/catalog.sqlite" \
      --run-id "$RUN_ID" \
      --epochs 3 \
      --noop-per-update 10 \
      --trajectory-fraction 0 \
      --multitask \
      --quiz-total-passes 3 \
      --skip-quiz-validation \
      --skip-epoch-validation \
      --skip-generation-validation \
      --eval-max-rows 14 \
      --eval-steps 0 \
      --save-steps 0
    ;;
  memory-test)
    CHECKPOINT="${CHECKPOINT:-$WORKSPACE/runs/$RUN_ID/checkpoints/epoch-03}"
    "$PYTHON_BIN" -m memory_training.evaluate_longmemeval_patch \
      --model "$MODEL" \
      --checkpoint "$CHECKPOINT" \
      --full-data "$SOURCE_ROOT/longmemeval_s_cleaned.json" \
      --test-questions "$DATA_ROOT/test_questions.jsonl" \
      --output "$WORKSPACE/runs/$RUN_ID/longmemeval-test" \
      --workspace "$WORKSPACE"
    ;;
  *)
    echo "usage: $0 {prepare|stats|train|memory-test}" >&2
    exit 2
    ;;
esac
