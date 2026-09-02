#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]] || [[ "$1" -lt 1 ]] || [[ "$1" -gt 20 ]]; then
  echo "usage: $0 TEMPORAL_SCENARIO_INDEX (1..20) [RUN_REVISION]" >&2
  exit 2
fi

scenario_index="$1"
scenario_tag="$(printf '%02d' "$scenario_index")"
run_revision="${2:-1}"
if [[ ! "$run_revision" =~ ^[1-9][0-9]*$ ]]; then
  echo "RUN_REVISION must be a positive integer" >&2
  exit 2
fi
repo_root="/home/hj153lee/PalmClaw"
ubuntu_root="$repo_root/ubuntu"
palmclaw_python="${PALMCLAW_PYTHON_BIN:-$repo_root/.conda/ubuntu-agent/bin/python}"
temporal_root="/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-temporal"
stage2_root="$temporal_root/stage2-temporal-terra-t${scenario_tag}-r${run_revision}"
hybrid_root="$temporal_root/hybrid-temporal-anchor-terra-t${scenario_tag}-r${run_revision}"
dialogue_root="$hybrid_root/dialogues"
anchored_stage2="$hybrid_root/stage2-v2-anchored.json"

mkdir -p "$stage2_root" "$hybrid_root"
cd "$ubuntu_root"

echo "[$(date -u +%FT%TZ)] T${scenario_tag} temporal Stage 2"
"$palmclaw_python" \
  evaluation/experiment-scripts/run_vehiclemembench_v2_temporal_stage2.py \
  --dataset-root /home/hj153lee/VehicleMemBench \
  --output-root "$stage2_root" \
  --model gpt-5.6-terra \
  --scenario "$scenario_index" \
  --timeout-seconds 1800 \
  --max-attempts 3

cp -f "$stage2_root/temporal-plan.json" "$hybrid_root/temporal-plan.json"
cp -f "$stage2_root/temporal-anchor-plan.json" \
  "$hybrid_root/temporal-anchor-plan.json"
cp -f "$stage2_root/stage2.json" "$anchored_stage2"

echo "[$(date -u +%FT%TZ)] T${scenario_tag} Hybrid dialogue and memory"
"$palmclaw_python" \
  evaluation/experiment-scripts/run_vehiclemembench_v2_hybrid_smoke.py \
  --stage2-path "$anchored_stage2" \
  --output-root "$hybrid_root" \
  --model gpt-5.6-terra \
  --timeout-seconds 1800 \
  --max-attempts 3

echo "[$(date -u +%FT%TZ)] T${scenario_tag} immediate Turn Quiz"
"$palmclaw_python" \
  evaluation/experiment-scripts/run_vehiclemembench_v2_turn_quiz_smoke.py \
  --dataset-root /home/hj153lee/VehicleMemBench \
  --stage2-path "$anchored_stage2" \
  --hybrid-artifact "$hybrid_root/hybrid.json" \
  --dialogue-root "$dialogue_root" \
  --output-root "$hybrid_root/turn-quiz-v4" \
  --model gpt-5.6-luna \
  --workers 4 \
  --timeout-seconds 1800 \
  --max-attempts 3

echo "[$(date -u +%FT%TZ)] T${scenario_tag} expanded Turn Quiz 30"
"$palmclaw_python" \
  evaluation/experiment-scripts/run_vehiclemembench_v2_turn_quiz_expansion.py \
  --dataset-root /home/hj153lee/VehicleMemBench \
  --stage2-path "$anchored_stage2" \
  --hybrid-artifact "$hybrid_root/hybrid.json" \
  --dialogue-root "$dialogue_root" \
  --immediate-artifact "$hybrid_root/turn-quiz-v4/turn-quizzes.json" \
  --output-root "$hybrid_root/turn-quiz-30-v1" \
  --target-quiz-count 30 \
  --model gpt-5.6-luna \
  --workers 4 \
  --timeout-seconds 1800 \
  --max-attempts 3

echo "[$(date -u +%FT%TZ)] T${scenario_tag} final Quiz 10"
"$palmclaw_python" \
  evaluation/experiment-scripts/run_vehiclemembench_v1_stage3_smoke.py \
  --dataset-root /home/hj153lee/VehicleMemBench \
  --stage2-path "$anchored_stage2" \
  --output-root "$hybrid_root/final-v1" \
  --dialogue-root "$dialogue_root" \
  --model gpt-5.6-luna \
  --scenario-id \
    "vehiclemembench-v2-temporal-anchor-t${scenario_tag}-r${run_revision}" \
  --public-scenario-index "$scenario_index" \
  --workers 4 \
  --timeout-seconds 1800 \
  --max-attempts 3

echo "[$(date -u +%FT%TZ)] T${scenario_tag} completed"
