#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]] || [[ "$1" -lt 2 ]] || [[ "$1" -gt 100 ]]; then
  echo "usage: $0 SCENARIO_INDEX (2..100)" >&2
  exit 2
fi

scenario_index="$1"
scenario_tag="$(printf '%02d' "$scenario_index")"
repo_root="/home/hj153lee/PalmClaw"
ubuntu_root="$repo_root/ubuntu"
palmclaw_python="${PALMCLAW_PYTHON_BIN:-$repo_root/.conda/ubuntu-agent/bin/python}"
v1_root="/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v1-reproduction"
v2_root="/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-hybrid"
stage2_root="$v1_root/stage2-pilot-terra-s${scenario_tag}-r1"
source_root="$v2_root/hybrid-pilot-terra-s${scenario_tag}-r1"
output_root="$v2_root/hybrid-pilot-terra-s${scenario_tag}-r2"
dialogue_root="$source_root/dialogues"
anchored_stage2="$source_root/stage2-v2-anchored.json"

if [[ ! -f "$stage2_root/stage2.json" ]] \
  || [[ ! -f "$source_root/memory-anchors.json" ]] \
  || [[ ! -f "$source_root/hybrid.json" ]]; then
  echo "S${scenario_tag} source Stage 2/anchor/Hybrid is incomplete" >&2
  exit 1
fi

mkdir -p "$output_root"
cd "$ubuntu_root"

echo "[$(date -u +%FT%TZ)] S${scenario_tag} rebuild normalized Hybrid memory"
"$palmclaw_python" evaluation/experiment-scripts/run_vehiclemembench_v2_hybrid_smoke.py \
  --stage2-path "$anchored_stage2" \
  --output-root "$output_root" \
  --dialogue-root "$dialogue_root" \
  --model gpt-5.6-terra \
  --timeout-seconds 1800 \
  --max-attempts 3

echo "[$(date -u +%FT%TZ)] S${scenario_tag} immediate Turn Quiz"
"$palmclaw_python" evaluation/experiment-scripts/run_vehiclemembench_v2_turn_quiz_smoke.py \
  --dataset-root /home/hj153lee/VehicleMemBench \
  --stage2-path "$anchored_stage2" \
  --hybrid-artifact "$output_root/hybrid.json" \
  --dialogue-root "$dialogue_root" \
  --output-root "$output_root/turn-quiz-v4" \
  --model gpt-5.6-terra \
  --workers 4 \
  --timeout-seconds 1800 \
  --max-attempts 3

echo "[$(date -u +%FT%TZ)] S${scenario_tag} expanded Turn Quiz 30"
"$palmclaw_python" evaluation/experiment-scripts/run_vehiclemembench_v2_turn_quiz_expansion.py \
  --dataset-root /home/hj153lee/VehicleMemBench \
  --stage2-path "$anchored_stage2" \
  --hybrid-artifact "$output_root/hybrid.json" \
  --dialogue-root "$dialogue_root" \
  --immediate-artifact "$output_root/turn-quiz-v4/turn-quizzes.json" \
  --output-root "$output_root/turn-quiz-30-v1" \
  --target-quiz-count 30 \
  --model gpt-5.6-terra \
  --workers 4 \
  --timeout-seconds 1800 \
  --max-attempts 3

echo "[$(date -u +%FT%TZ)] S${scenario_tag} final V1 Quiz"
"$palmclaw_python" evaluation/experiment-scripts/run_vehiclemembench_v1_stage3_smoke.py \
  --dataset-root /home/hj153lee/VehicleMemBench \
  --stage2-path "$anchored_stage2" \
  --output-root "$output_root/final-v1" \
  --dialogue-root "$dialogue_root" \
  --model gpt-5.6-terra \
  --scenario-id "vehiclemembench-v2-hybrid-s${scenario_tag}-r2" \
  --public-scenario-index "$scenario_index" \
  --workers 4 \
  --timeout-seconds 1800 \
  --max-attempts 3

echo "[$(date -u +%FT%TZ)] S${scenario_tag} normalized pipeline completed"
