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
hybrid_root="$v2_root/hybrid-pilot-terra-s${scenario_tag}-r1"
dialogue_root="$hybrid_root/dialogues"
anchored_stage2="$hybrid_root/stage2-v2-anchored.json"

mkdir -p "$stage2_root" "$hybrid_root"
cd "$ubuntu_root"

echo "[$(date -u +%FT%TZ)] S${scenario_tag} stage2"
stage2_profile_args=()
if (( scenario_index >= 21 )); then
  stage2_profile_args+=(--state-evolution)
fi
"$palmclaw_python" evaluation/experiment-scripts/run_vehiclemembench_v1_stage2_smoke.py \
  --dataset-root /home/hj153lee/VehicleMemBench \
  --output-root "$stage2_root" \
  --model gpt-5.6-terra \
  --candidate-group "$scenario_index" \
  --scenario "$scenario_index" \
  --timeout-seconds 1800 \
  --max-attempts 3 \
  "${stage2_profile_args[@]}"

echo "[$(date -u +%FT%TZ)] S${scenario_tag} causal anchors"
anchor_args=(
  evaluation/experiment-scripts/prepare_vehiclemembench_v2_memory_anchors.py
  --stage2-path "$stage2_root/stage2.json"
  --output-root "$hybrid_root"
  --model gpt-5.6-terra
  --timeout-seconds 1800
)
if [[ -f "$hybrid_root/memory-anchors.json" ]]; then
  anchor_args+=(--anchor-checkpoint "$hybrid_root/memory-anchors.json")
fi
"$palmclaw_python" "${anchor_args[@]}"

echo "[$(date -u +%FT%TZ)] S${scenario_tag} hybrid dialogues and memory"
"$palmclaw_python" evaluation/experiment-scripts/run_vehiclemembench_v2_hybrid_smoke.py \
  --stage2-path "$anchored_stage2" \
  --output-root "$hybrid_root" \
  --model gpt-5.6-terra \
  --timeout-seconds 1800 \
  --max-attempts 3

echo "[$(date -u +%FT%TZ)] S${scenario_tag} immediate Turn Quiz"
"$palmclaw_python" evaluation/experiment-scripts/run_vehiclemembench_v2_turn_quiz_smoke.py \
  --dataset-root /home/hj153lee/VehicleMemBench \
  --stage2-path "$anchored_stage2" \
  --hybrid-artifact "$hybrid_root/hybrid.json" \
  --dialogue-root "$dialogue_root" \
  --output-root "$hybrid_root/turn-quiz-v4" \
  --model gpt-5.6-terra \
  --workers 4 \
  --timeout-seconds 1800 \
  --max-attempts 3

echo "[$(date -u +%FT%TZ)] S${scenario_tag} expanded Turn Quiz 30"
"$palmclaw_python" evaluation/experiment-scripts/run_vehiclemembench_v2_turn_quiz_expansion.py \
  --dataset-root /home/hj153lee/VehicleMemBench \
  --stage2-path "$anchored_stage2" \
  --hybrid-artifact "$hybrid_root/hybrid.json" \
  --dialogue-root "$dialogue_root" \
  --immediate-artifact "$hybrid_root/turn-quiz-v4/turn-quizzes.json" \
  --output-root "$hybrid_root/turn-quiz-30-v1" \
  --target-quiz-count 30 \
  --model gpt-5.6-terra \
  --workers 4 \
  --timeout-seconds 1800 \
  --max-attempts 3

echo "[$(date -u +%FT%TZ)] S${scenario_tag} final V1 Quiz"
"$palmclaw_python" evaluation/experiment-scripts/run_vehiclemembench_v1_stage3_smoke.py \
  --dataset-root /home/hj153lee/VehicleMemBench \
  --stage2-path "$anchored_stage2" \
  --output-root "$hybrid_root/final-v1" \
  --dialogue-root "$dialogue_root" \
  --model gpt-5.6-terra \
  --scenario-id "vehiclemembench-v2-hybrid-s${scenario_tag}-r1" \
  --public-scenario-index "$scenario_index" \
  --workers 4 \
  --timeout-seconds 1800 \
  --max-attempts 3

echo "[$(date -u +%FT%TZ)] S${scenario_tag} completed"
