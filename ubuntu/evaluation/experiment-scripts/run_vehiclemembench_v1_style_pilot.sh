#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/hj153lee/PalmClaw"
ubuntu_root="$repo_root/ubuntu"
palmclaw_python="${PALMCLAW_PYTHON_BIN:-$repo_root/.conda/ubuntu-agent/bin/python}"
output_root="${V1_STYLE_OUTPUT_ROOT:-/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v1-style-augmentation-v1}"
selection="$output_root/source-selection.json"
pilot_root="$output_root/pilot-s44-r1"
stage2_root="$pilot_root/stage2"
scenario_root="$pilot_root/scenario"
dialogue_root="$scenario_root/dialogues"
anchored_stage2="$scenario_root/stage2-v2-anchored.json"

mkdir -p "$stage2_root" "$scenario_root"
cd "$ubuntu_root"

echo "[$(date -u +%FT%TZ)] V1-style pilot Stage 2"
"$palmclaw_python" evaluation/experiment-scripts/run_vehiclemembench_v1_stage2_smoke.py \
  --dataset-root /home/hj153lee/VehicleMemBench \
  --output-root "$stage2_root" \
  --model gpt-5.6-terra \
  --candidate-group 144 \
  --scenario 44 \
  --style-blueprint "$selection" \
  --state-evolution \
  --timeout-seconds 1800 \
  --max-attempts 3

echo "[$(date -u +%FT%TZ)] V1-style pilot causal anchors"
anchor_args=(
  evaluation/experiment-scripts/prepare_vehiclemembench_v2_memory_anchors.py
  --stage2-path "$stage2_root/stage2.json"
  --output-root "$scenario_root"
  --model gpt-5.6-terra
  --timeout-seconds 1800
)
if [[ -f "$scenario_root/memory-anchors.json" ]]; then
  anchor_args+=(--anchor-checkpoint "$scenario_root/memory-anchors.json")
fi
"$palmclaw_python" "${anchor_args[@]}"

echo "[$(date -u +%FT%TZ)] V1-style pilot parallel dialogue generation"
"$palmclaw_python" evaluation/experiment-scripts/run_vehiclemembench_v1_stage3_smoke.py \
  --dataset-root /home/hj153lee/VehicleMemBench \
  --stage2-path "$anchored_stage2" \
  --output-root "$scenario_root/dialogue-preflight" \
  --dialogue-root "$dialogue_root" \
  --model gpt-5.6-terra \
  --scenario-id vehiclemembench-v1-style-pilot-s44-r1 \
  --public-scenario-index 301 \
  --workers 8 \
  --timeout-seconds 1800 \
  --max-attempts 3 \
  --dialogue-only

echo "[$(date -u +%FT%TZ)] V1-style pilot dialogues and turn-wise memory"
"$palmclaw_python" evaluation/experiment-scripts/run_vehiclemembench_v2_hybrid_smoke.py \
  --stage2-path "$anchored_stage2" \
  --output-root "$scenario_root" \
  --model gpt-5.6-terra \
  --timeout-seconds 1800 \
  --max-attempts 3

echo "[$(date -u +%FT%TZ)] V1-style pilot final Quiz"
"$palmclaw_python" evaluation/experiment-scripts/run_vehiclemembench_v1_stage3_smoke.py \
  --dataset-root /home/hj153lee/VehicleMemBench \
  --stage2-path "$anchored_stage2" \
  --output-root "$scenario_root/final-v1" \
  --dialogue-root "$dialogue_root" \
  --model gpt-5.6-terra \
  --scenario-id vehiclemembench-v1-style-pilot-s44-r1 \
  --public-scenario-index 301 \
  --workers 4 \
  --timeout-seconds 1800 \
  --max-attempts 3

echo "[$(date -u +%FT%TZ)] V1-style pilot audit and cost"
"$palmclaw_python" evaluation/experiment-scripts/audit_vehiclemembench_v1_style_pilot.py \
  --dataset-root /home/hj153lee/VehicleMemBench \
  --selection "$selection" \
  --pilot-root "$pilot_root"

touch "$pilot_root/COMPLETED"
echo "[$(date -u +%FT%TZ)] V1-style pilot completed"
