#!/usr/bin/env bash
set -euo pipefail

run_tag="${1:-20260915-v1}"

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
data_parent=/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training
python_bin=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
source_root="${data_parent}/grouped-v2-v1-10-eval-fixed-noop5-seed45-v1"
manifest_root="${workspace}/benchmarks/natural-test24-manifests-${run_tag}"

scenarios=(
  86 87 88 89 90 91 92 93 94 95 96 97 98 99 100
  112 113 114 115 116 117 118 119 120
)
loads=(40 60 80)
intervals=(2 5 10)

cd "${repo_root}"
export PYTHONPATH="${repo_root}:${repo_root}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${manifest_root}"

prepare_natural_manifest() {
  local label=$1
  local data_root=$2
  local output="${manifest_root}/${label}.json"
  if [[ -s "${output}" ]]; then
    echo "[$(date -Is)] reuse natural Test24 manifest ${label}"
    return
  fi
  "${python_bin}" -m memory_training.cache_benchmark_manifest \
    --output "${output}" \
    --workspace "${workspace}" \
    --data-root "${data_root}" \
    --catalog-path "${data_root}/catalog.sqlite" \
    --scenarios "${scenarios[@]}" \
    --split all \
    --quiz-split test \
    --noop-per-update 5 \
    --sampling-seed 45
}

prepare_natural_manifest base "${source_root}"
for interval in "${intervals[@]}"; do
  prepare_natural_manifest \
    "k${interval}" \
    "${source_root}-delta-v3-compact-k${interval}-v1"
done

for load in "${loads[@]}"; do
  stress_root="${data_parent}/grouped-v2-v1-10-eval-stress-test24-update${load}-front-middle-v1"
  if [[ -e "${stress_root}" && ! -f "${stress_root}/COMPLETED" ]]; then
    echo "Incomplete stress dataset already exists: ${stress_root}" >&2
    exit 2
  fi
  if [[ ! -f "${stress_root}/COMPLETED" ]]; then
    echo "[$(date -Is)] prepare Test24 U${load}"
    "${python_bin}" -m memory_training.prepare_update_stress_suite \
      --source "${source_root}" \
      --output "${stress_root}" \
      --scenarios "${scenarios[@]}" \
      --target-updates "${load}" \
      --compaction-interval 5 \
      --allow-mixed-memory-splits
  else
    echo "[$(date -Is)] reuse Test24 U${load}"
  fi

  missing_intervals=()
  for interval in "${intervals[@]}"; do
    compact_root="${stress_root}-delta-v3-compact-k${interval}-v1"
    if [[ -e "${compact_root}" && ! -f "${compact_root}/COMPLETED" ]]; then
      echo "Incomplete compact dataset already exists: ${compact_root}" >&2
      exit 2
    fi
    if [[ ! -f "${compact_root}/COMPLETED" ]]; then
      missing_intervals+=("${interval}")
    fi
  done
  if (( ${#missing_intervals[@]} > 0 )); then
    echo "[$(date -Is)] prepare Test24 U${load} compact intervals: ${missing_intervals[*]}"
    "${python_bin}" -m memory_training.prepare_delta_v3_compaction_ablation \
      --source "${stress_root}" \
      --intervals "${missing_intervals[@]}"
  fi
done

"${python_bin}" -m memory_training.scripts.audit_full_test24_update_stress \
  --manifest-root "${manifest_root}"

echo "[$(date -Is)] Test24 Base/U40/U60/U80 data preparation complete"
