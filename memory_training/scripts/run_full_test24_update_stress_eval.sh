#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?Usage: $0 GPU MODEL METHOD CHECKPOINT [RUN_TAG]}"
model="${2:?Usage: $0 GPU MODEL METHOD CHECKPOINT [RUN_TAG]}"
method="${3:?Usage: $0 GPU MODEL METHOD CHECKPOINT [RUN_TAG]}"
checkpoint="${4:?Usage: $0 GPU MODEL METHOD CHECKPOINT [RUN_TAG]}"
run_tag="${5:-20260915-v1}"
requested_checkpoint="${checkpoint}"

case "${method}" in
  patch)
    compact_suffix=""
    manifest_label=base
    ;;
  delta_v3_compact_k2|delta_v3_compact_k5|delta_v3_compact_k10)
    interval="${method##*k}"
    compact_suffix="-delta-v3-compact-k${interval}-v1"
    manifest_label="k${interval}"
    ;;
  *)
    echo "METHOD must be patch or delta_v3_compact_k{2,5,10}: ${method}" >&2
    exit 2
    ;;
esac

repo_root=/home/hj153lee/PalmClaw
workspace=/mnt/data/hj153lee/PalmClaw/on-device-memory-training
data_parent=/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training
python_bin=/mnt/data/hj153lee/conda-envs/palmclaw-memory-sft/bin/python
source_name=grouped-v2-v1-10-eval-fixed-noop5-seed45-v1
manifest_root="${workspace}/benchmarks/natural-test24-manifests-20260915-v1"
model_label="${model//\//-}"
output_root="${workspace}/benchmarks/${model_label}-${method}-test24-update-stress-${run_tag}"
scenario_batch_size="${SCENARIO_BATCH_SIZE:-4}"
quiz_batch_size="${QUIZ_BATCH_SIZE:-16}"

# The selected Llama 3.2 1B k10 epoch-02 adapter is unavailable.  The
# 2026-09-15 campaign decision is to evaluate epoch-03 consistently for Base
# and all stress loads.  This compatibility branch also covers the already
# running queue process, whose in-memory function still passes epoch-02.
if [[ "${model}" == llama3.2-1b && \
      "${method}" == delta_v3_compact_k10 && \
      "${checkpoint}" == */epoch-02 ]]; then
  checkpoint="${checkpoint%/epoch-02}/epoch-03"
  export INCLUDE_BASE=1
  echo "[$(date -Is)] checkpoint override requested=${requested_checkpoint} effective=${checkpoint}; include_base=1"
fi

scenarios=(
  86 87 88 89 90 91 92 93 94 95 96 97 98 99 100
  112 113 114 115 116 117 118 119 120
)
categories=(base 40 60 80)
if [[ "${STRESS_ONLY:-0}" == 1 && "${INCLUDE_BASE:-0}" != 1 ]]; then
  categories=(40 60 80)
fi

if [[ ! -d "${checkpoint}/adapter" ]]; then
  echo "Checkpoint adapter not found: ${checkpoint}/adapter" >&2
  exit 2
fi

data_root_for() {
  local category=$1
  if [[ "${category}" == base ]]; then
    echo "${data_parent}/${source_name}${compact_suffix}"
  else
    echo "${data_parent}/grouped-v2-v1-10-eval-stress-test24-update${category}-front-middle-v1${compact_suffix}"
  fi
}

turn_manifest_for() {
  local category=$1
  local data_root=$2
  if [[ "${category}" == base ]]; then
    echo "${manifest_root}/${manifest_label}.json"
  else
    echo "${data_root}/turn_manifest.json"
  fi
}

for category in "${categories[@]}"; do
  data_root=$(data_root_for "${category}")
  turn_manifest=$(turn_manifest_for "${category}" "${data_root}")
  for required in catalog.sqlite quiz_sft.jsonl; do
    if [[ ! -f "${data_root}/${required}" ]]; then
      echo "Required data file is missing: ${data_root}/${required}" >&2
      exit 2
    fi
  done
  if [[ ! -s "${turn_manifest}" ]]; then
    echo "Turn manifest is missing: ${turn_manifest}" >&2
    exit 2
  fi
done

if [[ "${PREFLIGHT_ONLY:-0}" == 1 ]]; then
  echo "Preflight PASS: Test24 ${categories[*]} ${model} ${method}"
  exit 0
fi

mkdir -p "${output_root}/cache" "${output_root}/composite"
cd "${repo_root}"

export CUDA_VISIBLE_DEVICES="${gpu}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH="${repo_root}:${repo_root}/ubuntu/src${PYTHONPATH:+:${PYTHONPATH}}"

for category in "${categories[@]}"; do
  data_root=$(data_root_for "${category}")
  turn_manifest=$(turn_manifest_for "${category}" "${data_root}")
  label="${category}"
  if [[ "${category}" != base ]]; then
    label="update${category}"
  fi

  if [[ "${COMPOSITE_ONLY:-0}" != 1 ]]; then
    cache_output="${output_root}/cache/${label}/on"
    if [[ ! -s "${cache_output}/summary.json" ]]; then
      echo "[$(date -Is)] cache start GPU=${gpu} condition=${label} method=${method}"
      "${python_bin}" -m memory_training.benchmark_hf_prefix_cache \
        --model "${model}" \
        --method "${method}" \
        --checkpoint "${checkpoint}" \
        --turn-manifest "${turn_manifest}" \
        --output-dir "${cache_output}" \
        --workspace "${workspace}" \
        --data-root "${data_root}" \
        --catalog-path "${data_root}/catalog.sqlite" \
        --cache-mode on \
        --replay-mode controlled \
        --max-length 4096 \
        --max-new-tokens 768 \
        --warmup-turns 3 \
        --repetitions 1
    else
      echo "[$(date -Is)] cache reuse condition=${label} method=${method}"
    fi
  fi

  composite_output="${output_root}/composite/${label}"
  if [[ ! -s "${composite_output}/summary.json" ]] || \
     [[ "$(jq -r '.complete // false' "${composite_output}/summary.json")" != true ]]; then
    echo "[$(date -Is)] composite start GPU=${gpu} condition=${label} method=${method}"
    "${python_bin}" -m memory_training.evaluate_hf_closed_loop \
      --model "${model}" \
      --method "${method}" \
      --checkpoint "${checkpoint}" \
      --output-dir "${composite_output}" \
      --workspace "${workspace}" \
      --data-root "${data_root}" \
      --catalog-path "${data_root}/catalog.sqlite" \
      --scenarios "${scenarios[@]}" \
      --max-length 4096 \
      --max-new-tokens 768 \
      --quiz-max-new-tokens 256 \
      --scenario-batch-size "${scenario_batch_size}" \
      --quiz-batch-size "${quiz_batch_size}"
  else
    echo "[$(date -Is)] composite reuse condition=${label} method=${method}"
  fi
done

echo "[$(date -Is)] completed Test24 Base/U40/U60/U80: ${output_root}"
