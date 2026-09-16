#!/usr/bin/env bash
set -euo pipefail

root=/mnt/data/hj153lee/PalmClaw/on-device-memory-training/evaluations/patch-vs-delta-disagreement-analysis
snapshots=${root}/memory-snapshots
log=${root}/monitor-10min.log
models=(llama3.2-3b qwen3.5-2b granite4-1b llama3.2-1b granite4-350m qwen3.5-0.8b)
methods=(patch delta_v3_compact_k5)

while true; do
  completed=0
  {
    echo "CHECK $(date -u +%FT%TZ)"
    for model in "${models[@]}"; do
      line="${model}"
      for method in "${methods[@]}"; do
        dir=${snapshots}/${model}/${method}
        count=0
        if [[ -d ${dir}/scenarios ]]; then
          count=$(find "${dir}/scenarios" -maxdepth 1 -name 's*.json' -type f | wc -l)
        fi
        expected='?'
        if [[ -f ${dir}/manifest.json ]]; then
          expected=$(jq '.scenarios | length' "${dir}/manifest.json")
        fi
        done_flag=0
        if [[ -f ${dir}/summary.json ]] && jq -e '.complete == true' "${dir}/summary.json" >/dev/null; then
          done_flag=1
          completed=$((completed + 1))
        fi
        line+=" ${method}=${count}/${expected}(done=${done_flag})"
      done
      echo "${line}"
    done
    echo "COMPLETED ${completed}/12"
    if rg -i 'traceback|cuda out of memory|exception|killed' "${root}/logs" -g '*.log' >/dev/null 2>&1; then
      echo "ERROR_SCAN MATCH"
    else
      echo "ERROR_SCAN CLEAN"
    fi
  } >>"${log}"
  if [[ ${completed} -eq 12 ]]; then
    echo "FINISHED $(date -u +%FT%TZ)" >>"${log}"
    break
  fi
  sleep 600
done
