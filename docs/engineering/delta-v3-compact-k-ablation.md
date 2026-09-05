# Delta-v3 compact k ablation

## Fixed comparison

- Model: Granite 4 1B
- Method: the same compact `B/P/T` prompt and Delta-v3 operation target
- Variable: UPDATE compaction interval `k = 2, 5, 10`
- Training: identical scenarios, optimizer settings, seed, effective batch, and
  `NO_OP:UPDATE = 5:1`
- NO_OP sampling: uniform pending depth within each k, preventing a cache-age
  sampling bias between intervals

The prompt keeps the full base memory, ordered pending batches, turn ID,
timestamp, speaker fields, and text. Only redundant JSON field names and base
memory escaping are removed. The extra semantic rule is: apply `P` to `B` in
order before deciding the next operation.

## Prepare and validate data

```bash
python -m memory_training.prepare_delta_v3_compaction_ablation

python -m memory_training.validate_methods \
  --data-root <k-specific-data-root> \
  --methods delta_v3_compact_k2
```

Each k-specific root contains a regenerated `delta.jsonl`, matching
`compaction.jsonl`, catalog, manifest, preparation summary, and `COMPLETED`
marker. Summary/Patch/Quiz inputs are hard-linked because their content is
invariant.

## Train

```bash
memory_training/scripts/run_granite4_1b_delta_v3_compact_k_train.sh GPU K
```

Run once for K values 2, 5, and 10. Use distinct GPUs only when enough memory is
available; the script gives every run the same training settings.

## Acceptance checks

- Delta replay reaches the Patch trajectory hash on every turn.
- Maximum pending depths are 1, 4, and 9 respectively.
- All three datasets have identical row identities and final scenario hashes.
- Audit prompt/target token-length percentiles and any 4096-token truncation
  before starting the full runs.
- After training, compare quality separately from latency/cost. Latency reporting
  should include all turns, UPDATE-only turns, non-compaction UPDATEs, and
  compaction UPDATEs, with foreground and background prefill shown separately.

## Prepared-data audit (2026-09-03)

All three roots contain 362,774 aligned rows and passed full turn-by-turn replay.
Their final scenario hashes are identical. Compaction rows are 1,176 (k=2), 503
(k=5), and 298 (k=10).

Using the Granite 4 1B tokenizer over all 16,576 rows in validation scenarios
81–85 and 111:

| Prompt | Mean tokens | P95 | Maximum | Over 4096 |
| --- | ---: | ---: | ---: | ---: |
| Existing Delta-v3, k=5 | 633.35 | 942 | 1100 | 0 |
| Compact Delta-v3, k=2 | 519.38 | 753 | 813 | 0 |
| Compact Delta-v3, k=5 | 572.22 | 877 | 1036 | 0 |
| Compact Delta-v3, k=10 | 604.16 | 922 | 1153 | 0 |

The compact system instruction is 101 raw tokens versus 132 for the existing
Delta-v3 instruction. At the same k=5 state, the full mean prompt falls by 61.13
tokens (9.7%).
