# Appendix X. Compaction-Interval Results

Figure X summarizes the effect of the Delta-v3 compaction interval. This
appendix reports the corresponding aggregate and model-level results. Latency
is normalized within each model and workload to the matched Patch result.
Composite changes are also paired against the matched Patch result before
aggregation. Reported standard deviations therefore describe variation across
the six model configurations, rather than repeated-run uncertainty.

U40--U80 jointly increase update frequency and memory growth without
workload-specific adaptation. They are treated as stress operating points, not
as estimates of expected in-distribution performance.

## Aggregate results

| Workload | Method | Relative latency (%) | Latency reduction (%) | Composite (%) | Composite change (pp) |
|---|---|---|---|---|---|
| Base | Patch | 100.0 ± 0.0 | 0.0 ± 0.0 | 68.5 ± 4.7 | +0.0 ± 0.0 |
| Base | $k=2$ | 82.9 ± 8.5 | 17.1 ± 8.5 | 68.7 ± 2.8 | +0.2 ± 2.1 |
| Base | $k=5$ | 75.3 ± 4.6 | 24.7 ± 4.6 | 66.2 ± 3.8 | -2.3 ± 1.5 |
| Base | $k=10$ | 81.6 ± 10.3 | 18.4 ± 10.3 | 64.9 ± 4.5 | -3.6 ± 2.5 |
| U40 | Patch | 100.0 ± 0.0 | 0.0 ± 0.0 | 69.1 ± 5.4 | +0.0 ± 0.0 |
| U40 | $k=2$ | 74.8 ± 7.9 | 25.2 ± 7.9 | 67.1 ± 4.5 | -2.0 ± 3.2 |
| U40 | $k=5$ | 55.5 ± 3.1 | 44.5 ± 3.1 | 62.2 ± 6.8 | -7.0 ± 2.7 |
| U40 | $k=10$ | 54.8 ± 8.5 | 45.2 ± 8.5 | 60.3 ± 5.0 | -8.8 ± 3.9 |
| U60 | Patch | 100.0 ± 0.0 | 0.0 ± 0.0 | 68.3 ± 7.1 | +0.0 ± 0.0 |
| U60 | $k=2$ | 71.9 ± 8.0 | 28.1 ± 8.0 | 65.5 ± 5.0 | -2.8 ± 5.0 |
| U60 | $k=5$ | 47.3 ± 3.3 | 52.7 ± 3.3 | 61.6 ± 7.8 | -6.7 ± 3.4 |
| U60 | $k=10$ | 43.5 ± 6.2 | 56.5 ± 6.2 | 58.9 ± 6.4 | -9.4 ± 8.8 |
| U80 | Patch | 100.0 ± 0.0 | 0.0 ± 0.0 | 66.9 ± 7.5 | +0.0 ± 0.0 |
| U80 | $k=2$ | 69.7 ± 8.1 | 30.3 ± 8.1 | 65.2 ± 5.9 | -1.7 ± 3.8 |
| U80 | $k=5$ | 42.9 ± 3.4 | 57.1 ± 3.4 | 60.4 ± 9.3 | -6.5 ± 4.6 |
| U80 | $k=10$ | 37.3 ± 4.8 | 62.7 ± 4.8 | 56.6 ± 7.2 | -10.3 ± 7.4 |

Values are mean ± standard deviation across six models. Relative latency is
lower-is-better; latency reduction and Composite are higher-is-better.

## U80 model-level operating points

| Model | $k=2$ | $k=5$ | $k=10$ |
|---|---|---|---|
| Granite 350M | 24.2% / +1.9 pp | 55.2% / -11.5 pp | 63.1% / -8.0 pp |
| Qwen 0.8B | 39.4% / -0.6 pp | 61.9% / -11.0 pp | 55.7% / -5.3 pp |
| Llama 1B | 25.2% / -2.0 pp | 54.5% / -3.5 pp | 64.7% / -6.5 pp |
| Granite 1B | 25.7% / -9.1 pp | 54.8% / -9.1 pp | 67.9% / -24.5 pp |
| Qwen 2B | 42.0% / -0.5 pp | 60.9% / -3.5 pp | 58.4% / -12.4 pp |
| Llama 3B | 25.3% / +0.0 pp | 55.3% / -0.4 pp | 66.6% / -5.2 pp |

Each cell reports `latency reduction / Composite change` relative to the
matched Patch result. This table exposes the model dependence hidden by the
cross-model mean, particularly for the aggressive $k=10$ setting.

## Full model-level relative latency

| Model | Workload | Patch | $k=2$ | $k=5$ | $k=10$ |
|---|---|---|---|---|---|
| Granite 350M | Base | 100.0 | 95.9 | 82.7 | 83.0 |
| Granite 350M | U40 | 100.0 | 81.7 | 57.8 | 53.0 |
| Granite 350M | U60 | 100.0 | 78.4 | 48.5 | 42.6 |
| Granite 350M | U80 | 100.0 | 75.8 | 44.8 | 36.9 |
| Qwen 0.8B | Base | 100.0 | 74.0 | 68.3 | 92.6 |
| Qwen 0.8B | U40 | 100.0 | 66.2 | 50.8 | 66.3 |
| Qwen 0.8B | U60 | 100.0 | 63.1 | 42.5 | 52.4 |
| Qwen 0.8B | U80 | 100.0 | 60.6 | 38.1 | 44.3 |
| Llama 1B | Base | 100.0 | 83.4 | 74.5 | 74.9 |
| Llama 1B | U40 | 100.0 | 78.9 | 57.8 | 49.7 |
| Llama 1B | U60 | 100.0 | 76.8 | 50.2 | 40.7 |
| Llama 1B | U80 | 100.0 | 74.8 | 45.5 | 35.3 |
| Granite 1B | Base | 100.0 | 84.3 | 75.8 | 71.5 |
| Granite 1B | U40 | 100.0 | 79.3 | 57.6 | 47.0 |
| Granite 1B | U60 | 100.0 | 76.3 | 49.8 | 37.3 |
| Granite 1B | U80 | 100.0 | 74.3 | 45.2 | 32.1 |
| Qwen 2B | Base | 100.0 | 73.0 | 74.6 | 95.1 |
| Qwen 2B | U40 | 100.0 | 63.4 | 52.5 | 64.6 |
| Qwen 2B | U60 | 100.0 | 60.3 | 44.0 | 49.8 |
| Qwen 2B | U80 | 100.0 | 58.0 | 39.1 | 41.6 |
| Llama 3B | Base | 100.0 | 86.8 | 75.9 | 72.6 |
| Llama 3B | U40 | 100.0 | 79.5 | 56.8 | 48.0 |
| Llama 3B | U60 | 100.0 | 76.3 | 49.1 | 38.4 |
| Llama 3B | U80 | 100.0 | 74.7 | 44.7 | 33.4 |

Values are cache-ON update-cycle latency as a percentage of the matched Patch
latency; lower is better.

## Full model-level Composite

| Model | Workload | Patch | $k=2$ | $k=5$ | $k=10$ |
|---|---|---|---|---|---|
| Granite 350M | Base | 59.6 | 63.2 | 59.0 | 58.1 |
| Granite 350M | U40 | 60.9 | 61.6 | 51.8 | 54.0 |
| Granite 350M | U60 | 57.0 | 58.2 | 47.1 | 52.5 |
| Granite 350M | U80 | 54.6 | 56.6 | 43.2 | 46.7 |
| Qwen 0.8B | Base | 70.4 | 70.8 | 66.7 | 67.6 |
| Qwen 0.8B | U40 | 71.3 | 71.4 | 65.2 | 66.8 |
| Qwen 0.8B | U60 | 71.2 | 70.5 | 62.9 | 66.6 |
| Qwen 0.8B | U80 | 71.4 | 70.8 | 60.4 | 66.1 |
| Llama 1B | Base | 67.1 | 68.5 | 65.9 | 61.6 |
| Llama 1B | U40 | 64.4 | 62.0 | 55.6 | 55.2 |
| Llama 1B | U60 | 64.3 | 61.2 | 59.0 | 56.7 |
| Llama 1B | U80 | 61.7 | 59.7 | 58.1 | 55.2 |
| Granite 1B | Base | 71.7 | 69.7 | 67.4 | 64.3 |
| Granite 1B | U40 | 75.6 | 68.3 | 65.9 | 59.8 |
| Granite 1B | U60 | 78.0 | 65.4 | 68.2 | 51.8 |
| Granite 1B | U80 | 75.0 | 65.9 | 65.9 | 50.5 |
| Qwen 2B | Base | 71.1 | 69.3 | 69.1 | 70.6 |
| Qwen 2B | U40 | 71.5 | 72.2 | 66.1 | 62.1 |
| Qwen 2B | U60 | 71.3 | 70.6 | 65.6 | 60.1 |
| Qwen 2B | U80 | 71.5 | 71.0 | 67.9 | 59.0 |
| Llama 3B | Base | 71.1 | 70.6 | 69.2 | 67.2 |
| Llama 3B | U40 | 71.1 | 67.3 | 68.4 | 63.9 |
| Llama 3B | U60 | 68.0 | 67.0 | 67.1 | 65.8 |
| Llama 3B | U80 | 67.1 | 67.1 | 66.7 | 61.9 |

Values are Composite percentages on the same 24 scenarios.

## Data provenance for the current draft

The exact Figure inputs are provided in
`docs/engineering/results/section6-2-compaction-interval-six-model-raw.csv`.
The aggregate values are provided in
`docs/engineering/results/section6-2-compaction-interval-six-model-summary.csv`.
The current latency values are device-calibrated projections obtained by
applying Galaxy Z Fold7 prefill/decode throughput to controlled update-cycle
token traces. They should be relabeled if replaced by direct end-to-end device
measurements in the final manuscript.
