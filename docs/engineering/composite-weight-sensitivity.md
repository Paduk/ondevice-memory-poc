# Composite Weight Sensitivity and Scenario-Level Significance

## Analysis design

The reported Composite uses `Quiz ESM / Final-memory F1 / Update F1 = 0.60 / 0.25 / 0.15`. The primary sensitivity grid varies these weights in 0.025 increments while constraining Quiz ESM to 0.50--0.70, Final-memory F1 to 0.15--0.35, and Update F1 to 0.10--0.25. The weights always sum to one. A broader positive 0.05-step simplex is reported as a deliberately permissive diagnostic, not as the primary construct definition.

The primary grid contains **50** settings and the broad grid contains **171** settings. No model inference was rerun.

## Main-table ranking robustness

| Grid | Patch top | Patch model-level top | Winner counts |
|---|---:|---:|---|
| Primary | 100.0% | 95.3% | {"Patch": 50} |
| Broad | 97.7% | 75.3% | {"Patch": 167, "Delta-v3": 4} |

`Patch top` uses the six-model mean. `Patch model-level top` evaluates every model--weight pair separately.

Across the primary grid, Patch's mean margin over the strongest alternative ranges from **1.86 to 2.71 pp**. In the broad diagnostic, the only 4 non-Patch winners place at most 0.10 weight on Quiz ESM and at least 0.80 on Final-memory F1; these settings fall well outside the primary construct range.

## Compaction-interval Pareto robustness

| Workload | Primary grid: k=5 Pareto | Broad grid: k=5 Pareto | Default Pareto set |
|---|---:|---:|---|
| Base | 100.0% | 100.0% | k=2, k=5 |
| U40 | 100.0% | 92.4% | Patch, k=2, k=5, k=10 |
| U60 | 100.0% | 98.8% | Patch, k=2, k=5, k=10 |
| U80 | 100.0% | 99.4% | Patch, k=2, k=5, k=10 |

Across all model--workload--weight combinations, k=5 remains non-dominated in **83.9%** of primary-grid comparisons and **81.7%** of broad-grid comparisons.

### Model-level k=5 Pareto retention

| Model | Base | U40 | U60 | U80 |
|---|---:|---:|---:|---:|
| Granite 350M | 100.0% | 0.0% | 0.0% | 0.0% |
| Qwen 0.8B | 100.0% | 100.0% | 100.0% | 100.0% |
| Granite 1B | 100.0% | 100.0% | 100.0% | 100.0% |
| Llama 3.2 1B | 100.0% | 80.0% | 100.0% | 100.0% |
| Qwen 2B | 34.0% | 100.0% | 100.0% | 100.0% |
| Llama 3.2 3B | 100.0% | 100.0% | 100.0% | 100.0% |

The aggregate Pareto conclusion is fully stable over the primary grid, but model-level universality is not supported. The main exception is Granite 350M under U40--U80, where k=10 dominates k=5 throughout the primary grid. Thus the paper should describe k=5 as a robust cross-model operating point rather than the optimal setting for every individual model.

## Paired scenario-level results at the reported weights

Each comparison uses the same 24 scenarios for all methods. Mean differences are scenario-macro Composite differences averaged across six models. The 95% CI and paired randomization test operate on 24 scenario clusters after averaging within each scenario, treating the evaluated six models as the fixed model set. The machine-readable CSV additionally reports a two-axis hierarchical-bootstrap CI that treats both models and scenarios as varying. Holm adjustment is applied within the main-table and k-sweep families.

| Family | Workload | Contrast | Mean Δ (pp) | 95% CI | W/T/L | p | Holm p | Positive across primary weights |
|---|---|---|---:|---:|---:|---:|---:|---:|
| main | Base | Patch − Summary | +4.94 | [+3.53, +6.40] | 22/0/2 | <0.0001 | <0.0001 | 100.0% |
| main | Base | Patch − Delta-v3 | +2.47 | [+1.18, +3.82] | 17/0/7 | 0.0013 | 0.0013 | 100.0% |
| main | Base | Patch − Mem0 One-pass | +7.44 | [+4.48, +10.55] | 21/0/3 | <0.0001 | <0.0001 | 100.0% |
| k_sweep | Base | k=5 − Patch | -2.47 | [-3.80, -1.20] | 7/0/17 | 0.0014 | 0.0084 | 0.0% |
| k_sweep | Base | k=5 − k=2 | -2.68 | [-4.02, -1.26] | 6/0/18 | 0.0014 | 0.0084 | 0.0% |
| k_sweep | Base | k=5 − k=10 | +1.25 | [-0.10, +2.59] | 15/0/9 | 0.0867 | 0.1049 | 100.0% |
| k_sweep | U40 | k=5 − Patch | -7.02 | [-9.35, -4.94] | 2/0/22 | <0.0001 | 0.0001 | 0.0% |
| k_sweep | U40 | k=5 − k=2 | -5.02 | [-6.55, -3.48] | 3/0/21 | <0.0001 | 0.0001 | 0.0% |
| k_sweep | U40 | k=5 − k=10 | +1.83 | [+0.16, +3.59] | 16/0/8 | 0.0525 | 0.1049 | 100.0% |
| k_sweep | U60 | k=5 − Patch | -6.65 | [-8.69, -4.69] | 3/0/21 | <0.0001 | 0.0001 | 0.0% |
| k_sweep | U60 | k=5 − k=2 | -3.86 | [-5.48, -2.21] | 4/0/20 | 0.0003 | 0.0020 | 0.0% |
| k_sweep | U60 | k=5 − k=10 | +2.69 | [+1.03, +4.41] | 18/0/6 | 0.0052 | 0.0196 | 100.0% |
| k_sweep | U80 | k=5 − Patch | -6.42 | [-8.58, -4.43] | 1/0/23 | <0.0001 | 0.0001 | 0.0% |
| k_sweep | U80 | k=5 − k=2 | -4.79 | [-6.40, -3.17] | 4/0/20 | <0.0001 | 0.0002 | 0.0% |
| k_sweep | U80 | k=5 − k=10 | +3.79 | [+1.48, +6.11] | 17/0/7 | 0.0049 | 0.0196 | 100.0% |

## Interpretation boundary

Weight robustness and statistical significance answer different questions. Stable ranking over the predeclared grid shows that the qualitative conclusion does not depend on the single reported weighting. A non-significant paired test does not establish equivalence, and a Pareto point is not necessarily the highest-accuracy point. Accordingly, k=5 should be described as a balanced non-dominated operating point only in workloads and weight ranges for which the table supports that statement.

## Artifacts

- `docs/engineering/results/composite-weight-sensitivity.json`
- `docs/engineering/results/composite-weight-sensitivity-grid.csv`
- `docs/engineering/results/composite-weight-sensitivity-paired.csv`
- `docs/engineering/figures/composite-weight-sensitivity.pdf`
