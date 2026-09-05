# On-device Memory 최신 Validation·Test 결과

> **결과 기준 문서(Source of Truth).** 모델/방법론별 최신 성능을 조회하거나 갱신할 때는
> 개별 run 디렉터리를 다시 탐색하기 전에 이 문서를 먼저 확인한다. 새 평가가 완료되면 이
> 문서의 표, 원본 artifact 목록, 갱신 시각을 함께 업데이트한다.

- 마지막 갱신: 2026-09-05 UTC
- 평가 데이터:
  `grouped-v2-v1-10-eval-fixed-noop5-seed45-v1`
- Validation: S81–S85, S111 (6 scenarios)
- Test: S86–S100, S112–S120 (24 scenarios)
- 생성 설정: fixed seed, `do_sample=false`
- 기본 비교 seed: training seed 45
- 350M·1B 모델 계열: Granite 4
- 0.8B·2B 모델 계열: Qwen3.5

## Composite 정의

Validation과 Test 모두 보고 시 아래 동일한 복합점수를 사용한다.

```text
Composite = 0.60 × Quiz ESM
          + 0.25 × Final-state F1
          + 0.15 × Update F1
```

- **Quiz ESM:** 예측 memory로 실행한 tool call의 전체 상태가 정답과 정확히 일치한 비율
- **Final-state F1:** trajectory 종료 시 예측 memory와 정답 memory 사이의 F1
- **Update F1:** UPDATE가 필요한 turn을 탐지하고 올바르게 갱신한 성능의 F1
- Validation Composite는 best checkpoint 선택에 사용한다.
- Test Composite는 최종 성능 비교·보고용이며 checkpoint 선택에는 사용하지 않는다.
- Composite 동점 시 Quiz ESM → Final-state F1 → Update F1 → 이른 epoch 순으로 선택한다.

## 최신 결과

모든 성능 값은 `%`이다. 기본 표는 training seed 45 결과이며, 2B Patch seed46 반복 실험은
재현성 확인을 위해 별도 행으로 표시한다.

| 크기 | 방법론 | Train seed | Best epoch | Val Composite | Val ESM | Val State F1 | Val Update F1 | Test Composite | Test ESM | Test State F1 | Test Update F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 350M | Patch | 45 | 4 | 53.15 | 54.58 | 37.69 | 73.17 | 59.63 | 60.73 | 43.92 | 81.38 |
| 350M | Delta-v3 | 45 | 4 | 49.07 | 49.17 | 38.29 | 66.67 | 57.26 | 58.13 | 42.12 | 79.03 |
| 350M | Summary | 45 | 4 | 51.55 | 53.33 | 36.62 | 69.28 | 55.27 | 55.62 | 42.45 | 75.26 |
| 0.8B | Patch | 45 | 3 | 60.65 | 66.25 | 41.09 | 70.86 | 70.84 | 76.15 | 50.21 | 84.02 |
| 0.8B | Delta-v3 | 45 | 3 | 59.46 | 62.92 | 44.93 | 69.86 | 63.86 | 66.98 | 47.43 | 78.77 |
| 0.8B | Summary | 45 | 3 | 57.45 | 60.83 | 39.48 | 73.89 | 64.38 | 67.29 | 46.51 | 82.49 |
| 1B | Patch | 45 | 3 | 60.68 | 62.92 | 45.96 | 76.24 | 71.74 | 77.92 | 50.23 | 82.86 |
| 1B | Delta-v3 | 45 | 3 | 56.23 | 59.58 | 38.21 | 72.84 | 68.43 | 73.65 | 47.91 | 81.79 |
| 1B | Summary | 45 | 3 | 55.59 | 57.08 | 43.20 | 70.24 | 69.90 | 76.25 | 46.69 | 83.16 |
| 2B | Patch | 45 | 3 | 62.03 | 66.25 | 42.84 | 77.11 | 69.58 | 74.48 | 50.64 | 81.51 |
| 2B | Delta-v3 | 45 | 3 | 59.58 | 61.25 | 48.27 | 71.72 | 67.29 | 71.98 | 49.18 | 78.71 |
| 2B | Summary | 45 | 4 | 58.98 | 62.08 | 43.43 | 72.48 | 65.61 | 69.69 | 47.51 | 79.47 |
| 2B | Patch (repeat) | 46 | 3 | 63.07 | 68.33 | 43.90 | 73.99 | 71.13 | 75.83 | 52.28 | 83.75 |

## Delta-v3 compact k ablation (`k=2,5,10`)

Delta-v3 compact memory의 UPDATE compaction interval `k` 영향을 비교한다. 상세한 실험
설계와 데이터 audit은 [Delta-v3 compact k ablation](delta-v3-compact-k-ablation.md)에
기록한다. 이 표는 위 최신 결과 표의 일반 Delta-v3와 별도 실험군이며, Granite 4
350M·1B와 Qwen3.5 0.8B 결과를 포함한다.

- 공통 학습 설정: 4 epochs, uniform-depth sampling
- Granite 4 350M: batch size 8, gradient accumulation 2
- Qwen3.5 0.8B: batch size 4, gradient accumulation 4
- Granite 4 1B: batch size 2, gradient accumulation 8
- Validation: S81–S85, S111 전체와 closed-loop quiz 240개
- Test: S86–S100, S112–S120 전체와 closed-loop quiz 960개
- 모든 Test artifact는 `complete=true`, 24/24 scenarios이다.
- Seed 45와 46은 실제 Validation·Test sample ID가 같고 `do_sample=false`이므로 직접 비교한다.

모든 성능 값은 `%`이다.

| 모델 | k | Train seed | Best epoch | Val Composite | Val ESM | Val State F1 | Val Update F1 | Test Composite | Test ESM | Test State F1 | Test Update F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Granite 4 350M | 2 | 45 | 3 | **53.59** | **54.17** | **41.79** | 70.93 | **63.22** | **66.35** | 44.88 | **81.23** |
| Granite 4 350M | 5 | 45 | 3 | 51.39 | 51.67 | 38.96 | 70.97 | 59.01 | 60.21 | **44.92** | 77.68 |
| Granite 4 350M | 10 | 45 | 4 | 50.19 | 50.00 | 37.05 | **72.84** | 58.12 | 58.65 | 43.15 | 80.94 |
| Qwen3.5 0.8B | 2 | 45 | 3 | 62.03 | 65.83 | 45.62 | 74.16 | 70.78 | 75.83 | 49.86 | 85.41 |
| Qwen3.5 0.8B | 2 | 46 | 3 | 56.42 | 58.75 | 43.86 | 68.03 | 63.86 | 65.31 | 51.40 | 78.81 |
| Qwen3.5 0.8B | 5 | 45 | 3 | 58.87 | 61.67 | 46.68 | 68.00 | 66.68 | 69.69 | 50.78 | 81.15 |
| Qwen3.5 0.8B | 5 | 46 | 4 | 62.45 | 64.58 | 48.59 | 77.02 | 66.71 | 69.69 | 48.84 | 84.62 |
| Qwen3.5 0.8B | 10 | 45 | 3 | 59.70 | 60.00 | 49.43 | 75.64 | 67.55 | 71.56 | 47.83 | 84.38 |
| Qwen3.5 0.8B | 10 | 46 | 4 | 60.60 | 61.25 | 50.87 | 74.21 | 67.80 | 71.35 | 50.18 | 82.92 |
| Granite 4 1B | 2 | 45 | 3 | 57.26 | 60.83 | 40.84 | 70.30 | **69.73** | **74.90** | 48.55 | **84.37** |
| Granite 4 1B | 5 | 45 | 4 | 54.77 | 56.25 | 42.30 | 69.62 | 67.38 | 70.73 | **49.22** | 84.22 |
| Granite 4 1B | 10 | 45 | 4 | **58.88** | **62.92** | **42.72** | 69.68 | 64.33 | 66.35 | 48.40 | 82.79 |

### Seed 46 - Seed 45 차이

| 모델 | k | Val Composite Δ | Test Composite Δ | Test ESM Δ | Test State F1 Δ | Test Update F1 Δ |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5 0.8B | 2 | -5.61 | -6.92 | -10.52 | +1.54 | -6.60 |
| Qwen3.5 0.8B | 5 | +3.58 | +0.03 | 0.00 | -1.95 | +3.46 |
| Qwen3.5 0.8B | 10 | +0.90 | +0.24 | -0.21 | +2.35 | -1.46 |

- `k=2`는 Seed 46에서 Test Composite가 `6.92`%p 하락해 seed 민감도가 크다.
- `k=5`와 `k=10`의 Test Composite seed 차이는 각각 `0.03`%p, `0.24`%p로 작다.
- Seed 45의 최고 Test Composite는 `k=2`의 `70.78`, Seed 46은 `k=10`의 `67.80`이다.
- Granite 4 1B에서는 `k=2`가 Test Composite `69.73`으로 Compact 설정 중 가장 높다.
- Granite 4 350M에서도 `k=2`가 Test Composite `63.22`로 가장 높고, 기존 Delta-v3보다
  `5.96`%p 높다.
- Granite 4 1B의 Validation 최고는 `k=10`이지만 Test 최고는 `k=2`여서 작은 Validation
  split의 순위 변동 가능성이 다시 관찰된다.

### Granite 4 1B 기존 Delta-v3 대비

동일한 training seed 45의 기존 Delta-v3를 baseline으로 사용한다.

| 방법 | Best epoch | Val Composite | 기존 대비 Δ | Test Composite | 기존 대비 Δ | Test ESM | Test State F1 | Test Update F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 기존 Delta-v3 | 3 | 56.23 | — | 68.43 | — | 73.65 | 47.91 | 81.79 |
| Compact K=2 | 3 | 57.26 | +1.03 | **69.73** | **+1.30** | **74.90** | 48.55 | **84.37** |
| Compact K=5 | 4 | 54.77 | -1.46 | 67.38 | -1.06 | 70.73 | **49.22** | 84.22 |
| Compact K=10 | 4 | **58.88** | **+2.65** | 64.33 | -4.10 | 66.35 | 48.40 | 82.79 |

- 기존 Delta-v3 대비 Test Composite가 개선된 설정은 K=2 하나이며 `+1.30`%p다.
- K=5는 State/Update F1이 높지만 ESM 하락으로 Composite가 `-1.06`%p 낮다.
- K=10은 Validation Composite는 가장 높지만 Test Composite는 baseline보다 `4.10`%p
  낮아 Validation 순위와 Test 일반화가 일치하지 않는다.

### Granite 4 350M 기존 Delta-v3 대비

동일한 training seed 45의 기존 Delta-v3를 baseline으로 사용한다.

| 방법 | Best epoch | Val Composite | 기존 대비 Δ | Test Composite | 기존 대비 Δ | Test ESM | Test State F1 | Test Update F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 기존 Delta-v3 | 4 | 49.07 | — | 57.26 | — | 58.13 | 42.12 | 79.03 |
| Compact K=2 | 3 | **53.59** | **+4.52** | **63.22** | **+5.96** | **66.35** | 44.88 | **81.23** |
| Compact K=5 | 3 | 51.39 | +2.31 | 59.01 | +1.75 | 60.21 | **44.92** | 77.68 |
| Compact K=10 | 4 | 50.19 | +1.12 | 58.12 | +0.86 | 58.65 | 43.15 | 80.94 |

- 세 Compact 설정 모두 기존 Delta-v3보다 Test Composite가 높다.
- K=2는 ESM과 Update F1이 함께 상승해 Test Composite 개선 폭이 가장 크다.
- K=5는 State F1이 가장 높지만 Update F1이 baseline보다 낮고, K=10의 이득은 작다.

### Ablation 원본 artifact

아래 경로는 모두
`/mnt/data/hj153lee/PalmClaw/on-device-memory-training/runs/` 기준이다.

| 모델 | k | Train seed | Run 디렉터리 | Validation artifact | Test artifact |
|---|---:|---:|---|---|---|
| Granite 4 350M | 2 | 45 | `granite4-350m-delta_v3_compact_k2-multitask-noop5-uniform-depth-e4-b8-trainseed45-r1` | `eval-fixed-validation-epoch-03.json` | `eval-fixed-test-best-epoch-03/summary.json` |
| Granite 4 350M | 5 | 45 | `granite4-350m-delta_v3_compact_k5-multitask-noop5-uniform-depth-e4-b8-trainseed45-r1` | `eval-fixed-validation-epoch-03.json` | `eval-fixed-test-best-epoch-03/summary.json` |
| Granite 4 350M | 10 | 45 | `granite4-350m-delta_v3_compact_k10-multitask-noop5-uniform-depth-e4-b8-trainseed45-r1` | `eval-fixed-validation-epoch-04.json` | `eval-fixed-test-best-epoch-04/summary.json` |
| Qwen3.5 0.8B | 2 | 45 | `qwen35-0.8b-delta_v3_compact_k2-multitask-noop5-uniform-depth-e4-b4-trainseed45-evalfixed-noop5-r1` | `eval-fixed-validation-epoch-03.json` | `eval-fixed-test-best-epoch-03/summary.json` |
| Qwen3.5 0.8B | 2 | 46 | `qwen35-0.8b-delta_v3_compact_k2-multitask-noop5-uniform-depth-e4-b4-trainseed46-evalfixed-noop5-r1` | `eval-fixed-validation-epoch-03.json` | `eval-fixed-test-best-epoch-03/summary.json` |
| Qwen3.5 0.8B | 5 | 45 | `qwen35-0.8b-delta_v3_compact_k5-multitask-noop5-uniform-depth-e4-b4-trainseed45-evalfixed-noop5-r1` | `eval-fixed-validation-epoch-03.json` | `eval-fixed-test-best-epoch-03/summary.json` |
| Qwen3.5 0.8B | 5 | 46 | `qwen35-0.8b-delta_v3_compact_k5-multitask-noop5-uniform-depth-e4-b4-trainseed46-evalfixed-noop5-r1` | `eval-fixed-validation-epoch-04.json` | `eval-fixed-test-best-epoch-04/summary.json` |
| Qwen3.5 0.8B | 10 | 45 | `qwen35-0.8b-delta_v3_compact_k10-multitask-noop5-uniform-depth-e4-b4-trainseed45-evalfixed-noop5-r1` | `eval-fixed-validation-epoch-03.json` | `eval-fixed-test-best-epoch-03/summary.json` |
| Qwen3.5 0.8B | 10 | 46 | `qwen35-0.8b-delta_v3_compact_k10-multitask-noop5-uniform-depth-e4-b4-trainseed46-evalfixed-noop5-r1` | `eval-fixed-validation-epoch-04.json` | `eval-fixed-test-best-epoch-04/summary.json` |
| Granite 4 1B | 2 | 45 | `granite4-1b-delta_v3_compact_k2-multitask-noop5-uniform-depth-e4-b2-trainseed45-r1` | `eval-fixed-validation-epoch-03.json` | `eval-fixed-test-best-epoch-03/summary.json` |
| Granite 4 1B | 5 | 45 | `granite4-1b-delta_v3_compact_k5-multitask-noop5-uniform-depth-e4-b2-trainseed45-r1` | `eval-fixed-validation-epoch-04.json` | `eval-fixed-test-best-epoch-04/summary.json` |
| Granite 4 1B | 10 | 45 | `granite4-1b-delta_v3_compact_k10-multitask-noop5-uniform-depth-e4-b2-trainseed45-r1` | `eval-fixed-validation-epoch-04.json` | `eval-fixed-test-best-epoch-04/summary.json` |

## 원본 artifact

아래 경로는 모두
`/mnt/data/hj153lee/PalmClaw/on-device-memory-training/runs/` 기준이다.

| 모델·방법 | Run 디렉터리 | Validation artifact | Test artifact |
|---|---|---|---|
| 350M Patch | `granite4-350m-patch-multitask-noop5-full6val-grouped-v2-v1-10-e4-b8-trainseed45-r1` | `eval-fixed-v2-validation-best-epoch-04.json` | `eval-fixed-v2-test-best-epoch-04/summary.json` |
| 350M Delta-v3 | `granite4-350m-delta-v3-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed45-r1` | `eval-fixed-v2-validation-best-epoch-04.json` | `eval-fixed-v2-test-best-epoch-04/summary.json` |
| 350M Summary | `granite4-350m-summary-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed45-r1` | `eval-fixed-v2-validation-best-epoch-04.json` | `eval-fixed-v2-test-best-epoch-04/summary.json` |
| 0.8B Patch | `qwen35-0.8b-patch-multitask-noop5-grouped-v2-v1-10-e4-b8-trainseed45-evalfixed-noop5-r1` | `eval-fixed-validation-epoch-03.json` | `eval-fixed-test-best-epoch-03/summary.json` |
| 0.8B Delta-v3 | `qwen35-0.8b-delta-v3-multitask-noop5-grouped-v2-v1-10-e4-b4-trainseed45-evalfixed-noop5-r2` | `eval-fixed-validation-epoch-03.json` | `eval-fixed-test-best-epoch-03/summary.json` |
| 0.8B Summary | `qwen35-0.8b-summary-multitask-noop5-grouped-v2-v1-10-e4-b4-trainseed45-evalfixed-noop5-r2` | `eval-fixed-validation-epoch-03.json` | `eval-fixed-test-best-epoch-03/summary.json` |
| 1B Patch | `granite4-1b-patch-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed45-r1` | `eval-fixed-v2-validation-best-epoch-03.json` | `eval-fixed-v2-test-best-epoch-03/summary.json` |
| 1B Delta-v3 | `granite4-1b-delta-v3-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed45-r1` | `eval-fixed-v2-validation-best-epoch-03.json` | `eval-fixed-v2-test-best-epoch-03/summary.json` |
| 1B Summary | `granite4-1b-summary-multitask-noop5-trainfirst-grouped-v2-v1-10-e4-b8-trainseed45-r1` | `eval-fixed-v2-validation-best-epoch-03.json` | `eval-fixed-v2-test-best-epoch-03/summary.json` |
| 2B Patch | `qwen35-2b-patch-multitask-noop5-grouped-v2-v1-10-e5-b4-r1` | `eval-fixed-v2-validation-epoch-03.json` | `eval-fixed-v2-test-best-epoch-03/summary.json` |
| 2B Delta-v3 | `qwen35-2b-delta-v3-multitask-noop5-grouped-v2-v1-10-e4-b2-trainseed45-evalfixed-noop5-r1` | `eval-fixed-v2-validation-epoch-03.json` | `eval-fixed-v2-test-best-epoch-03/summary.json` |
| 2B Summary | `qwen35-2b-summary-multitask-noop5-grouped-v2-v1-10-e4-b2-r1` | `eval-fixed-v2-validation-epoch-04.json` | `eval-fixed-v2-test-best-epoch-04/summary.json` |
| 2B Patch seed46 | `qwen35-2b-patch-multitask-noop5-grouped-v2-v1-10-e5-b4-trainseed46-r2` | `eval-fixed-v2-validation-epoch-03.json` | `eval-fixed-v2-test-best-epoch-03/summary.json` |

## 업데이트 규칙

1. 새 결과를 반영하기 전에 평가 데이터, scenario split, decoding 설정이 위 기준과 같은지
   확인한다. 다른 protocol 결과는 같은 표에 혼합하지 않는다.
2. 모든 validation epoch에 Composite를 계산하고 가장 높은 epoch를 선택한다.
3. 선택된 checkpoint의 완성된 Test(`complete=true`, 24/24)만 표에 반영한다.
4. 원본 JSON 값으로 Composite를 계산하고, 표에는 마지막에만 소수점 둘째 자리로 반올림한다.
5. 기본 표는 seed45를 유지한다. 다른 seed 결과는 `repeat` 행으로 추가하며 기본 결과를
   조용히 대체하지 않는다.
6. 표를 수정할 때 원본 artifact 경로와 마지막 갱신 날짜도 함께 수정한다.
7. 결과 해석이나 보고서를 만들 때는 이 표를 먼저 사용하고, 누락되었거나 더 최신인 run이
   확인될 때만 원본 artifact를 다시 읽는다.

## 현재 요약

- seed45 기준 모든 크기에서 Patch가 Validation·Test Composite 1위다.
- 전체 Test Composite 최고는 1B Patch `71.74`다.
- 2B Patch seed46 반복 결과는 `71.13`으로 1B Patch와 `0.61`%p 차이다.
- 비-Patch 최고는 1B Summary `69.90`이다.
- 350M→0.8B 구간에서 성능 상승이 가장 크고, 그 이후에는 방법론과 seed 영향이 더 크다.
- Granite 4 1B Delta-v3 Compact 중 최고 Test Composite는 K=2의 `69.73`이다.
- Granite 4 350M Delta-v3 Compact 중 최고 Test Composite는 K=2의 `63.22`이며, 기존
  Delta-v3보다 `5.96`%p 높다.
- Delta-v3 compact k ablation에서 Qwen3.5 0.8B의 `k=5`, `k=10`은 seed 간 Test
  Composite 차이가 `0.24`%p 이하지만, `k=2`는 `6.92`%p 차이를 보였다.

## KV-cache stress canary

Validation S81-S85를 시나리오당 20/40/60/80 UPDATE로 확장한 cache-ON controlled
replay를 반복 1회 실행했다. 80 UPDATE의 UPDATE 직후 평균 evaluated prefill tokens는
Patch `1,754.5`, Delta k2 `964.0`, k5 `602.9`, k10 `659.9`였다. 이는 최종 통계가
아니다. 별도 predicted closed-loop 평가의 80 UPDATE Composite는 Patch `64.16`, k2
`62.29`, k5 `48.18`, k10 `57.77`이었다. 성능 보존형 trade-off는 k2, 더 공격적인
평균·p95 latency trade-off는 k10이 유력하다.

동일 규칙의 held-out Test S86-S90에서도 16/16 cache와 16/16 Composite job을 완료했다.
80 UPDATE의 UPDATE 직후 평균 evaluated prefill tokens는 Patch `1,737.6`, k2 `954.7`,
k5 `596.0`, k10 `654.2`였고, Composite는 각각 `63.20`, `66.94`, `48.52`, `58.13`이었다.
즉 k2는 이 Test cell에서 Composite를 `+3.74`%p 높이면서 prefill을 `45.1%` 줄였다.
실행 조건과 전체 결과는
[Delta-v3 Update Stress KV-cache Canary 결과](delta-v3-update-stress-canary-results.md)를
참조한다.

Granite 4 1B에서도 같은 held-out Stress Test를 16/16 완료했다. 80 UPDATE 평균 prefill
절감률은 k2 `27.6%`, k5 `60.6%`, k10 `72.6%`였지만 Composite는 Patch `68.78` 대비
k2 `55.46`, k5 `58.63`, k10 `39.62`였다. 따라서 token 이점은 모델군을 넘어
재현됐지만 accuracy를 보존하는 k는 모델 의존적이다. 전체 비교는
[Qwen 0.8B와 Granite 1B Stress Test 비교](delta-v3-stress-test-qwen-granite-comparison.md)를
참조한다.
