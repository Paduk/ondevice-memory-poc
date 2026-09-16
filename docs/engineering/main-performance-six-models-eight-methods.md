# Main Test Performance: 6 Models × 8 Methods

- 갱신: 2026-09-10 UTC
- Test: S86–S100, S112–S120 (24 scenarios, 960 quizzes)
- 상태 갱신 평가: fixed NO_OP 1:5 표본, 2,364 turns (UPDATE 394, NO_OP 1,970)
- 생성: fixed seed, `do_sample=false`
- 단위: `%`, 높을수록 좋다.

```text
Composite = 0.60 × Quiz ESM
          + 0.25 × Final-memory F1
          + 0.15 × Update F1
```

## 메인 테이블

| Model | Method | Tested checkpoint | Composite ↑ | Quiz ESM ↑ | Final-memory F1 ↑ | Update F1 ↑ |
|---|---|---:|---:|---:|---:|---:|
| Granite 350M | No Memory | Quiz-only E3 | N/A | 19.17 | N/A | N/A |
|  | Recent-window (768 tok.) | Quiz-only E3 | N/A | 51.67 | N/A | N/A |
|  | KV/LWW | Quiz-only E3 | 42.71 | 54.06 | 13.49 | 46.03 |
|  | Summary | MT E4 | 55.28 | 55.62 | 42.45 | 75.26 |
|  | Patch | MT E4 | **59.62** | 60.73 | 43.92 | 81.38 |
|  | Delta-v3 (`k=5`) | MT E3 | 59.01 | 60.21 | 44.92 | 77.68 |
|  | Mem0 One-pass | MT E3 | 58.56 | **67.40** | 25.78 | 77.84 |
|  | Mem0 Two-stage | MT E3 | 55.96 | 64.17 | 23.73 | 76.86 |
| Qwen 0.8B | No Memory | Quiz-only E3 | N/A | 19.79 | N/A | N/A |
|  | Recent-window (768 tok.) | Quiz-only E3 | N/A | 58.75 | N/A | N/A |
|  | KV/LWW | Quiz-only E3 | 42.65 | 53.96 | 13.49 | 46.03 |
|  | Summary | MT E3 | 64.38 | 67.29 | 46.51 | 82.49 |
|  | Patch | MT E3 | **70.39** | 74.17 | 52.93 | 84.40 |
|  | Delta-v3 (`k=5`) | MT E3 | 66.68 | 69.69 | 50.78 | 81.15 |
|  | Mem0 One-pass | MT E3 | 65.50 | **77.29** | 25.50 | 85.01 |
|  | Mem0 Two-stage | MT E3 | 59.87 | 68.65 | 26.71 | 80.00 |
| Granite 1B | No Memory | Quiz-only E3 | N/A | 26.35 | N/A | N/A |
|  | Recent-window (768 tok.) | Quiz-only E3 | N/A | 63.44 | N/A | N/A |
|  | KV/LWW | Quiz-only E3 | 45.84 | 59.27 | 13.49 | 46.03 |
|  | Summary | MT E3 | 69.90 | 76.25 | 46.69 | 83.16 |
|  | Patch | MT E3 | **71.74** | **77.92** | 50.23 | 82.86 |
|  | Delta-v3 (`k=5`) | MT E4 | 67.38 | 70.73 | 49.22 | 84.22 |
|  | Mem0 One-pass | MT E2 | 54.31 | 61.98 | 25.35 | 71.90 |
|  | Mem0 Two-stage | MT E3 | 57.06 | 65.10 | 24.94 | 78.41 |
| Llama 3.2 1B | No Memory | Quiz-only E3 | N/A | 20.83 | N/A | N/A |
|  | Recent-window (768 tok.) | Quiz-only E3 | N/A | 54.90 | N/A | N/A |
|  | KV/LWW | Quiz-only E3 | 41.21 | 51.56 | 13.49 | 46.03 |
|  | Summary | MT E4 | 60.30 | 60.83 | 47.74 | 79.11 |
|  | Patch | MT E3 | **67.06** | 71.98 | 46.49 | 81.64 |
|  | Delta-v3 (`k=5`) | MT E4 | 65.91 | 67.29 | 51.75 | 83.96 |
|  | Mem0 One-pass | MT E3 | 62.74 | **73.33** | 25.12 | 83.06 |
|  | Mem0 Two-stage | MT E3 | 58.60 | 67.19 | 25.24 | 79.83 |
| Qwen 2B | No Memory | Quiz-only E3 | N/A | 22.71 | N/A | N/A |
|  | Recent-window (768 tok.) | Quiz-only E3 | N/A | 63.44 | N/A | N/A |
|  | KV/LWW | Quiz-only E3 | 46.40 | 60.21 | 13.49 | 46.03 |
|  | Summary | MT E4 | 65.61 | 69.69 | 47.51 | 79.47 |
|  | Patch | MT E3 | **71.13** | 75.83 | 52.28 | 83.75 |
|  | Delta-v3 (`k=5`) | MT E4 | 69.10 | 71.67 | 54.00 | 84.01 |
|  | Mem0 One-pass | MT E3 | 61.89 | 71.98 | 26.33 | 80.77 |
|  | Mem0 Two-stage | MT E2 | 64.63 | **76.35** | 24.61 | 84.44 |
| Llama 3.2 3B | No Memory | Quiz-only E3 | N/A | 24.90 | N/A | N/A |
|  | Recent-window (768 tok.) | Quiz-only E3 | N/A | 62.40 | N/A | N/A |
|  | KV/LWW | Quiz-only E3 | 45.21 | 58.23 | 13.49 | 46.03 |
|  | Summary | MT E3 | 66.55 | 69.79 | 49.59 | 81.84 |
|  | Patch | MT E4 | **71.06** | 74.27 | 54.97 | 85.05 |
|  | Delta-v3 (`k=5`) | MT E4 | 69.22 | 72.71 | 53.31 | 81.78 |
|  | Mem0 One-pass | MT E3 | 64.00 | **74.90** | 26.23 | 83.38 |
|  | Mem0 Two-stage | MT E3 | 59.89 | 69.38 | 25.13 | 79.88 |

`MT`는 memory update와 quiz를 함께 학습한 multitask checkpoint다. 굵은 Composite는
각 모델의 상태형 방법 중 최고값이며, 굵은 Quiz ESM은 모든 방법 중 최고값이다.

## 방법별 6모델 평균

| Method | Composite ↑ | Quiz ESM ↑ | Final-memory F1 ↑ | Update F1 ↑ |
|---|---:|---:|---:|---:|
| No Memory | N/A | 22.29 | N/A | N/A |
| Recent-window (768 tok.) | N/A | 59.10 | N/A | N/A |
| KV/LWW | 44.01 | 56.22 | 13.49 | 46.03 |
| Summary | 63.67 | 66.58 | 46.75 | 80.22 |
| Patch | **68.50** | **72.48** | 50.14 | **83.18** |
| Delta-v3 (`k=5`) | 66.22 | 68.72 | **50.66** | 82.13 |
| Mem0 One-pass | 61.17 | 71.15 | 25.72 | 80.33 |
| Mem0 Two-stage | 59.34 | 68.47 | 25.06 | 79.90 |

## 해석 및 비교 규칙

- 공통 비교 지표는 960개 quiz의 `Quiz ESM`이다. No Memory와 Recent-window는 지속
  memory나 update decision이 없으므로 memory 지표와 Composite를 `N/A`로 둔다.
- Patch가 6/6 모델에서 상태형 Composite 1위이며, 6모델 평균도 `68.50`으로 가장 높다.
- Delta-v3는 평균 Final-memory F1이 `50.66`으로 가장 높고, Patch보다 Composite는
  `2.28`%p 낮다.
- Mem0 One-pass는 높은 Quiz ESM(`71.15`)에도 strict Final-memory F1(`25.72`)이 낮다.
- KV/LWW는 전체 67,423개 원본 turn으로 상태를 누적하되, Update F1은 다른 상태형 방법과
  같은 fixed 2,364-turn 표본에서 계산했다. Final-memory F1은 모델과 무관한 결정론적
  memory writer 결과이므로 여섯 reader에서 동일하다.
- No Memory·Recent-window·KV/LWW는 동일한 모델별 Quiz-only E3 reader를 사용한다.
  Summary·Patch·Delta-v3·Mem0는 각 방법의 multitask checkpoint를 사용하므로, 표는
  **end-to-end method system 비교**이지 memory writer만 교체한 reader-controlled
  ablation은 아니다.
- Patch의 Qwen 0.8B·2B 행만 seed 46, 나머지 학습형 행은 seed 45다. Mem0 epoch는 현재
  teacher-forced validation loss로 선택되어, 최종 논문에서는 모든 epoch를 동일한
  closed-loop Validation Composite로 선택하는 보완이 필요하다.

## 원본 결과

- Summary·Delta-v3·Patch 유형별 분석:
  [유형별 Test 정확도](three-method-quiz-type-accuracy-by-model.md)
- 기존 학습형 5방법: [6 Models × 5 Methods](main-performance-six-models-five-methods.md)
- No Memory·Recent-window·KV/LWW Quiz:
  `/mnt/data/hj153lee/PalmClaw/on-device-memory-training/evaluations/quiz-only-e3-baselines`
- KV/LWW memory·Composite:
  `/mnt/data/hj153lee/PalmClaw/on-device-memory-training/evaluations/quiz-only-e3-baselines/kv_lww-memory-composite.json`
