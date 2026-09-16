# Main Test Performance: 6 Models × 5 Methods

- 확인 시각: 2026-09-10 UTC
- Test: S86–S100, S112–S120 (24 scenarios, 2,364 turns, 960 quizzes)
- 생성: fixed seed, `do_sample=false`
- Delta-v3: compact `k=5`
- 단위: `%`; Composite·Quiz ESM·F1은 높을수록, False-update·Invalid는 낮을수록 좋다.

```text
Composite = 0.60 × Quiz ESM
          + 0.25 × Final-memory F1
          + 0.15 × Update F1
```

## 완료 감사

6개 모델의 Mem0-style One-pass와 Two-stage 총 12개 run을 원본 artifact에서 확인했다.

- 학습: 12/12 `COMPLETED`
- best-checkpoint Validation: 12/12 완료 (6 scenarios, 240 quizzes)
- fixed Test: 12/12 `complete=true`, 각각 24/24 scenarios
- 각 Test workload: 2,364 turns, 960 quizzes

## 통합 성능표

| Model | Method | Seed | Tested epoch | Composite ↑ | Quiz ESM ↑ | Final-memory F1 ↑ | Update F1 ↑ | False-update ↓ | Invalid ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Granite 350M | Patch | 45 | 4 | **59.62** | 60.73 | 43.92 | 81.38 | 1.83 | 6 |
|  | Summary | 45 | 4 | 55.28 | 55.62 | 42.45 | 75.26 | 1.37 | 10 |
|  | Delta-v3 (`k=5`) | 45 | 3 | 59.01 | 60.21 | 44.92 | 77.68 | 1.42 | 6 |
|  | Mem0 One-pass | 45 | 3 | 58.56 | 67.40 | 25.78 | 77.84 | 2.94 | 8 |
|  | Mem0 Two-stage | 45 | 3 | 55.96 | 64.17 | 23.73 | 76.86 | 1.47 | 2 |
| Qwen 0.8B | Patch | 46 | 3 | **70.39** | 74.17 | 52.93 | 84.40 | 1.62 | 1 |
|  | Summary | 45 | 3 | 64.38 | 67.29 | 46.51 | 82.49 | 1.12 | 3 |
|  | Delta-v3 (`k=5`) | 45 | 3 | 66.68 | 69.69 | 50.78 | 81.15 | 0.96 | 2 |
|  | Mem0 One-pass | 45 | 3 | 65.50 | 77.29 | 25.50 | 85.01 | 2.39 | 1 |
|  | Mem0 Two-stage | 45 | 3 | 59.87 | 68.65 | 26.71 | 80.00 | 1.17 | 3 |
| Granite 1B | Patch | 45 | 3 | **71.74** | 77.92 | 50.23 | 82.86 | 2.89 | 6 |
|  | Summary | 45 | 3 | 69.90 | 76.25 | 46.69 | 83.16 | 2.54 | 3 |
|  | Delta-v3 (`k=5`) | 45 | 4 | 67.38 | 70.73 | 49.22 | 84.22 | 1.42 | 3 |
|  | Mem0 One-pass | 45 | 2 | 54.31 | 61.98 | 25.35 | 71.90 | 0.71 | 9 |
|  | Mem0 Two-stage | 45 | 3 | 57.06 | 65.10 | 24.94 | 78.41 | 1.02 | 17 |
| Llama 3.2 1B | Patch | 45 | 3 | **67.06** | 71.98 | 46.49 | 81.64 | 2.74 | 8 |
|  | Summary | 45 | 4 | 60.30 | 60.83 | 47.74 | 79.11 | 0.71 | 6 |
|  | Delta-v3 (`k=5`) | 45 | 4 | 65.91 | 67.29 | 51.75 | 83.96 | 1.12 | 7 |
|  | Mem0 One-pass | 45 | 3 | 62.74 | 73.33 | 25.12 | 83.06 | 3.30 | 3 |
|  | Mem0 Two-stage | 45 | 3 | 58.60 | 67.19 | 25.24 | 79.83 | 1.47 | 13 |
| Qwen 2B | Patch | 46 | 3 | **71.13** | 75.83 | 52.28 | 83.75 | 2.34 | 0 |
|  | Summary | 45 | 4 | 65.61 | 69.69 | 47.51 | 79.47 | 0.71 | 4 |
|  | Delta-v3 (`k=5`) | 45 | 4 | 69.10 | 71.67 | 54.00 | 84.01 | 1.17 | 1 |
|  | Mem0 One-pass | 45 | 3 | 61.89 | 71.98 | 26.33 | 80.77 | 1.88 | 3 |
|  | Mem0 Two-stage | 45 | 2 | 64.63 | 76.35 | 24.61 | 84.44 | 1.68 | 0 |
| Llama 3.2 3B | Patch | 45 | 4 | **71.06** | 74.27 | 54.97 | 85.05 | 1.27 | 2 |
|  | Summary | 45 | 3 | 66.55 | 69.79 | 49.59 | 81.84 | 0.81 | 0 |
|  | Delta-v3 (`k=5`) | 45 | 4 | 69.22 | 72.71 | 53.31 | 81.78 | 0.91 | 1 |
|  | Mem0 One-pass | 45 | 3 | 64.00 | 74.90 | 26.23 | 83.38 | 2.08 | 1 |
|  | Mem0 Two-stage | 45 | 3 | 59.89 | 69.38 | 25.13 | 79.88 | 0.61 | 0 |

## 핵심 관찰

- Patch가 6/6 모델에서 Composite 1위다.
- Mem0 One-pass가 Two-stage보다 4/6 모델에서 높고, Two-stage는 Granite 1B와 Qwen
  2B에서만 높다.
- Mem0 계열은 5/6 모델에서 가장 높은 Quiz ESM을 만들었지만, Final-memory F1이
  `23.73–26.71`로 다른 세 방법의 `42.45–54.97`보다 크게 낮아 Composite가 하락했다.
  이는 downstream utility와 gold-memory 표면 일치도가 서로 다르게 움직임을 뜻한다.
- Qwen 0.8B Mem0 One-pass는 Update F1 `85.01`과 Quiz ESM `77.29`로 강하지만 낮은
  Final-memory F1 때문에 Composite는 Patch보다 `4.89`%p 낮다.

## 비교 규칙

- 기존 3방법 수치는
  [6 Models × 3 Methods draft](main-performance-five-models-three-methods-draft.md)의
  선택 규칙을 유지했다.
- Qwen 0.8B와 Qwen 2B의 Patch만 기존 메인 표와 동일하게 seed 46이고, 나머지 행은
  seed 45다. 따라서 완전한 동일-seed 비교가 필요하면 두 Patch 행의 seed 45 결과를
  별도 보조표로 제시해야 한다.
- 기존 세 방법은 Validation Composite로 epoch를 선택했다. 현재 Mem0 두 방법은 학습 중
  teacher-forced validation loss로 선택된 checkpoint 하나만 closed-loop Validation·Test한
  결과다. 따라서 이 표는 현재 완료 결과 비교이며, 논문 최종 공정 비교에는 Mem0의 4개
  epoch를 모두 동일한 Validation Composite로 선택하는 절차가 추가로 필요하다.
- `Final-memory F1`은 evaluator JSON의 `memory.final_state_f1`과 동일한 지표다.
