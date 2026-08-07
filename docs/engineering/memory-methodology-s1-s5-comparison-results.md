# Recursive Summary vs 최신 Ours Scenario 1–5 결과

상태: `50-task 단일 run 비교 완료 — selector·binding 범용성 확인, 구현은 보류`

## 목적과 조건

Scenario 6–10의 7개 안정 실패에서 찾은 문제가 특정 구간에만 맞춘 것인지 확인하기
위해, 이전에 두 최신 방법을 직접 비교하지 않았던 Scenario 1–5를 평가했다.

- Scenario 1–5, 각 10개 Quiz, 총 50 task
- Quiz model: `gpt-5.6-terra`
- Memory model: `gpt-5.6-luna`
- Embedding: `text-embedding-3-small`, 256 dimensions
- Recursive Summary cache를 먼저 생성하고 최신 방법이 동일 cache를 재사용
- selector·binding 코드는 변경하지 않은 상태

## 결과

| 방법 | ESM | State F1 | Tool F1 | Argument exact | Recall@k |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Recursive Summary** | **0.58** | **0.807** | **0.664** | **0.54** | N/A |
| Post-normalized Recursive-assisted Ours | 0.52 | 0.706 | 0.562 | 0.48 | 0.550 |
| Ours 차이 | -0.06 | -0.101 | -0.102 | -0.06 | - |

이번 단일 run에서는 최신 Ours가 Recursive Summary보다 모든 E2E 정확도 지표에서
낮았다. 고정 cache Agent 반복 전이므로 `0.06` 차이를 확정적인 방법론 차이로
해석하지는 않지만, 최신 Ours가 Scenario 1–5에서 우위라는 근거는 없다.

## Scenario별 ESM

| Scenario | Recursive Summary | 최신 Ours |
| ---: | ---: | ---: |
| 1 | **0.80** | 0.70 |
| 2 | **0.70** | 0.50 |
| 3 | **0.50** | 0.40 |
| 4 | 0.70 | 0.70 |
| 5 | 0.20 | **0.30** |

두 방법이 모두 성공한 task는 20개, Recursive Summary만 성공한 task는 9개,
최신 Ours만 성공한 task는 6개, 둘 다 실패한 task는 15개다. 성공 task 15개가
방법 사이에서 교체돼, ESM 합계만으로는 동작 차이를 설명할 수 없다.

## 최신 Ours 병목

최신 Ours의 24개 ESM 실패를 Recall 기준으로 나누면 다음과 같다.

| 실패 상태 | Task | 의미 |
| --- | ---: | --- |
| Recall 0 | 9 | 필요한 값이 선택 Fact에 없음; 저장·검색 앞단 대상 |
| Partial Recall | 8 | 값 또는 필수 runtime argument 일부만 있음 |
| Full Recall | 7 | 필요한 값이 있는데도 selector·binding·Agent 실행에서 실패 |

전체 50건 중 22건에서 reference Tool 하나 이상이 selector에 없었다. 실패 24건
중 14건이 이 selector 누락을 포함했고, 성공 26건 중에도 8건은 Agent discovery가
누락을 보완했다. 따라서 selector 문제는 Scenario 6–10의 특정 7건에만 국한되지
않지만, selector 누락 하나만으로 모든 실패가 설명되지는 않는다.

둘 다 실패한 15건에서 최신 Ours의 1차 진단은 selector miss 8건, memory argument
mapping rejection 6건, Tool 실행 오류 1건이었다. 그중 최신 Ours의 Recall이 full인
task도 4건이다. Recursive Summary도 같은 15건에서 Agent Tool 누락 11건과 argument
mismatch 4건으로 실패했다. 구조화 Fact만의 문제가 아니라 검색된 Memory를 실제
호출로 바꾸는 공통 실행 계층도 병목이다.

## 효율

| 항목 | Recursive Summary | 최신 Ours |
| --- | ---: | ---: |
| Memory generation calls | 239 | 1,105 |
| Memory generation input tokens | 603,666 | 3,716,272 |
| Agent input tokens | 205,846 | 175,323 |
| 평균 discovery calls | 1.10 | 0.26 |

최신 Ours는 Agent input을 약 14.8%, discovery를 약 76% 줄였지만, Memory 생성
input은 약 6.2배이고 이번 run의 정확도는 더 낮다. 최신 방식의 1,105회에는
재사용한 Recursive Summary 239회와 Fact extraction·validation 866회가 포함된다.

## 판정

1. Scenario 1–5에서도 Fact 누락과 selector·binding 실패가 함께 나타나므로 문제
   자체는 6–10에 국소적이지 않다.
2. 그러나 `1`, `right`, `evening/night` 세 사례를 바로 규칙으로 고치는 것은 여전히
   범위가 좁다.
3. 다음 단계는 Scenario 1–10 전체 100 task의 selector miss, compound call,
   Fact→Tool mapping, required argument, condition, Fact absence를 동일 taxonomy로
   offline 분류하는 것이다.
4. 그 분류에서 여러 Scenario를 함께 덮는 구조적 수정만 구현하고, task별 literal과
   Gold-derived default는 금지한다.

Scenario 6–10과 합친 100-task 구조 분석은
[Scenario 1–10 구조적 병목 비교](memory-methodology-s1-s10-structural-bottleneck-comparison.md)에
기록했다.

## Artifact

- Recursive Summary: `ubuntu/evaluation/vehiclemembench/cf3067dc-ebca-4d50-92a7-bea257cc1950`
- 최신 Ours: `ubuntu/evaluation/vehiclemembench/a9215a40-ff50-4722-9173-f33d1ad02f4c`
