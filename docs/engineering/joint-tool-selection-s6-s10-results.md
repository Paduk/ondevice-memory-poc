# Joint Tool Selection Scenario 6–10 결과

상태: `후보 reranker 동일-cache paired E2E 1회 완료, 반복 평가 전`

## 비교 조건

- 범위: VehicleMemBench Scenario 6–10, 50 task
- Agent: `gpt-5.6-terra`
- Memory: Post-normalized Schema-informed Recursive-assisted Fact
- 기준선 run: `6f72b35d-cda4-482e-a960-9671daf2933e`
- 공동 선택 run: `54f60f95-a637-4218-907c-16323ef6a83a`
- 각 scenario의 base Fact, assisted Fact, Recursive Summary component cache key가
  두 run에서 모두 동일함을 확인했다.

처음 비교 대상으로 삼았던 과거 Ours run은 Memory 생성 시점이 달라 Fact와
retrieval 품질이 일치하지 않았다. 따라서 현재 코드로 기준선을 다시 실행해
Memory 차이를 제거했다.

## 결과

| 지표 | 기존 Ours | 공동 Tool 선택 |
| --- | ---: | ---: |
| Exact State Match | 0.52 | 0.52 |
| State F1 | 0.6558 | 0.6478 |
| Tool F1 | 0.5513 | 0.5433 |
| Argument Exact | 0.48 | 0.48 |
| Tool selector miss | 16 | 14 |
| 평균 discovery call | 0.40 | 0.28 |
| 평균 선택 Tool 수 | 2.88 | 4.04 |
| 불필요한 vehicle call | 21 | 20 |
| Agent input tokens | 178,178 | 172,360 |

- 공동 selector가 실제로 달라진 task는 20개였다.
- 이 20개에서는 ESM 개선 2개, 악화 0개였다.
- selector가 같았던 task에서는 ESM 개선 1개, 악화 3개였다. 이 차이는 Tool 후보가
  동일하므로 Agent sampling 변동의 영향으로 해석해야 한다.
- Desired-state plan은 7개 task에서 적용됐지만 multi-Fact plan은 0개였다. 따라서
  이 실행은 개선 2의 검증 자료로 사용하지 않는다.

## 판단

개선 1은 필요한 Tool 접근성과 실행 효율에는 긍정적인 신호가 있다. selector miss와
discovery call이 줄었고, selector가 실제 변경된 task 집합에서는 ESM 악화 없이 2개가
개선됐다. 그러나 전체 ESM과 Argument Exact는 같고 State/Tool F1은 각각 0.008
낮아서, 단일 run만으로 최종 성능 향상을 확정할 수는 없다.

기본 경로 승격 전 동일-cache 반복 실행으로 변화량이 Agent 변동보다 안정적인지
확인한다. 현재는 opt-in 상태를 유지한다.

후속 reranker 구현은 기존 route를 보존하면서 Fact 기반 복구 후보를 최고 점수 1개로
제한하고 후보 선정 근거를 prompt에 제공한다. 고정-cache audit은 통과했으며 live E2E
재평가 결과는 다음과 같다.

## 후보 reranker 재평가

- 수정판 run: `cd0adf63-01a9-4087-b9e5-11113fb7ff83`
- 기준선은 위와 같은 `6f72b35d-cda4-482e-a960-9671daf2933e`이다.
- Scenario 6–10의 base Fact, assisted Fact, Recursive Summary component cache key가
  모두 동일함을 다시 확인했다.

| 지표 | 기존 Ours | 후보 reranker |
| --- | ---: | ---: |
| Exact State Match | 0.52 | 0.58 |
| State F1 | 0.6558 | 0.7358 |
| Tool F1 | 0.5513 | 0.5633 |
| Argument Exact | 0.48 | 0.50 |
| Tool selector miss | 16 | 12 |
| 평균 discovery call | 0.40 | 0.22 |
| 평균 선택 Tool 수 | 2.88 | 3.48 |
| 불필요한 vehicle call | 21 | 26 |

실제 selector membership이 바뀐 30개 task에서는 ESM 개선 4개, 악화 0개였다.
전체에서는 개선 4개와 악화 1개로 순증 3개다. 유일한 악화 task는 selector membership이
기준선과 같아 후보 복구 자체의 악화로 보기는 어렵다. 다만 불필요 호출이 5개 증가한
trade-off가 있으므로 반복 평가와 extra-call 원인 축소 전까지 opt-in을 유지한다.
