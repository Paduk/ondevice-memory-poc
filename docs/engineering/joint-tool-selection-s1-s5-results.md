# Joint Tool Selection Scenario 1–5 결과

상태: `후보 reranker 동일-cache paired E2E 1회 완료, 반복 평가 전`

## 비교 조건

- 범위: VehicleMemBench Scenario 1–5, 50 task
- Agent: `gpt-5.6-terra`
- 기준선 run: `aa9417b8-f918-433c-b379-99e269869430`
- 공동 선택 run: `10392a4a-76fa-453c-9549-e5ab2c782fb8`
- 모든 scenario에서 base Fact, assisted Fact, Recursive Summary component cache key가
  두 run 사이에 동일함을 확인했다.

## 결과

| 지표 | 기존 Ours | 공동 Tool 선택 |
| --- | ---: | ---: |
| Exact State Match | 0.52 | 0.50 |
| State F1 | 0.7163 | 0.7005 |
| Tool F1 | 0.5653 | 0.5347 |
| Argument Exact | 0.44 | 0.44 |
| Tool selector miss | 13 | 11 |
| 평균 discovery call | 0.34 | 0.28 |
| 평균 선택 Tool 수 | 2.32 | 4.10 |
| 불필요한 vehicle call | 26 | 31 |
| Agent input tokens | 181,687 | 184,700 |

- selector가 달라진 task는 21개였다.
- 이 집합에서 ESM 개선 1개, 악화 2개였다.
- 개선 task에서는 기존 selector가 놓친 정답 Tool을 공동 selector가 포함했다.
- 악화 task에서는 정답 Tool이 후보에 있었지만, 후보가 같은 domain의 여러 Tool로
  넓어진 뒤 Agent가 호출하지 않았다.
- Desired-state plan 적용은 2개였고 multi-Fact plan은 0개였다. 개선 2의 효과를
  평가하는 자료는 아니다.

## 판단

개선 1은 S6–10과 마찬가지로 selector miss와 discovery call을 줄였지만, S1–5에서는
후보 확장으로 불필요한 호출이 증가하고 최종 ESM/F1이 낮아졌다. 따라서 현재의
deterministic union/ranking을 그대로 기본 경로로 채택하지 않는다.

다음 수정은 같은 domain의 유사 Tool을 최대치까지 펼치는 대신 query와 Fact가 직접
지지하는 소수 후보를 우선하고, 선정 근거를 Agent context에 명시하는 것이다. 수정 후
동일 cache로 다시 평가한다.

후속 reranker 구현은 완료됐다. 고정-cache S1–10 audit에서는 평균 후보 3.23개,
reference Tool 완전 포함 60/100을 기록했다.

## 후보 reranker 재평가

- 수정판 run: `152ab674-d4ce-42bb-8628-cdb3ad1fb795`
- 기준선은 위와 같은 `aa9417b8-f918-433c-b379-99e269869430`이다.
- Scenario 1–5의 base Fact, assisted Fact, Recursive Summary component cache key가
  모두 동일함을 다시 확인했다.

| 지표 | 기존 Ours | 후보 reranker |
| --- | ---: | ---: |
| Exact State Match | 0.52 | 0.62 |
| State F1 | 0.7163 | 0.7738 |
| Tool F1 | 0.5653 | 0.6270 |
| Argument Exact | 0.44 | 0.54 |
| Tool selector miss | 13 | 8 |
| 평균 discovery call | 0.34 | 0.26 |
| 평균 선택 Tool 수 | 2.32 | 2.98 |
| 불필요한 vehicle call | 26 | 22 |
| Agent input tokens | 181,687 | 193,592 |

전체 ESM은 개선 5개, 악화 0개로 순증 5개다. 실제 selector membership이 달라진
33개 task에서는 ESM 개선 3개, 악화 0개였고, selector가 같은 task에서도 2개가
개선됐다. State F1은 6개 개선·2개 악화, Tool F1은 5개 개선·2개 악화였다.

초기 공동 선택판과 비교하면 평균 후보 수는 4.10에서 2.98로 줄면서 ESM은
0.50에서 0.62로 회복·상승했다. 불필요 호출도 기준선보다 4개 줄었지만 Agent input은
약 6.6% 증가했다. Desired-state plan은 4개 task에서 적용됐고 multi-Fact call은
0개이므로 이번 결과도 개선 2의 효과를 입증하지는 않는다. 개선 1에는 강한 긍정
신호지만, 모델 실행 변동을 분리하기 위한 동일-cache 반복 평가 전까지 opt-in을
유지한다.
