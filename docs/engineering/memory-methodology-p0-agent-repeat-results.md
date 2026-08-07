# P0 고정 Memory Agent 반복 평가 결과

상태: `완료 — Post-normalized 유지, ESM 개선 주장은 보류`

## 목적과 조건

Recursive-assisted Ours와 Post-normalized Ontology-assisted의 단일 run ESM
`0.60` 동률이 Agent 실행 변동인지 확인했다. 각 방법에서 Scenario 6–10 Memory
cache 하나를 고정하고 동일한 50 task를 Agent `gpt-5.6-terra`로 3회 평가했다.

두 방법은 같은 Recursive Summary component를 사용했다. 각 방법 안에서는 3회
모두 Scenario별 cache key와 `memory_fingerprint`가 일치했다. Retrieval trace는
DB에 추가되므로 파일 전체 hash가 아니라 Fact·Summary 내용을 반영하는 runtime
fingerprint로 불변성을 확인했다.

기존 7월 Recursive-assisted run의 Summary component가 cache에서 정리되어 있어
그 run은 P0 반복에 섞지 않았다. 현재 Post-normalized와 같은 Summary component로
Recursive-assisted cache를 한 번 생성한 뒤 그 cache를 세 Agent 결과에 고정했다.

## 3회 결과

| 방법 | Run | ESM | State F1 | Tool F1 | Arg exact | Recall@k |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Recursive-assisted | `60795558-33e4-47b5-80e9-0d677ed27a08` | 0.58 | 0.683 | 0.579 | 0.44 | 0.627 |
| Recursive-assisted | `3f13fc70-8e4b-40cd-beba-3c7a5ec39a09` | 0.62 | 0.735 | 0.620 | 0.46 | 0.627 |
| Recursive-assisted | `a336f01b-1ef6-41b5-a4f2-6c34625e6952` | 0.58 | 0.687 | 0.553 | 0.42 | 0.627 |
| Post-normalized | `15f5a789-8dd5-4d6c-9190-887290bd5bff` | 0.60 | 0.745 | 0.606 | 0.48 | 0.667 |
| Post-normalized | `90617ffa-9c7c-4b19-a598-646f69574e8a` | 0.60 | 0.685 | 0.613 | 0.50 | 0.667 |
| Post-normalized | `aa143ce5-6c9d-40e9-aac9-7c694b9e38d4` | 0.64 | 0.807 | 0.640 | 0.54 | 0.667 |

## 평균

| 방법 | ESM | State F1 | Tool F1 | Arg exact | Recall@k |
| --- | ---: | ---: | ---: | ---: | ---: |
| Recursive-assisted | 0.593 | 0.702 | 0.584 | 0.440 | 0.627 |
| **Post-normalized** | **0.613** | **0.746** | **0.620** | **0.507** | **0.667** |
| 차이 | +0.020 | +0.044 | +0.036 | +0.067 | +0.040 |

Post-normalized는 150개 task-run 중 92개, Recursive-assisted는 89개에서 ESM에
성공했다. 그러나 두 방법의 ESM run 범위가 각각 0.04이고, 50개 task의 반복 성공률
차이를 이용한 거친 95% 구간은 `-0.094~+0.134`로 0을 포함한다. `+0.020`을 확정적인
ESM 개선으로 해석하지 않는다.

## Task 안정성

| 방법 | 항상 성공 3/3 | 항상 실패 0/3 | 실행마다 변동 |
| --- | ---: | ---: | ---: |
| Recursive-assisted | 26 | 15 | 9 |
| Post-normalized | 25 | 13 | 12 |

Post-normalized가 더 자주 성공한 task와 Recursive-assisted가 더 자주 성공한
task는 각각 10개로 같다. Post-normalized만 3/3 성공하고 기존 방식은 0/3인 task는
3개이며, 반대는 2개다. 두 방법 모두 0/3인 안정 실패는 7개다.

시나리오별 평균 ESM은 다음과 같다.

| Scenario | Recursive-assisted | Post-normalized |
| ---: | ---: | ---: |
| 6 | 0.600 | **0.767** |
| 7 | 0.300 | **0.367** |
| 8 | 0.733 | **0.867** |
| 9 | **0.600** | 0.400 |
| 10 | **0.733** | 0.667 |

개선이 모든 Scenario에서 같은 방향은 아니며 Scenario 9에서는 명확한 회귀가
있다. 이는 ESM 평균 차이가 작은 이유와 task churn을 함께 설명한다.

## 판정

1. Post-normalized는 평균 ESM이 낮지 않고 Recall, State F1, Tool F1, argument
   exact가 모두 높아 현재 연구 후보로 유지한다.
2. Ontology가 최종 ESM을 확실히 높였다는 주장은 하지 않는다.
3. 다음 단계는 Ontology 확장보다 두 방법이 모두 실패한 7개 안정 실패를 출발점으로
   selector와 argument binding의 범용 원인을 분리하는 P1이다.
4. P1에서는 task ID나 Gold 값을 규칙에 넣지 않고 Tool schema 기반 수정만 허용한다.

