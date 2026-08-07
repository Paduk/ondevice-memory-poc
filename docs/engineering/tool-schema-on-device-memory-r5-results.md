# Fact-First Memory R5 10% 평가 결과

상태: `Scenario 1–10 방법별 100 task 평가 완료 — 500-task 확대 보류`

평가일: 2026-07-29

## 1. 평가 조건

- VehicleMemBench Scenario 1–5, 총 50 task
- Agent: `gpt-5.6-terra`
- Memory: `gpt-5.6-luna`
- Embedding: `text-embedding-3-small`, 256 dimensions
- Fact 생성: 최대 32 turns 또는 4,096 source tokens 단위 batch
- Retrieval: Fact-first hybrid top-k + 요청 시 Tool late binding
- Gold 정보는 Oracle 진단에서만 사용

주요 artifact:

- [Fact-first](../../ubuntu/evaluation/vehiclemembench/facf95c3-ebbe-4fb8-bbfc-1edd1c91311d/results.md)
- [Fact-first + Gold Tool](../../ubuntu/evaluation/vehiclemembench/558c60f5-ef0d-49d6-b068-8e65a298b691/results.md)
- [Schema-first + Gold Tool](../../ubuntu/evaluation/vehiclemembench/fe8c2185-dc8c-445a-abd6-8a07adcb0391/results.md)
- [Structured Hybrid](../../ubuntu/evaluation/vehiclemembench/84ec01dd-34ea-49fe-9808-ceb6cdd070b6/results.md)
- [Schema-first](../../ubuntu/evaluation/vehiclemembench/403cbce2-5351-4255-9723-c63baf4098e7/results.md)
- [Scenario 6–10 holdout 비교](tool-schema-on-device-memory-holdout-s6-s10-results.md)

## 2. 50-task 결과

| 방법 | ESM | State F1 | Tool F1 | Argument exact | Recall@k | Agent input |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No Memory | 0.20 | 0.360 | 0.210 | 미집계 | - | 120,428 |
| Schema-first | 0.30 | 0.460 | 0.287 | 0.22 | 0.11 | 76,858 |
| Schema-first + Gold Tool | 0.42 | 0.673 | 0.433 | 0.34 | 0.227 | 61,326 |
| Structured Hybrid | 0.56 | 0.873 | 0.627 | 0.50 | 0.47 | 201,902 |
| **Fact-first** | **0.52** | **0.698** | **0.556** | **0.48** | **0.45** | **162,518** |
| Fact-first + Gold Tool | 0.56 | 0.781 | 0.577 | 0.54 | 0.49 | 103,824 |
| Gold Memory | 0.90 | 0.975 | 0.913 | 미집계 | - | 156,969 |

Fact-first는 Schema-first 대비 ESM `+0.22`, Recall@k `+0.34`이고,
Structured Hybrid 대비 ESM `-0.04`, Recall@k `-0.02`이다. Hybrid보다 Agent
input은 19.5% 적지만 선택 Memory context는 평균 357.7 tokens로 Hybrid의
239.1 tokens보다 49.6% 많다.

Gold Tool을 주면 Fact-first ESM은 0.52에서 0.56으로 증가한다. Routing 개선
여지는 있으나 성능 차이가 0.04이므로 남은 오차는 Fact 누락·조건 검색·
late-binding·Agent argument 결정에도 분산되어 있다.

## 3. Memory 생성과 Validation

- extraction 410 batch 호출, 불확실 후보 semantic validation 28 batch 호출
- candidate 76개
- 적용 68개: `ADD 60`, `UPDATE 8`
- 중복 `NOOP 1`, semantic reject 1, review 보존 6
- 직접 적용률 89.5%, 적용 또는 NOOP 해결률 90.8%
- active record 60개, superseded version 8개
- 생성 입력 1,338,100 tokens, 출력 37,845 tokens
- 생성 단계 Cloud 전송 전 탐지된 민감 span 1개를 redaction했고 노출은 0개

과거 Schema proposal의 relaxed Validation replay는 50개 고유 후보 중 기존
37개 적용에 더해 12개를 추가 수용하고 1개를 review로 돌릴 수 있다고
예측했다. 이 replay는 LLM을 다시 호출하지 않은 후보 생존 분석이며, 완화
record를 DB에 적용한 E2E 점수는 아니다.

## 4. 비용과 효율

2026-07-29 OpenAI 표준 단가를 적용한 uncached 추정치다.

| 구분 | 추정비용 |
| --- | ---: |
| Fact Memory 생성·semantic validation | $1.5652 |
| Fact retrieval embedding | $0.0003 |
| Fact-first Agent 50 task | $0.5233 |
| **Fact-first 주 평가 합계** | **$2.0888** |
| Gold Tool + Fact 추가 진단 | $0.3512 |
| Gold Tool + Schema 추가 진단 | $0.2644 |

Structured Hybrid를 같은 단가로 환산하면 Memory와 Agent를 합해 약 $1.2273
이다. Fact-first는 Agent 입력은 줄였지만 Memory 생성 호출이
`68 Memory LLM calls → 410 extraction + 28 validation calls`로 증가해 전체
비용은 더 높다. 중단한 초기 prompt probe 비용 약 $0.09를 포함한 R5 실제
증분 추정치는 약 $2.79다. 재사용 cache의 과거 생성비는 Oracle 진단
증분비용에 다시 포함하지 않았다.

## 5. 실패 분석

Fact-first의 ESM 실패는 24/50 task이며 원인은 서로 중복될 수 있다.

- reference Tool route 누락: 14 task
- 선택 Fact의 실행 argument 변환 거절: 23 task
- Agent Tool omission: 10 task
- Agent argument mismatch: 7 task
- Tool 실행 오류: 2 task
- retrieval이 완전히 빈 task: 0

실행 argument 변환 거절 152건 중 149건은
`incomplete_or_invalid_arguments`다. top-k Fact는 검색됐지만 하나의 Fact만으로
필수 runtime argument를 완성하지 못한 경우까지 모두 hint 생성 실패로
기록된 결과다. 다음 개선은 불필요한 변환 시도를 줄이고 여러 Fact·현재
역할·질문 값을 조합하는 late-binding planner를 강화해야 한다.

## 6. Gate 판정

| Gate | 결과 | 판정 |
| --- | ---: | --- |
| candidate acceptance ≥ 80% | 89.5% | 통과 |
| 합성 correction UPDATE | 100% 회귀 테스트 | 통과 |
| Recall@k ≥ 0.35 | 0.45 | 통과 |
| ESM ≥ 0.50 | 0.52 | 통과 |
| Hybrid 대비 Agent input 감소 | -19.5% | 통과 |
| Hybrid 대비 Memory context 감소 | +49.6% | 실패 |

핵심 정확도 gate는 통과했지만 Hybrid보다 State/Tool F1이 낮고 context가 더
크다. 따라서 바로 50 Scenario·500 task로 확대하지 않고, R5.1에서
retrieval rerank와 multi-Fact late binding을 개선한 뒤 50-task를 재평가한다.

추가로 동결된 설정을 Scenario 6–10에 적용한 holdout 결과는 Hybrid ESM
0.56, Fact-first ESM 0.50이다. 방법별 Scenario 1–10 총 100 task에서도
Hybrid 0.56, Fact-first 0.51로 성능 격차가 재현됐다. 자세한 결과는
[holdout 평가](tool-schema-on-device-memory-holdout-s6-s10-results.md)에
기록한다.
