# Fact-First vs Structured Hybrid Holdout 평가

상태: `Scenario 6–10, 방법별 50 task 완료`

평가일: 2026-07-29

## 1. 조건

- R5 Scenario 1–5 평가 이후 코드·prompt·top-k 설정을 변경하지 않음
- 처음 사용하는 VehicleMemBench Scenario 6–10
- Structured Hybrid와 Fact-first를 같은 run에서 평가
- Agent `gpt-5.6-terra`, Memory `gpt-5.6-luna`
- Gold Memory·Gold Tool 미사용

Artifact:

- [Scenario 6–10 결과](../../ubuntu/evaluation/vehiclemembench/ffdd8096-c990-46f6-ab9f-0644f61df56a/results.md)

## 2. 전체 결과

| 방법 | ESM | State F1 | Tool F1 | Argument exact | Recall@k | Agent input | Context tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Structured Hybrid | 0.56 | 0.751 | 0.607 | 0.48 | 0.567 | 182,311 | 235.8 |
| Fact-first | 0.50 | 0.661 | 0.507 | 0.42 | 0.49 | 170,545 | 360.1 |

Fact-first는 Hybrid보다 Agent input이 6.5% 적지만 Memory context는 52.7%
많다. ESM은 `-0.06`, State F1은 `-0.09`, Tool F1은 `-0.101`이다.

## 3. 시나리오별 ESM

| Scenario | Structured Hybrid | Fact-first |
| --- | ---: | ---: |
| 6 | 0.70 | 0.50 |
| 7 | 0.70 | 0.30 |
| 8 | 0.40 | 0.80 |
| 9 | 0.50 | 0.30 |
| 10 | 0.50 | 0.60 |

동일 task 비교에서 Fact-first가 이긴 task는 8개, 진 task는 11개, 같은
결과는 31개다. Scenario 8과 10에서는 Fact-first가 앞섰으므로 모든
도메인에서 일관되게 열세인 것은 아니지만 시나리오 편차가 크다.

## 4. Memory 생성·Validation

| 항목 | Structured Hybrid | Fact-first |
| --- | ---: | ---: |
| 후보 | 82 | 69 |
| 적용된 record version | 81 | 58 |
| 최종 active | 71 | 47 |
| superseded version | 10 | 11 |
| reject | 1 | 2 |
| review 보류 | 0 | 8 |
| 중복 NOOP | 0 | 1 |
| 주요 Memory LLM 호출 | 78 | 449 + semantic 26 |

Fact-first의 직접 적용률은 84.1%로 기준 80%를 통과했지만, active Memory가
Hybrid보다 24개 적다. Holdout 성능 하락에는 Fact 후보 누락과 review 보류로
인한 가용 Memory 부족이 실제로 기여했다.

## 5. 실패 분해

ESM 실패는 Hybrid 22건, Fact-first 25건이다. 원인은 중복 집계된다.

Fact-first:

- 25개 실패 task 모두에서 하나 이상의 argument late-binding 거절
- Tool selector miss 15건
- Tool omission 10건
- argument mismatch 9건
- 실패 task 평균 Recall@k 0.36

Structured Hybrid:

- Tool omission 19건
- argument mismatch 11건
- 실패 task 평균 Recall@k 0.424

공통 병목은 Memory 누락·검색과 Agent Tool/argument 결정이다. Fact-first
고유 병목은 적은 active Fact, 별도 Tool routing, multi-Fact argument
late-binding 실패다.

## 6. Scenario 1–10 누적

방법별 총 100 task의 단순 합산 결과:

| 방법 | ESM | State F1 | Tool F1 | Argument exact | Recall@k |
| --- | ---: | ---: | ---: | ---: | ---: |
| Structured Hybrid | 0.56 | 0.812 | 0.617 | 0.49 | 0.518 |
| Fact-first | 0.51 | 0.680 | 0.531 | 0.45 | 0.47 |

초기 구간에서 Fact-first ESM 0.52, holdout에서 0.50으로 큰 붕괴는 없었다.
그러나 Hybrid와의 ESM 격차 약 0.05 및 State/Tool F1 격차가 재현됐다.

## 7. 비용·판정

- Memory generation 605 calls
  - Hybrid Memory 78, generation embedding 52
  - Fact extraction 449, semantic validation 26
- retrieval embedding 105 calls
- Agent 100 task
- 전체 추정비용 `$3.2871`
- Cloud 전송 전 민감 span 16개 redaction, 노출 0
- Artifact privacy audit 통과

결론적으로 Fact-first 가설은 100-task에서 Schema-first보다 개선됐지만
Structured Hybrid를 넘지 못했다. 다음 평가는 새 시나리오를 더 사용하는
것보다 active Fact recall과 multi-Fact late binding을 먼저 개선한 뒤 동일
holdout cache·task에서 회귀 비교하는 편이 적절하다.
