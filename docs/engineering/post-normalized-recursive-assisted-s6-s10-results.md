# Post-normalized Recursive-assisted Ours Scenario 6–10 결과

상태: `50-task E2E 완료, 구조·검색 개선 확인, 주 방법 채택은 반복 평가 전 보류`

## 조건

- profile: `cloud_schema_informed_recursive_assisted_fact_patch`
- Scenario 6–10, 각 10개 Quiz, 총 50 task
- Quiz model: `gpt-5.6-terra`
- Memory model: `gpt-5.6-luna`
- Embedding: `text-embedding-3-small`, 256 dimensions
- 기존 high-recall Fact 생성 후, 로컬 Ontology matcher가 확실한 후보만 정규화
- 불확실한 후보는 원문 predicate, value, applicability, evidence를 보존

## 전체 결과

| 방법 | ESM | State F1 | Tool F1 | Argument exact | Recall@k |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ours | 0.50 | 0.661 | 0.507 | 0.42 | 0.490 |
| Recursive-assisted Ours | **0.60** | 0.728 | 0.573 | 0.44 | 0.603 |
| 앞단 Ontology-assisted | 0.56 | 0.722 | 0.573 | **0.50** | 0.563 |
| **Post-normalized assisted** | **0.60** | **0.745** | **0.606** | 0.48 | **0.667** |

Post-normalized 방식은 실패했던 앞단 Ontology 방식보다 ESM `+0.04`, Recall@k
`+0.103`으로 회복했다. 직접 비교 대상인 기존 Recursive-assisted와는 ESM이
`0.60`으로 같지만 State F1 `+0.017`, Tool F1 `+0.033`, argument exact `+0.04`,
Recall@k `+0.063`이다.

따라서 Ontology를 생성 제약이 아니라 생성 후 보수적 정규화로 쓰는 방향은
타당하다. 다만 현재 단일 run에서는 최종 ESM 상승까지 확인되지 않았다.

## 시나리오별 ESM

| Scenario | Recursive-assisted | 앞단 Ontology | Post-normalized |
| ---: | ---: | ---: | ---: |
| 6 | 0.50 | 0.50 | **0.60** |
| 7 | 0.50 | 0.50 | 0.50 |
| 8 | 0.80 | 0.80 | **0.90** |
| 9 | 0.30 | **0.50** | 0.30 |
| 10 | **0.90** | 0.50 | 0.70 |

기존 Recursive-assisted 대비 새 성공 10건, 새 실패 10건, 공통 성공 20건,
공통 실패 10건이다. 같은 30건 성공이라도 task가 크게 교체됐으므로 ESM 동률을
안정적인 동작 일치로 해석할 수 없다.

Scenario 6 smoke는 같은 고정 Memory cache에서 ESM 0.80이었지만 이번 전체 run의
Scenario 6 subset은 0.60이었다. 이 차이는 Memory 생성이 아니라 Quiz Agent 실행
변동이 최종 ESM에 상당히 개입한다는 직접적인 예다.

## Candidate 보존과 정규화

| 항목 | 결과 |
| --- | ---: |
| Fact extraction calls | 899 |
| LLM 생성 candidate | 108 |
| Speaker resolved / rejected | 104 / 4 |
| Canonicalized | 36 |
| 원본 fallback | 68 |
| 최종 candidate / active Fact | 104 / 81 |
| Superseded versions | 6 |

Post-normalizer에 도달한 104개 candidate는 36개만 canonicalize하고 나머지 68개를
모두 원형으로 보존했다. fallback 원인은 value mapping 불확실 32, ambiguous match
21, 명시 target 미지원 8, capability term 없음 6, 약한 lexical match 1이다.

Speaker 단계에서 4개가 거절됐고 이후 semantic validator에서 5개가 거절됐다.
즉 post-normalization 자체가 후보를 삭제한 사례는 없지만, 전체 파이프라인의
모든 후보가 무조건 저장되는 것은 아니다.

앞단 Ontology 방식에서 Scenario 10에 없었던 정답 Fact 4개 중 steering-wheel
heating level 2, navigation volume 90, display language English 3개가 이번 active
Fact에 복구됐다. feet/window airflow는 여전히 없다. 생성 coverage 회복이
Scenario 10 ESM `0.50 → 0.70`에 기여했지만 완전 복구에는 이르지 못했다.

## 비용과 구조 품질

| 항목 | Recursive-assisted | Post-normalized | 변화 |
| --- | ---: | ---: | ---: |
| Candidate | 115 | 104 | -9.6% |
| Active Fact | 79 | 81 | +2.5% |
| Superseded versions | 15 | 6 | -60.0% |
| Fact model tokens | 2,874,348 | 3,205,699 | +11.5% |
| Semantic-review tokens | 104,126 | 88,494 | -15.0% |

추가 local Ontology embedding input은 15,217 tokens다. Candidate 11개 감소는
post-normalizer의 삭제가 아니라 별도 LLM extraction run의 생성 변동에서
발생했다. 반면 active Fact는 2개 늘고 version은 9개 줄어, 보수적 canonical
predicate와 exact linking이 잘못된 UPDATE를 줄였다는 구조적 신호가 있다.

## 남은 병목과 판단

50개 task의 주요 실패 분류는 Tool selector miss 12건, memory argument mapping
rejection 7건, tool execution error 4건이다. Memory recall은 개선됐지만 검색된
정보를 정확한 Tool과 argument로 바꾸는 후단 병목이 계속 최종 ESM을 제한한다.

결론은 다음과 같다.

1. 앞단 Ontology 제약은 폐기하고 post-normalization 방향을 유지한다.
2. Recall, Tool F1, version 품질 개선은 의미가 있지만 ESM 개선 주장은 아직 하지
   않는다.
3. 주 profile로 확정하기 전, 각 방법의 고정 Memory cache를 재사용해 Quiz Agent만
   2–3회 반복하고 평균 ESM과 task별 안정성을 비교한다.
4. 이후 개선 우선순위는 Ontology를 더 세분화하는 것보다 selector와 late
   argument binding이다.

후속 고정 Memory 3회 반복에서 평균 ESM은 기존 Recursive-assisted 0.593,
Post-normalized 0.613이었다. `+0.020`은 run 변동 범위 안이므로 ESM 개선 주장은
보류하지만 Recall, State/Tool F1, argument exact 평균 우위는 유지됐다. 자세한
결과는 [P0 Agent 반복 평가](memory-methodology-p0-agent-repeat-results.md)에 있다.

## Artifact

- Post-normalized: `ubuntu/evaluation/vehiclemembench/15f5a789-8dd5-4d6c-9190-887290bd5bff`
- Recursive-assisted baseline: `ubuntu/evaluation/vehiclemembench/24dcf1f2-40af-48da-b893-420e4404525c`
- 앞단 Ontology-assisted: `ubuntu/evaluation/vehiclemembench/a92d4bdb-d7e5-49f8-860f-07a77131e0e6`
- Ours baseline: `ubuntu/evaluation/vehiclemembench/ffdd8096-c990-46f6-ab9f-0644f61df56a`
