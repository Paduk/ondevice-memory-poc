# Schema-informed Fact Scenario 6 저장 smoke 결과

상태: `Fact-only v1·sparse v2·speaker entity v1 및 Scenario 6 E2E 완료`

## 조건

- profile: `cloud_schema_informed_fact_patch`
- Memory model: `gpt-5.6-luna`, reasoning `low`
- Embedding: `text-embedding-3-small`, 256 dimensions
- Fact batch: 최대 32 turns / 4,096 tokens
- History: 2,698 turns, 85 Fact LLM calls
- Quiz Agent와 생성형 답변 호출: 0
- 비교 기준: Scenario 6의 검수된 10개 evidence에 대해 expected ontology
  capability, target, Gold value가 최종 active Fact에 존재하는지 확인

## 결과

| 단계 | 결과 |
| --- | ---: |
| matcher evidence Top-2 | 10/10 |
| matcher 실제 batch Top-8 | 8/10 |
| 정답 evidence에서 Fact candidate 생성 | 6/10 |
| 최종 active 정답 Fact | 4/10 |
| 기존 Ours 검수상 정답 Fact | 2/10 |

최종 active 정답 Fact는 `vehicle-06-02`, `vehicle-06-03`,
`vehicle-06-04`, `vehicle-06-09`였다. 기존 Ours 대비 coverage는 두 배지만
절대 증가는 2건이므로 큰 개선으로 단정하기에는 부족하다.

## 실패 분해

| Task | 단계 | 원인 |
| --- | --- | --- |
| `vehicle-06-00` | LLM 생성 | batch 후보에는 `audio.volume`이 있었지만 candidate를 만들지 않음 |
| `vehicle-06-01` | LLM 생성 | batch 후보에는 `display.brightness`가 있었지만 candidate를 만들지 않음 |
| `vehicle-06-05` | matcher | `media.quality`가 batch Top-8에서 탈락 |
| `vehicle-06-06` | semantic gate | candidate는 생성됐지만 auto-reverse 의미가 원문에 명시되지 않았다고 거절 |
| `vehicle-06-07` | matcher | `climate.temperature`가 batch Top-8에서 탈락 |
| `vehicle-06-08` | linker | fan speed 10을 저장했지만 사람 identity가 분리되지 않아 이후 fan speed 3에 대체됨 |

ontology local validator가 거절한 candidate는 0건이었다. 주요 병목은
ontology schema 자체보다 batch 후보 축약, contextual preference 생성 판단,
그리고 `entity_id=vehicle_scenario_6`로 수렴한 뒤 서로 다른 사람의 Fact를
연결하는 기존 linker에 있다.

## 호출량

| 항목 | Schema-informed | 기존 Ours |
| --- | ---: | ---: |
| Fact LLM calls | 85 | 85 |
| Fact input tokens | 411,358 | 244,293 |
| Fact output tokens | 56,713 | 5,647 |
| Fact total tokens | 468,071 | 249,940 |

모든 2,698개 user message에 assessment를 반환하게 한 현재 형식 때문에
출력 token이 크게 늘었다. 따라서 이 형식을 그대로 6–10 전체 평가에
확대하기 전에 assessment 출력 압축이 필요하다.

### 후속 효율화

`schema-informed-fact-memory-candidate-v2`에서는 LLM이 `candidate`와
`uncertain`만 sparse assessment로 반환하고, 생략된 message는 런타임이
`no_durable_vehicle_fact`로 계산한다. 진단 count와 candidate-evidence 연결
검증은 유지하며 prompt/schema version을 올려 기존 cache와 분리했다.

동일한 Scenario 6 조건으로 재실행한 결과는 다음과 같다.

| Fact extraction | 전체 assessment v1 | Sparse v2 | 변화 |
| --- | ---: | ---: | ---: |
| Calls | 85 | 85 | 0 |
| Input tokens | 411,358 | 409,367 | -0.5% |
| Output tokens | 56,713 | 8,361 | **-85.3%** |
| Total tokens | 468,071 | 417,728 | **-10.8%** |
| Correct active Fact | 4/10 | 4/10 | 0 |

semantic validation까지 포함하면 전체 Memory token은 475,561에서
422,814로 11.1% 줄었다. v2의 sparse count는 candidate message 12,
uncertain 2, 로컬에서 추론한 no-durable message 2,684로 합계 2,698개이며,
모든 History message에 대한 진단 coverage도 유지됐다. Candidate 수는
14개에서 11개로 달라졌지만 정답 evidence candidate는 6/10, 최종 active
정답 Fact는 4/10으로 동일해 이번 실행에서는 정확도 저하가 관찰되지 않았다.

### Speaker entity 후속 수정

`vehicle-history-speaker-v1`은 candidate evidence의 `message_id`가 가리키는
History line에서 speaker를 로컬로 읽고 canonical `entity_id`로 사용한다.
추가 LLM 호출은 없으며, 한 candidate의 evidence에 서로 다른 speaker가
섞이면 보수적으로 거절한다. cache config에 entity-resolution version을
포함해 이전 sparse cache와 분리했다.

Scenario 6 재실행에서는 13개 candidate가 모두 실제 speaker로 정규화됐고
speaker ambiguity 거절은 0건이었다. Jonathan의 `fan_speed=10`과 Michael의
`fan_speed=3`은 서로 다른 entity로 저장되어, 기존처럼 뒤의 값이 앞의 값을
덮지 않았다. 따라서 task08에 필요한 predicate/value/evidence Fact가 최종
active로 보존됐다. 이 record의 target은 `unspecified`였지만 ontology상 가능한
Tool target이 `climate` 하나뿐이므로 실행 시 유일하게 binding할 수 있다.

| Sparse profile | 기존 v2 | Speaker entity v1 |
| --- | ---: | ---: |
| Fact calls | 85 | 85 |
| Input tokens | 409,367 | 431,329 |
| Output tokens | 8,361 | 9,462 |
| Active records | 8 | 10 |
| 정답 predicate/value/entity active Fact | 4/10 | 5/10 |

활성 Fact가 더 많이 보존되어 이후 batch의 linking context input은 늘었지만,
output token은 전체-assessment v1의 56,713보다 83.3% 낮다. 한편 Michael의
`climate.fan_speed=3`을 같은 사람의 `climate.circulation=outside` UPDATE로
잘못 묶는 cross-predicate linker 오류가 별도로 확인됐다. speaker 수정은
사람 간 overwrite를 해결하지만, 같은 사람의 서로 다른 predicate를
alias로 오인하는 문제까지 해결하지는 않는다.

## 판단

세밀한 matcher 튜닝보다 다음 두 가지 범용 수정의 우선순위가 높다.

1. evidence가 속한 History line의 speaker를 canonical entity로 사용해
   사람 간 overwrite를 막는다.
2. per-message assessment 전체 출력을 없애거나 압축해 호출 비용을 낮춘다.

Sparse assessment와 speaker entity resolution은 유지한다. 다음 범용 수정은
linker가 서로 다른 canonical predicate를 UPDATE 대상으로 선택하지 못하게
하는 것이다. 아래 Agent E2E 결과상 selector·argument binding도 함께
분리해서 개선해야 한다.

## Speaker entity Scenario 6 E2E

새 speaker entity cache를 재사용해 10문제를 실행한 ESM은 3/10이었다.
성공은 task02, task04, task09였으며, 전체 성능 향상은 확인되지 않았다.

다만 entity 수정의 직접 목표였던 task08에서는 Jonathan의
`climate_fan_speed=10`이 retrieval rank 1로 선택됐고 Agent도 정확한 Tool과
`speed=10`을 호출했다. 실패 원인은 Memory가 아니라 runtime binding이
query의 현재 driver 역할을 적용해 `zone=driver`를 만든 반면 Gold는
`zone=all`을 요구한 것이다. task03도 동일하게 정답 circulation Fact를
가져왔지만 `zone=driver` 대 `zone=all` 차이로 실패했다.

따라서 speaker entity 수정은 사람 간 overwrite와 해당 Fact retrieval을
해결했지만, ESM 병목은 다음 단계인 selector·argument binding으로 이동했다.
현재 확인된 범용 후속 대상은 서로 다른 canonical predicate 간 UPDATE 금지와,
Fact에 zone이 명시되지 않았을 때 query의 현재 역할을 자동 주입할지 여부를
Tool/benchmark 의미에 맞게 결정하는 것이다.

## Artifact

- Cache:
  `ubuntu/evaluation/vehiclemembench-memory/d745b359794da154/scenario-06/d234d44e92494b93`
- Session: `b64ce337-ea02-4c33-88dd-aea609e39543`
- Sparse v2 cache:
  `ubuntu/evaluation/vehiclemembench-memory/d745b359794da154/scenario-06/c5d6dea3a3d9372f`
- Sparse v2 session: `55215d55-4502-45c3-89a9-6465a13a93e5`
- Speaker entity cache:
  `ubuntu/evaluation/vehiclemembench-memory/d745b359794da154/scenario-06/b36aa7639cbe70b0`
- Speaker entity session: `291325cf-e1ec-4a7f-bbef-6386b91a908d`
- Speaker entity E2E:
  `ubuntu/evaluation/vehiclemembench/a16a8c02-2e57-4427-9d8f-c6741fbe169c`
