# Post-normalized Recursive-assisted Ours Scenario 6 결과

상태: `구현 및 Scenario 6 E2E smoke 완료, Scenario 6–10 후속 평가 완료`

## 변경 범위

기존 profile `cloud_schema_informed_recursive_assisted_fact_patch`의 실패한
앞단 제약 방식을 post-normalized v1으로 교체했다.

1. Base와 assisted pass는 기존 Recursive-assisted Ours의 high-recall prompt와
   `_FactMemoryBatchPayload`를 그대로 사용한다.
2. LLM 입력에는 Ontology Top-k를 넣지 않는다.
3. 생성된 candidate를 한 batch로 로컬 lexical+embedding matcher에 보낸다.
4. capability, target, typed value를 모두 안전하게 결정할 수 있을 때만 canonical
   Fact로 변환한다.
5. 불확실하거나 값 schema가 맞지 않으면 candidate를 거절하지 않고 원본
   predicate, value, applicability, evidence를 그대로 저장한다.

Evidence speaker 정규화는 모든 candidate에 적용한다. Linker는 canonical
predicate에만 exact-match를 적용하고, fallback된 자유형 predicate는 기존 alias
linking을 유지한다. Cache composition은
`post-normalized-recursive-assisted-fact-extraction-v1`으로 분리했다.

## Scenario 6 결과

| 방법 | ESM | State F1 | Tool F1 | Argument exact | Recall@k |
| --- | ---: | ---: | ---: | ---: | ---: |
| 기존 Recursive-assisted | 0.50 | - | - | - | - |
| 앞단 Ontology-assisted | 0.50 | 0.687 | 0.467 | 0.40 | 0.600 |
| **Post-normalized assisted** | **0.80** | **0.926** | **0.807** | **0.70** | **0.700** |

성공은 task00, 01, 02, 03, 04, 06, 07, 09다. 실패는 다음 두 건이다.

- task05: media quality 480p와 video switch가 여전히 누락됨
- task08: fan speed 10 Fact와 값은 맞지만 `zone=driver` 대 Gold `all`

task07의 temperature 21은 post-normalized assisted pass가 새로 저장해 직접
회복했다. task03과 task06의 개선에는 같은 Fact context에서 Agent가 이번에는
정확한 zone/side를 선택한 실행 변동도 포함되므로 0.80 전체를 Memory 개선으로
귀속하지 않는다.

## Candidate 보존과 정규화

| 항목 | 결과 |
| --- | ---: |
| LLM candidate | 17 |
| 저장 candidate / active Fact | 17 / 17 |
| Canonicalized | 6 |
| 원본 fallback | 11 |
| Speaker resolved / rejected | 17 / 0 |

Base pass는 11개 중 2개, Summary-assisted pass는 6개 중 4개를 canonicalize했다.
Fallback 원인은 ambiguous match 7, value mapping 불확실 3, 명시 target 미지원 1이다.
어느 경우에도 candidate를 삭제하지 않았다.

비교하면 기존 일반 Recursive-assisted Scenario 6은 candidate 18, active Fact 12,
앞단 Ontology 방식은 15/13이었다. 이번 17/17은 생성 coverage를 거의 원래
수준으로 복구하면서 사람·predicate overwrite를 방지한 결과다.

## 호출량

| Fact extraction | Calls | Tokens |
| --- | ---: | ---: |
| 기존 Recursive-assisted cache | 170 | 529,042 |
| 앞단 Ontology-assisted | 170 | 958,438 |
| Post-normalized assisted | 170 | 617,951 |

Post-normalized 방식의 local ontology embedding은 4,148 input tokens였다.
앞단 Ontology 방식보다 Fact extraction token은 35.5% 적지만, active linking
context가 17개로 늘어 기존 Recursive-assisted보다 16.8% 많다.

## 판단

구현 목표인 candidate 비제약, 불확실 시 원본 보존, 확실한 후보만 canonicalize,
speaker 분리와 mixed linking이 모두 확인됐다. Scenario 6 ESM도 유망하지만 단일
Agent run의 실행 변동이 있으므로 채택 판단은 동일 설정 Scenario 6–10 50-task
평가 후 내린다.

후속 50-task run에서 같은 Scenario 6 Memory cache의 ESM은 0.60으로 달라졌다.
전체 ESM은 기존 Recursive-assisted와 같은 0.60이고 Recall@k는 0.603에서
0.667로 개선됐다. 최종 판단은
[Scenario 6–10 결과](post-normalized-recursive-assisted-s6-s10-results.md)를 따른다.

## 검증

- 관련 테스트: 83 passed, 1 skipped
- Ruff: 통과
- E2E: `ubuntu/evaluation/vehiclemembench/d70abb45-3ce7-4ce3-b235-ac74ef924fe2`
- Base cache: `ubuntu/evaluation/vehiclemembench-memory/d745b359794da154/scenario-06/6c2ce7462b6f8da7`
- Assisted cache: `ubuntu/evaluation/vehiclemembench-memory/d745b359794da154/scenario-06/541dbea1771c776e`
