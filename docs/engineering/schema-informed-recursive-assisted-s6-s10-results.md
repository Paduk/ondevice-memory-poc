# Schema-informed Recursive-assisted Ours Scenario 6–10 결과

상태: `50-task E2E 완료, 현재 profile 채택 보류`

## 조건

- profile: `cloud_schema_informed_recursive_assisted_fact_patch`
- Scenario 6–10, 각 10개 Quiz, 총 50 task
- Quiz model: `gpt-5.6-terra`
- Memory model: `gpt-5.6-luna`
- Embedding: `text-embedding-3-small`, 256 dimensions
- 비교 대상과 dataset, Agent prompt, batch, retrieval 설정 동일
- 각 방법의 수치는 현재 보존된 단일 50-task E2E run 기준

## 전체 결과

| 방법 | ESM | State F1 | Tool F1 | Argument exact | Recall@k |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ours | 0.50 | 0.661 | 0.507 | 0.42 | 0.490 |
| Recursive-assisted Ours | **0.60** | **0.728** | 0.573 | 0.44 | **0.603** |
| Ontology + Recursive-assisted Ours | 0.56 | 0.722 | **0.573** | **0.50** | 0.563 |

Ontology 방식은 Ours보다 ESM `+0.06`이지만, 직접 비교 대상인 기존
Recursive-assisted Ours보다 `-0.04`다. Tool F1은 사실상 같고 argument exact는
`+0.06`이지만 Fact coverage와 ESM이 낮아져 현재 형태를 개선으로 판단하지 않는다.

## 시나리오별 ESM

| Scenario | Ours | Recursive-assisted | Ontology-assisted |
| ---: | ---: | ---: | ---: |
| 6 | 0.50 | 0.50 | 0.50 |
| 7 | 0.30 | 0.50 | 0.50 |
| 8 | 0.80 | 0.80 | 0.80 |
| 9 | 0.30 | 0.30 | 0.50 |
| 10 | 0.60 | **0.90** | 0.50 |

Scenario 9의 `+0.20`보다 Scenario 10의 `-0.40`이 커서 전체 성능이 하락했다.

기존 Recursive-assisted와 taskwise 비교하면 새 성공 11건, 새 실패 13건,
공통 성공 17건, 공통 실패 9건이다. 동일 성공 subset에서 단순히 2건만 줄어든
것이 아니라 성공 task가 많이 교체됐으며, 순손실은 2건이다.

## Fact coverage와 비용

| 항목 | Recursive-assisted | Ontology-assisted | 변화 |
| --- | ---: | ---: | ---: |
| Candidate | 115 | 71 | -38.3% |
| Active Fact | 79 | 55 | -30.4% |
| Superseded versions | 15 | 3 | -80.0% |
| Fact extraction calls | 898 | 899 | +1 |
| Fact model tokens | 2,874,348 | 4,947,354 | +72.1% |
| 전체 Memory generation input | 3,528,988 | 5,515,743 | +56.3% |

canonical predicate exact-linking으로 잘못된 overwrite와 version 증가는 줄었지만,
Fact 수 감소가 precision 향상만을 뜻하지는 않았다. recall과 ESM이 함께 떨어졌고
ontology payload 때문에 거의 같은 호출 수에서 token은 크게 늘었다.

## 실패 위치

Scenario 10의 최종 active Fact 9개에는 다음 정답에 필요한 Fact가 없었다.

- Jack의 steering-wheel heating level 2
- Jack의 feet/window airflow
- Brenda의 navigation volume 90
- Christine의 work display language English

이들은 candidate event에도 없으므로 semantic validator나 linker가 거절·병합한
것이 아니라 그 이전의 matcher/LLM candidate 생성 단계에서 누락됐다.

다른 대표 퇴행은 다음과 같다.

- Scenario 6 task03·08: 정답 Fact와 값은 찾았지만 `zone=driver` 대 Gold `all`
- Scenario 6 task06: `side=both` 대 Gold `left`
- Scenario 7 task03·04·06·07: 필요한 lighting/temperature Fact retrieval 실패
- Scenario 9 task04: child-lock 값은 맞지만 `window=rear` 대 Gold `all`

따라서 canonical 저장은 일부 argument를 정확하게 만들었지만, ontology Top-k를
Fact 생성 앞단의 제약으로 사용한 현재 방식은 high-recall 특성을 훼손했다.

## 판단과 다음 방향

현재 profile은 기존 Recursive-assisted Ours를 대체하지 않는다. 유지할 가치가
있는 부분은 evidence speaker 정규화와 canonical predicate 간 잘못된 UPDATE
방지다.

ESM `-0.04`가 통계적으로 확정적인 차이인지는 고정 Memory cache를 재사용한
Agent 반복 실행으로 확인할 수 있다. 그러나 candidate·active Fact·recall 감소와
token 증가는 이미 같은 방향이므로, 현재 구성에 개선 증거가 없다는 1차 판단은
바뀌지 않는다.

다음 실험은 ontology로 candidate 생성을 제한하는 방식을 완화하고,
기존 high-recall free-form candidate를 먼저 만든 뒤 ontology를 후단
정규화·검증에만 사용하는 구성이 적절하다. 이 방식은 현재 확인된 canonical
저장의 장점을 유지하면서 candidate 44건 감소를 피할 수 있는지 검증해야 한다.

이 후속 방식은 post-normalized v1으로 구현됐고 Scenario 6 smoke에서 candidate
17개를 모두 보존하며 ESM 0.80을 기록했다. 단일 run 결과이므로 전체 판단은
[Post-normalized Scenario 6 결과](post-normalized-recursive-assisted-s6-results.md)
및 후속 Scenario 6–10 평가로 분리한다.

## Artifact

- Ontology-assisted: `ubuntu/evaluation/vehiclemembench/a92d4bdb-d7e5-49f8-860f-07a77131e0e6`
- Recursive-assisted baseline: `ubuntu/evaluation/vehiclemembench/24dcf1f2-40af-48da-b893-420e4404525c`
- Ours baseline: `ubuntu/evaluation/vehiclemembench/ffdd8096-c990-46f6-ab9f-0644f61df56a`
