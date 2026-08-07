# Schema-informed Recursive-assisted Ours Scenario 6 결과

상태: `구현 및 Scenario 6 E2E smoke 완료`

## 목적과 구성

새 profile `cloud_schema_informed_recursive_assisted_fact_patch`는 다음 순서로
동작한다.

1. History를 ontology-aware Fact extractor로 처리한다.
2. 같은 History의 Recursive Summary를 recall checklist로만 사용해 누락 Fact를
   한 번 더 찾는다.
3. 두 pass 모두 동일한 ontology validation과 evidence 기반 speaker 정규화를
   거친 뒤 하나의 Fact DB에 저장한다.

Summary 문장 자체는 evidence가 아니다. 보조 pass도 반드시 현재 History batch의
원문 `message_id`를 근거로 제시해야 저장할 수 있다. 기존
`cloud_recursive_assisted_fact_patch`는 변경하지 않고 별도 profile과 cache로
분리했다.

## 범용 안전장치

- canonical ontology predicate가 서로 다르면 linker가 UPDATE로 병합하지 않는다.
- 이 exact-predicate 정책은 schema-informed profile에만 적용한다. 기존 자유 형식
  Ours의 predicate alias 동작은 유지한다.
- base와 assisted pass 모두 evidence line의 실제 speaker를 canonical entity로
  사용한다.
- ontology, entity-resolution, linking-policy, Recursive Summary hash를 cache
  config에 포함한다.

이로써 같은 Michael의 `climate_circulation=outside`와
`climate_fan_speed=3`이 별도 Fact로 남았다. Jonathan의
`climate_fan_speed=10`도 별도 entity로 보존됐다.

## Scenario 6 결과

조건은 10개 Quiz, Quiz model `gpt-5.6-terra`, Memory model `gpt-5.6-luna`,
embedding `text-embedding-3-small`이다.

| 지표 | 결과 |
| --- | ---: |
| Exact State Match | **5/10 (0.50)** |
| State F1 | 0.687 |
| Tool F1 | 0.467 |
| Argument exact match | 0.400 |
| Retrieval recall@k | 0.600 |
| 최종 candidate / active Fact | 15 / 13 |

직전 schema-informed speaker profile의 ESM 0.30보다 task00과 task01이 추가로
성공해 0.20 올랐다. 다만 Scenario 6의 기존 Ours 기준 0.50과 같은 수준이며,
이 한 시나리오만으로 전체 Scenario 6–10 개선을 결론 내릴 수는 없다.

## Summary 보조 pass의 직접 효과

보조 pass가 새로 저장한 active Fact는 다음 세 개다.

| Fact | E2E 영향 |
| --- | --- |
| Samuel, music volume 15, presentation focus | task00 정답 복구 |
| Samuel, display brightness 90, sunlight glare | task01 정답 복구 |
| Samuel, climate temperature 26, fever/chills | task07 query에는 노이즈로 작용 |

Jonathan의 mirror auto-reverse true 후보도 만들었지만 semantic REVIEW로 남아
저장되지 않았다. 즉 Summary recall은 실제 생성 누락 두 건을 회수했지만,
관련성이 낮은 Fact도 한 건 추가했다.

## 남은 실패

| Task | 관찰된 병목 |
| --- | --- |
| 03 | 정답 circulation Fact를 rank 1로 찾고 값도 맞췄으나 `zone=driver` 대 Gold `all` |
| 05 | media quality 480p 및 복합 실행 정보가 Fact에 없음 |
| 06 | mirror 후보가 REVIEW로 남고 실행 시 `side=both` 대 Gold `left` |
| 07 | 필요한 temperature 21 Fact가 없고 temperature 26 Fact가 retrieval noise가 됨 |
| 08 | 정답 fan speed 10 Fact를 rank 1로 찾고 값도 맞췄으나 `zone=driver` 대 Gold `all` |

가장 선명한 다음 병목은 Fact 생성이 아니라 late binding이다. task03과 task08은
Memory retrieval까지 성공했으므로, schema에 없는 zone을 현재 requester 역할로
자동 채우는 규칙을 범용적으로 검토해야 한다.

## 호출량

| 단계 | 생성 호출 | Input tokens | Output tokens |
| --- | ---: | ---: | ---: |
| Fact base + assisted | 170 | 941,824 | 16,614 |
| Fact semantic validation | 2 | 4,882 | 302 |
| Recursive Summary | 47 | 121,849 | 4,226 |
| 합계 | **219** | **1,068,555** | **21,142** |

Fact extraction은 85회에서 170회로 두 배가 됐다. Scenario 6에서 ESM 2건을
회수했지만 호출 비용 증가가 크므로, 전체 평가 전에 assisted pass를
누락 가능성이 높은 batch에만 제한하는 최적화 후보를 남긴다.

## 판단

연결은 기능적으로 문제없고, Summary로 누락 Fact를 회수한다는 가설도 두 건에서
확인됐다. exact-predicate와 speaker guard도 의도대로 동작했다. 하지만 현재
결과는 기존 Ours를 넘지 못했고 비용은 크게 늘었다. 다음 순서는 Scenario 6의
zone/side late-binding 오류를 정답별 예외가 아닌 Tool argument 정책으로 해결할
수 있는지 먼저 검증한 뒤, 동일 설정을 Scenario 6–10으로 확대하는 것이다.

## Artifact

- E2E: `ubuntu/evaluation/vehiclemembench/b6912182-0c9e-4cba-8f3b-da310af224a7`
- Base Fact cache: `ubuntu/evaluation/vehiclemembench-memory/d745b359794da154/scenario-06/9a3312e90bc5dd0c`
- Assisted Fact cache: `ubuntu/evaluation/vehiclemembench-memory/d745b359794da154/scenario-06/d17a8bf3b3d21e99`
