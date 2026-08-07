# Schema-informed Fact ontology v1 검수 결과

상태: `ontology·matcher·Fact extractor profile 구현 및 Scenario 6 smoke 완료`

기준 입력: VehicleMemBench `evaluation/functions_schema.json`

## 결과

| 항목 | 결과 |
| --- | ---: |
| 원본 Tool | 111 |
| 원본 argument | 155 |
| canonical capability | 64 |
| concrete target | 34 |
| value argument | 107 |
| selector argument | 42 |
| content argument | 6 |
| 설명에서 복구한 range·enum 제약 | 50 |
| 누락 Tool·argument | 0 |
| 같은 capability·target 내 value type 충돌 | 0 |

- Source schema SHA-256:
  `e86e5ca5203a14b82064ed563f3baf67c577895915578da42450a48602cb614c`
- Ontology SHA-256:
  `702d797805809b906f8b612549e78c0995c7bc70a543f4b177c4691eb7276eb1`

ontology 입력은 Tool 이름, 설명과 parameter schema뿐이다. QA, History,
Gold Memory와 reference Tool call은 읽지 않았다.

## 주요 통합

- `audio.volume`: music, navigation, radio, video
- `display.brightness`: center display, HUD, instrument panel, overhead screen
- `display.language`·`display.time_format`: 세 display 계열
- `closure.open`·`closure.open_degree`·`closure.locked`: door, fuel port,
  trunk, sunroof, sunshade, window의 호환 기능
- `heating.enabled`·`heating.level`: mirror, seat, steering wheel
- `lighting.enabled`: 실제 조명 종류는 target으로 유지
- `system.power`: `switch`, `is_on` 중 실제 전원 의미만 통합

`auto_brightness`와 brightness 값, `open`과 open degree, feature enable과
system power, heating enable과 level은 서로 다른 capability로 유지했다.

## Target 정책

target enum은 capability별로 다르다. 명시된 target은 해당 enum에서
선택하고, 현재 설치된 호환 target이 하나뿐이면 그것을 자동 선택한다.
호환 target이 여러 개이고 원문이 모호하면 `unspecified`, 원문에 명시된
새 target이 ontology에 없으면 `other`로 둔다. 여러 target에 적용되는
문장은 target별 atomic Fact로 분리한다.

## 최소 extractor 연결

- profile: `cloud_schema_informed_fact_patch`
- 저장 batch의 user 문장을 로컬 lexical+embedding으로 검색해 문장당 상위
  2개, batch당 최대 8개 capability만 LLM에 제공한다.
- 기존 Ours와 동일하게 Fact 추출 LLM은 batch당 한 번만 호출한다.
- LLM은 `candidate`와 `uncertain`인 message만 sparse assessment로 반환한다.
  나머지는 로컬에서 `no_durable_vehicle_fact`로 계산해 생성 누락 관측은
  유지하면서 대량의 부정 출력은 없앴다.
- 반환된 Fact는 로컬에서 retrieved predicate, target fallback, target별 값
  schema를 검증하며, 원문 evidence는 기존 Fact와 동일하게 보존한다.
- candidate evidence가 가리키는 History speaker를 canonical entity로 사용해
  서로 다른 사람의 같은 setting이 덮어써지지 않게 한다.
- 기존 `cloud_fact_patch` 코드와 cache는 변경하지 않고 별도 profile/cache로
  분리했다.

## 남은 범위

- 원본 JSON Schema는 `seat`, `zone`, `side` 등의 selector enum을 제공하지
  않는다. v1은 이를 임의로 만들지 않고 string selector로 유지한다.
- 설명에 명시된 단순 숫자 범위와 slash·quoted enum만 보수적으로 복구한다.
- selector 자체의 ontology enum 확장, retry와 별도 DB table은 MVP에서
  추가하지 않았다.
- Scenario 6 생성형 LLM smoke에서는 최종 active 정답 Fact가 기존 검수
  2/10에서 4/10으로 늘었지만, 6–10 E2E 전이므로 최종 정답률 개선 여부는
  아직 결론 내리지 않는다.

## Scenario 6 matcher-only 결과

생성형 LLM 없이 `text-embedding-3-small` 256차원과 실제 Fact batch 설정
(`32 turns / 4096 tokens`, 문장별 Top-2, batch Top-8)으로 먼저 평가했다.

| 지표 | 결과 |
| --- | ---: |
| evidence 문장 Top-1 | 8/10 |
| evidence 문장 Top-2 | 10/10 |
| 실제 batch Top-8 | 8/10 |
| Top-2 후보의 target binding 포함 | 10/10 |
| 생성형 LLM 호출 | 0 |

개별 evidence에서는 10건 모두 정답 capability를 찾았으므로 ontology의
의미 연결 가능성은 확인됐다. 다만 32개 문장의 후보를 batch당 8개로 다시
줄일 때 `media.quality`, `climate.temperature` 두 건이 무관한 문장의 높은
후보 점수에 밀렸다. 따라서 LLM 평가 전에 stopword성 lexical overlap과
문장별로 정규화된 semantic score의 batch 간 비교 문제를 먼저 줄이는 것이
필요하다. 이 결과는 matcher 단계만의 진단이며 Fact 생성·저장 또는 최종
정답률 개선을 뜻하지 않는다.

## 구현

- [v1 ontology 데이터](../../ubuntu/src/palmclaw_ubuntu/vehicle_bench/data/vehicle_fact_ontology_v1.json)
- [loader, matcher와 validator](../../ubuntu/src/palmclaw_ubuntu/vehicle_bench/vehicle_fact_ontology.py)
- [Schema-informed provider](../../ubuntu/src/palmclaw_ubuntu/providers.py)
- [profile과 prompt](../../ubuntu/src/palmclaw_ubuntu/vehicle_bench/memory.py)
- [ontology 단위 테스트](../../ubuntu/tests/test_vehicle_fact_ontology.py)
- [provider 단위 테스트](../../ubuntu/tests/test_providers.py)
- [matcher-only 평가기](../../ubuntu/src/palmclaw_ubuntu/vehicle_bench/ontology_evaluation.py)
- [Scenario 6 matcher label](../../ubuntu/src/palmclaw_ubuntu/evaluation_datasets/vehiclemembench_ontology_matcher_s6_v1.json)
- [Scenario 6 Fact-only smoke 결과](schema-informed-fact-s6-smoke-results.md)
