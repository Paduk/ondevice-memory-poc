# Schema-informed Fact ontology v1 계획

상태: `v1 ontology와 최소 Schema-informed Ours 구현 완료, 평가 전`

## 목표

VehicleMemBench의 공개 Tool interface만 사용해 111개 Tool을 작은 범용
Fact ontology로 압축한다. QA, History, Gold Memory, reference Tool call은
ontology 생성에 사용하지 않는다. 기존 Tool-independent Ours는 변경하지
않고 이후 별도 Schema-informed Ours profile에서 사용한다.

## 작업 순서

1. `functions_schema.json`의 Tool 이름, 설명, parameter, 타입과 제약을
   손실 없이 inventory로 읽는다.
2. 각 Tool을 `capability`, `target`, `operation`, argument별
   `value/selector/content` 역할로 매핑한다.
3. 공통 의미는 통합하되 target과 값 타입·범위가 다른 항목은 분리한다.
4. target은 capability별 허용 enum으로 동결하고 `unspecified`와
   `other` fallback을 둔다.
5. 111개 Tool coverage, argument coverage, 중복 binding, 타입·제약 충돌과
   Tool 역매핑을 자동 검증한다.
6. ontology 데이터, 검수 보고서와 단위 테스트를 함께 동결한다.
7. 각 Fact batch를 ontology에 로컬 hybrid 검색하고 최대 8개 canonical
   capability만 기존 1회 Fact 추출 호출에 제공한다.
8. 모든 user message의 판정 여부와 predicate·target·value를 로컬에서
   검증하는 별도 `cloud_schema_informed_fact_patch` profile로 연결한다.

## v1 산출물

- versioned canonical ontology와 Tool binding
- Tool·argument coverage 및 모호성 검수 결과
- cache 재현성을 위한 source schema hash와 ontology hash
- 기존 Ours에 영향을 주지 않는 loader·validator와 테스트
- 전체 ontology나 Tool schema를 prompt에 넣지 않는 소형 matcher
- 기존 Ours cache와 분리된 Schema-informed Fact profile

## 완료 기준

- Tool coverage `111/111`
- 모든 Tool argument에 역할 지정
- 허용되지 않은 target은 자동 저장하지 않음
- 명시되지 않은 target을 임의 추론하지 않음
- ontology에서 원래 Tool·argument schema로 역추적 가능
- Ruff 및 관련 pytest 통과

구현 결과와 남은 범위는
[Schema-informed Fact ontology v1 검수 결과](schema-informed-fact-ontology-v1.md)에
기록한다.
