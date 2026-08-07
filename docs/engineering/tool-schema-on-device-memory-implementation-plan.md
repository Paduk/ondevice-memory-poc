# Fact-First Incremental Memory PoC 구현 계획

상태: `R0–R5 및 Scenario 6–10 holdout 완료 — 500-task 확대 보류`

최종 검토: 2026-07-29

관련 문서:

- [방법론](tool-schema-on-device-memory-methodology.md)
- [기존 개선·평가 기록](tool-schema-patch-improvement-priority-plan.md)
- [R5 10% 평가 결과](tool-schema-on-device-memory-r5-results.md)
- [Scenario 6–10 holdout](tool-schema-on-device-memory-holdout-s6-s10-results.md)

## 1. 목표

현재의 `Tool schema-first extraction`을 다음 구조로 변경한다.

```text
대화
  → 사실 기반 고재현율 Memory 후보 추출
  → 기존 record와 연결해 ADD·UPDATE·MERGE·DELETE
  → 유연한 의미 검증 + 엄격한 안전 검증
  → SQLite 저장

질문
  → entity·role·condition 해석
  → Memory-first retrieval + Tool schema routing
  → top-k Memory를 Tool argument로 late binding
  → Agent 실행
```

OpenAI Cloud LLM으로 PoC를 먼저 검증하며, Local SLM과 Android 이식은
동일 contract가 안정화된 뒤 진행한다.

## 2. 현재 기준선

VehicleMemBench 5개 Scenario, 50/500 task 결과:

| 방법 | ESM | State F1 | Tool F1 | Argument exact | Recall@k |
| --- | ---: | ---: | ---: | ---: | ---: |
| Structured Hybrid | 0.56 | 0.873 | 0.627 | 0.50 | 0.47 |
| Ours v3 | 0.30 | 0.460 | 0.287 | 0.22 | 0.11 |

Ours v3 Memory 생성:

- Cloud Memory 호출 410회, input 4,218,125 token
- proposal 58개, 적용 37개, 거절 21개
- 중복 retry를 합친 거절 고유 사실 약 13개
- 성공한 UPDATE·MERGE·DELETE 0개
- 50개 task 중 36개에서 선택 Memory 0개

해결할 주요 원인은 다음과 같다.

1. 111개 Tool ontology를 추출 prompt에 강제해 Fact 후보 coverage가 낮음
2. literal evidence와 좁은 UPDATE marker로 Validation rejection이 큼
3. Tool-first hard routing 때문에 정답 Memory가 검색 후보에 들어오지 않음
4. 인물·driver/passenger·coreference·시간 조건 해석이 약함
5. 영속 Memory 값과 요청별 runtime argument가 혼합됨

## 3. 구현 원칙

- **추출은 high recall**: Tool argument 완성 여부로 Fact를 버리지 않는다.
- **저장은 versioned patch**: 동일 Fact는 UPDATE, 없으면 ADD한다.
- **안전 규칙은 유지**: evidence·권한·PII·transaction은 엄격하게 검사한다.
- **의미 검증은 유연하게**: 숫자·boolean·alias·변경 표현을 정규화한다.
- **Tool binding은 late binding**: 검색된 Fact를 현재 요청의 역할과 결합한다.
- **Routing은 recall 우선**: Tool route를 Memory 검색의 hard filter로 쓰지 않는다.
- **Gold 정보는 진단 전용**: 생성·검색·Agent prompt에 정답을 넣지 않는다.

## 4. Phase R0 — Offline 재현 기반

목표: 새 Cloud Memory 호출 없이 변경 효과를 반복 진단할 기반을 만든다.

구현:

- 기존 5개 Scenario의 58개 proposal·evidence를 read-only replay
- Validation rule별 통과·거절과 중복된 retry proposal 분리
- 50개 query의 route, condition, selected Memory를 정적 재실행
- 기존 artifact와 동일한 task ID별 failure diff 생성
- 기존 SQLite cache는 수정하지 않고 파생 DB에만 결과 저장

구현 파일:

```text
vehicle_bench/diagnostics.py
vehicle_bench/memory.py
validation.py
storage.py
cli.py
tests/test_vehicle_memory.py
```

완료 조건:

- Memory LLM 호출 없이 58개 proposal의 재검증 결과를 생성한다.
- Agent 호출 없이 route·candidate·Recall@k 변화를 비교한다.
- retry 중복과 고유 Fact 수를 별도 보고한다.

구현·실행 결과:

- 5개 Scenario SQLite cache를 read-only로 열고 모델 호출 0회로 실행
- raw proposal 58개, retry 중복 제거 후 고유 proposal 50개 재현
- 고유 proposal: 기존 적용 37개, 거절 13개
- 범용 숫자·boolean·UPDATE 정규화 preview:
  기존 적용 37개 + 추가 통과 예상 12개 + review 1개
- 50개 task 중 선택 Memory 0개 36건, Tool mis-routing 22건
- 정답 값과 일치하는 active Memory가 있는데 route가 놓친 task 10건
- ambiguous route 15건, 실제 classifier 사용 0건
- 기존 cache와 cases artifact는 수정하지 않고 파생 artifact만 생성

Artifact:
`ubuntu/evaluation/vehiclemembench/403cbce2-5351-4255-9723-c63baf4098e7/r0-offline`

주의: 완화 결과는 저장된 proposal의 deterministic preview다. 아직 proposal을
DB에 적용하거나 Agent E2E 성능을 다시 측정한 결과가 아니다.

## 5. Phase R1 — Fact-first schema와 storage

목표: Tool argument와 독립적인 작은 Fact record를 도입한다.

구현:

- `FactMemoryRecord`, `FactMemoryIdentity`, `FactMemoryPatch` contract
- `entity_id`, `predicate`, `value`, `identity_conditions`, `applicability`
- 선택적 `capability_hints`; exact Tool name과 runtime role은 필수 아님
- active, superseded, merged, deleted, review 상태와 evidence lineage
- 기존 Hybrid와 v3 Patch table을 변경하지 않는 별도 migration
- profile별 DB·cache 격리

예시:

```json
{
  "user_id": "default_user",
  "entity_id": "patricia",
  "predicate": "headrest_height",
  "value": 44,
  "identity_conditions": {},
  "applicability": {},
  "capability_hints": ["seat"]
}
```

구현 파일:

```text
models.py
storage.py
fact_memory.py
fact_memory_schema.py
migrations/014_fact_first_memory.sql
tests/test_fact_memory.py
```

완료 조건:

- `Patricia의 headrest=44`를 저장하면서 `seat=driver`를 요구하지 않는다.
- 같은 Fact의 값 변경은 새 active record가 아니라 새 version이 된다.
- 기존 Summary·Structured Hybrid·schema-patch profile이 그대로 동작한다.

구현 결과:

- `FactMemoryIdentity`, `FactMemoryRecord`, `FactMemoryPatch`와 JSON parser 추가
- Fact identity를 `user + entity + predicate + identity_conditions +
  applicability`로 고정하고 값·Tool argument는 key에서 제외
- 기존 테이블을 수정하지 않는 `fact_memory_*` 전용 migration과 repository 추가
- ADD·UPDATE·MERGE·DELETE를 단일 transaction으로 적용하고 idempotency 보장
- UPDATE는 이전 값을 보존한 immutable version, DELETE는 tombstone version 생성
- evidence message·원문 quote·span과 record status event를 영속 추적
- `Patricia.headrest_height=44` 저장 및 `44→48` 버전 변경 테스트 통과
- 전체 Ruff 및 pytest 회귀 테스트 통과

R1은 저장·Patch 기반만 구현한다. Cloud high-recall 추출과 기존 record 연결은
R2, 완화 Validation은 R3, retrieval·Tool late binding은 R4에서 연결한다.

## 6. Phase R2 — High-recall 추출과 Patch linker

목표: Hybrid 수준으로 Fact 후보를 폭넓게 추출하면서 Patch 방식을 유지한다.

구현:

- Memory LLM prompt에서 111개 전체 Tool ontology 제거
- entity, predicate, value, conditions, evidence 중심의 compact output
- 최대 32 turns 또는 4,096 source token batch 유지
- batch의 관련 entity·predicate active record만 모델에 제공
- canonical entity·predicate·identity condition 기반 candidate linker
- exact key가 없을 때 alias·semantic similarity로 기존 record 후보 생성
- linker 결과에 따라 ADD 또는 UPDATE를 결정
- 동등한 중복은 MERGE, 명시적 철회는 DELETE 후보로 변환
- 동일 validation 오류를 재호출하지 않고 review/no-op으로 종결

구현 파일:

```text
contracts.py
providers.py
background_memory.py
memory_patch.py
storage.py
vehicle_bench/memory.py
tests/test_background_memory.py
tests/test_providers.py
```

완료 조건:

- extraction model이 Tool schema를 몰라도 지속적 차량 preference를 반환한다.
- ADD뿐 아니라 UPDATE가 합성 correction fixture에서 실제 적용된다.
- 후보가 없는 batch는 정상 empty 결과로 종료된다.
- Memory 생성 call·token을 기존 v3와 분리 기록한다.

구현 결과:

- `FactMemoryModel`과 `FactMemoryExtractionResponse` Cloud contract 추가
- OpenAI structured output은 entity·predicate·value·조건·근거만 반환하며
  Tool ontology와 111개 Tool schema를 입력하지 않음
- active Fact 전체를 보내지 않고 batch lexical match와 최근 record를 합친
  최대 24개 linking context만 제공
- source message 순서대로 candidate를 연결하여 ADD·UPDATE·MERGE·DELETE 적용
- exact canonical key가 없으면 부분 이름 alias와 predicate token similarity로
  기존 record 후보를 찾고, 복수 충돌은 자동 적용하지 않고 review 처리
- 동일 값은 NOOP, 값 변경은 immutable UPDATE version으로 기록
- 32 turns 또는 4,096 source token 한도의 idle-time batch worker 연결
- candidate 하나의 구조·근거 오류는 job retry 대신 review로 격리
- Fact candidate decision, model usage, latency, 비용을 기존 schema-patch와
  구분된 trace와 metrics로 저장
- 일반 Ubuntu runtime feature flag와 VehicleMem `fact_patch` cache build 경로 추가
- 합성 correction에서 `ADD → UPDATE → NOOP` 및 alias UPDATE 통합 테스트 통과

R2에서는 Fact 생성·연결까지만 수행한다. review/rejection의 의미 검증 완화는
R3, 질문 시 retrieval과 Tool late binding은 R4 범위다.

## 7. Phase R3 — Validation 재설계

목표: 안전성은 유지하면서 literal 표현 차이로 Fact를 버리지 않는다.

엄격하게 유지:

- JSON 구조와 필수 필드
- 고정 user·scope와 cross-session target
- evidence quote의 실제 원문 존재
- PII·secret 정책
- transaction, version, tombstone, idempotency

유연하게 변경:

- 숫자 단어와 숫자: `three ↔ 3`
- boolean·행동 표현: `turn on ↔ true`
- enum alias: `outside air ↔ outside`
- UPDATE 표현: `set it to`, `turning up`, `I prefer`, `back to`
- entity alias와 대소문자·부분 이름

추가 정책:

- runtime-bound field는 evidence entailment 대상에서 제외
- 불확실한 semantic entailment는 `review`, 구조 오류는 `rejected`
- valid proposal 적용과 invalid proposal 격리
- retry proposal dedup 후 한 번만 결과 기록

구현 파일:

```text
validation.py
normalization.py
memory_patch.py
background_memory.py
storage.py
tests/test_memory_patch.py
tests/test_fact_memory.py
```

완료 조건:

- 기존 거절 고유 Fact 약 13개를 offline replay해 생존·review·거절로 분류한다.
- `seat=Gary` 같은 잘못된 runtime binding은 저장 값에서 제거한다.
- 완전 Validation skip 없이 rejection 원인을 설명 가능한 code로 남긴다.

구현 결과:

- 숫자 단어·numeric string, boolean 행동 표현, 공조 enum alias를 canonical
  value로 정규화한 뒤 evidence와 비교
- `seat`, `driver`, `person`, `zone` 등 요청 시 결정할 runtime field를
  저장 value·identity condition·applicability에서 제거
- 불확실한 값 entailment, 낮은 confidence, 모호한 UPDATE·DELETE 의도는
  자동 폐기하지 않고 `review`로 격리
- 원문에 유일하게 존재하는 quote의 잘못된 span은 보정하고 correction
  trace 저장
- 존재하지 않는 근거, 다른 session/batch 근거, assistant 근거,
  PII·secret, 잘못된 구조는 candidate 단위 `rejected`
- 한 candidate의 실패가 같은 batch의 valid candidate 적용이나 worker
  완료를 막지 않음
- validation disposition·code·span correction·제외 runtime field를 SQLite
  trace에 저장하고 metrics에서 code별 집계

기존 5개 Scenario의 고유 거절 proposal 13개를 모델 호출 없이 replay한
결과는 `would_accept=12`, `would_review=1`, `still_reject=0`이다. 이는 기존
schema-patch proposal에 대한 deterministic 생존 분석이며, Fact-first
Memory를 새로 생성하거나 E2E Agent 성능을 측정한 결과는 아니다.

Artifact:
`ubuntu/evaluation/vehiclemembench/403cbce2-5351-4255-9723-c63baf4098e7/r3-validation-replay`

## 7.1 Phase R3.1 — 불확실 후보 Batch Semantic Validation

목표: 표현별 규칙을 계속 추가하지 않고 deterministic Validation이 확정하지
못한 후보만 Cloud LLM으로 의미 검증한다.

구현:

- extraction batch에서 `review` 후보와 변경 의도가 불확실한 후보만 수집
- 같은 최대 32 turns/4,096 tokens worker 주기에서 semantic call 최대 1회
- candidate, exact evidence, proposed operation, 관련 active record만 제공
- structured output `ACCEPT·REVIEW·REJECT`, confidence, evidence relation,
  reason을 sequence index별로 반환
- confidence 0.7 미만의 ACCEPT·REJECT는 자동 적용하지 않고 `review`
- hard rejection은 semantic model 입력·override 대상에서 제외
- semantic failure·timeout·불완전 응답은 batch 전체를 재추출하지 않고
  해당 후보를 `review`로 유지
- 승인 후보는 source turn 순서대로 기존 linker와 versioned Patch에 적용
- 별도 `fact_memory_semantic_validation` model-call trace와 token·cost 집계

구현 결과:

- deterministic UPDATE marker가 없는 paraphrase와 literal value가 직접
  등장하지 않는 후보를 한 번의 batch semantic call로 승인 가능
- semantic 승인·거절·보류와 원래 candidate confidence를 validation trace에
  기록하고 승인 confidence를 record confidence로 사용
- 존재하지 않는 quote·다른 session/batch·assistant evidence·PII·구조 오류는
  semantic model로 보내지 않음
- semantic provider 실패 fixture에서 job은 재호출 없이 completed되고 후보는
  `review`로 보존
- 기존 Fact worker는 semantic 후보가 없으면 추가 모델 호출 0회

## 8. Phase R4 — Memory-first retrieval과 Late Tool Binding

목표: 질문에 Tool 이름이 드러나지 않아도 관련 Fact를 찾고 실행한다.

구현:

1. query에서 entity·speaker·driver/passenger·시간·날씨·상황 추출
2. entity partition의 Fact를 BM25·embedding으로 검색
3. query→Memory와 query→Tool schema 후보의 합집합 생성
4. condition compatibility와 semantic score로 공동 rerank
5. empty·low-confidence·ambiguous일 때 Cloud classifier/reranker fallback
6. 선택 Fact와 현재 역할을 결합해 runtime Tool argument 생성
7. JSON schema와 VehicleWorld semantic enum 검증
8. Agent에게 top-k Fact와 선택 Tool만 제공

예시:

```text
Memory: Patricia.headrest_height = 44
Query role: Patricia = driver
Late binding: seat_set_headrest_height(seat="driver", value=44)
```

Tool-first route는 후보 점수로 유지하되 hard SQL partition으로 사용하지 않는다.
Top-k와 token budget은 계속 적용한다.

구현 파일:

```text
memory_router.py
tool_memory_execution.py
context.py
agent.py
application.py
vehicle_bench/runner.py
storage.py
tests/test_tool_memory_retrieval.py
tests/test_vehicle_bench.py
```

완료 조건:

- `I'm driving today`처럼 암시적인 요청도 entity Memory에서 후보를 찾는다.
- 여러 사람이 함께 등장해도 현재 요청 주체의 Fact를 우선한다.
- `after dusk`와 `8:00 PM`, `her call`과 대상 인물을 연결한다.
- route가 틀려도 Memory 후보가 완전히 제거되지 않는다.
- runtime argument가 simulator의 실제 허용 값까지 통과한다.

구현 결과:

- migration `017`에 Fact embedding cache, retrieval run, 후보별 BM25·embedding·
  entity·condition·route score와 선택 사유 trace를 분리 저장
- 동일 user/session의 active Fact 전체를 검색 시작점으로 사용하며 Tool route를
  SQL hard filter로 사용하지 않음
- 인물 alias, 요청자, driver/passenger, 12시간제 시각, dusk/night, 날씨·상황을
  요청별 `FactQueryContext`로 해석
- BM25·embedding 점수에 entity·condition compatibility와 Tool route 가산점을
  결합하고 top-k·token budget 적용
- 선택 Fact의 영속 값과 현재 역할을 결합해 `seat`, `light`, `zone`, `side`를
  요청 시 생성하고 JSON Schema 및 runtime enum allowlist로 검증
- query→Tool route와 Fact→Tool late-binding 결과를 합쳐 Agent Tool 후보 구성;
  route가 비어도 선택 Fact와 late-bound Tool은 유지
- 일반 Ubuntu Agent는 Fact Memory 기능 사용 시 선택 Tool schema만 모델에
  제공하고, VehicleMem은 선택 Tool을 dynamic registry에 preload
- VehicleMem 프로필 `cloud_fact_patch`, 진단용 `oracle_tool_fact_patch` 추가
- 실제 VehicleWorld fixture에서 Fact 검색 → Tool preload → argument 실행 →
  Exact State Match 및 argument exact match 성공
- 별도 foreground classifier 호출은 추론 비용을 늘리므로 PoC 기본값에서는
  추가하지 않음. 모호한 top-k는 기존 Cloud Agent 호출이 최종 rerank하며,
  Tool router의 optional classifier contract와 dynamic discovery fallback은 유지

검증 fixture:

- Tool route가 empty여도 `Patricia.headrest_height=44` Fact 검색
- 여러 인물 중 실제 요청자 및 driver/passenger 역할 우선순위
- `after dusk`와 `8:00 PM` 조건 호환
- `Patricia=driver`를 `seat="driver"`로 late binding
- `cloud_fact_patch` VehicleWorld E2E 실행

## 9. Phase R5 — Ablation과 10% 재평가

상태: `완료`

결과 요약:

- Fact-first 50-task: ESM 0.52, State F1 0.698, Tool F1 0.556,
  argument exact 0.48, Recall@k 0.45
- Fact candidate 76개 중 68개 적용, acceptance 89.5%
- Gold Tool + Fact ESM 0.56으로 routing 영향 분리
- Hybrid 대비 Agent input 19.5% 감소, Memory context 49.6% 증가
- 정확도 최소 gate는 통과했으나 context 및 Hybrid 성능 격차로 500-task 보류
- 상세 수치와 실패 분석은
  [R5 평가 결과](tool-schema-on-device-memory-r5-results.md)에 기록

먼저 기존 cache로 offline 실험을 수행한다.

| Memory 구성 | Routing | 목적 |
| --- | --- | --- |
| v3 Validation | 현재 Routing | 기존 baseline 재현 |
| v3 Validation | Gold Tool | mis-routing 영향 |
| 완화 Validation replay | 현재 Routing | rejection 영향 |
| 완화 Validation replay | Gold Tool | 두 병목 제거 상한 |

그다음 Fact-first Memory를 새로 생성해 동일 5개 Scenario·50 task를 평가한다.

비교:

- No Memory
- Structured Hybrid
- Ours v3 schema-first
- Ours v4 fact-first
- v4 Gold Tool diagnostic
- v4 Gold Memory diagnostic

주요 gate:

- 고유 candidate acceptance 80% 이상
- 합성 correction에서 UPDATE 성공률 100%
- Retrieval Recall@k 0.11 → 최소 0.35
- 50-task ESM 0.30 → 최소 0.50
- Hybrid 대비 Agent input·context token 이점 유지
- unsupported·conflict Memory 비율을 acceptance와 함께 보고

R5 gate를 통과한 뒤에만 50개 Scenario·500 task로 확대한다.

## 10. 이후 단계

### R6 — Local SLM

- `FactMemoryModel`과 `PatchMemoryModel`의 OpenAI-compatible local provider
- Cloud와 동일 history·cache·scorer로 품질·속도·노출량 비교
- malformed output 복구, quantization, context 길이 검증

### R7 — Android

- 동일 schema·Patch engine을 Android SQLite에 연결
- WorkManager의 charging·idle·thermal 조건
- foreground Tool 요청과 background Memory worker 격리
- 강제 종료·재부팅 복구, RAM·배터리·발열 측정

## 11. 수행 순서

```text
R0 offline replay
  → R1 Fact schema
  → R2 high-recall extraction/linker
  → R3 Validation
  → R4 retrieval/late binding
  → R5 10% ablation
  → R6 Local SLM
  → R7 Android
```

각 Phase는 기존 50개 task의 동일 task ID와 frozen artifact로 회귀 검증한다.
새 Cloud Memory 생성 전에는 예상 호출 수와 비용을 먼저 산출한다.
