# A-MEM Baseline 구현 실행 계획

상태: `6회차 완료 · Cloud smoke 전`

작성일: `2026-08-06`

상위 설계 문서:
[A-MEM VehicleMemBench Baseline 구현 계획](amem-vehiclemembench-baseline-implementation-plan.md)

## 1. 문서 목적

이 문서는 `cloud_amem` baseline의 **코드 구현에만** 사용한다. A-MEM 방법론,
실험 가설, Cloud 비용과 Scenario 평가 기준은 상위 설계 문서를 source of truth로
삼고 여기에서 반복하지 않는다.

구현은 6회차로 나눈다.

| 회차 | 범위 | 종료 checkpoint |
| ---: | --- | --- |
| 1 | Domain model·DB·Repository | A-MEM 데이터를 안전하게 저장·조회 가능 |
| 2 | 순차 ingestion·evolution engine | Fake model로 graph snapshot 생성 가능 |
| 3 | OpenAI A-MEM provider | Mock client로 construction/evolution 계약 검증 |
| 4 | Query retrieval·graph expansion | 읽기 전용 Top-k+linked retrieval 가능 |
| 5 | VehicleMemBench profile 통합 | `cloud_amem` fake E2E 실행 가능 |
| 6 | 안정화·전체 회귀·문서화 | Cloud smoke 직전 구현 완료 |

6회차가 끝나기 전에는 실제 Cloud Memory 생성이나 VehicleMemBench 성능 평가를
실행하지 않는다. Cloud smoke와 Scenario 평가는 상위 문서의 Phase A4~A6에서
별도로 진행한다.

## 2. 공통 구현 원칙

모든 회차에 다음 규칙을 적용한다.

- 기존 Memory, Fact, Ours profile과 cache key를 변경하지 않는다.
- A-MEM은 VehicleMemBench 전용 독립 strategy로 먼저 구현한다.
- 공식 `agentic-memory` 패키지와 ChromaDB·SentenceTransformers를 의존성에
  추가하지 않는다.
- migration은 additive하게 추가하며 기존 table을 재해석하지 않는다.
- History 원문, timestamp, speaker와 source ID는 evolution으로 변경하지 않는다.
- LLM이 반환한 note ID는 항상 runtime이 제공한 candidate 집합과 대조한다.
- query-time retrieval은 note, link, version과 fingerprint를 변경하지 않는다.
- Gold Memory, Gold Tool call, task ID별 literal은 생성·검색 입력에 사용하지 않는다.
- 실제 OpenAI 호출 없이 모든 구현 회차를 검증할 수 있어야 한다.
- 각 회차는 targeted test, 전체 `pytest`, Ruff가 통과해야 완료로 표시한다.

공통 검증 명령:

```bash
cd ubuntu
pytest
ruff check src tests
```

한 회차에서 전체 test가 기존 사용자 변경 때문에 실패하면 A-MEM 관련 targeted
test 결과와 기존 실패임을 분리해 기록한다. A-MEM 회귀를 남긴 채 다음 회차로
넘어가지 않는다.

## 3. 진행 상태

| 회차 | 상태 | 구현 결과 | 검증 |
| ---: | --- | --- | --- |
| 1 | 완료 | Domain model·019 migration·Repository·storage test 구현 | targeted 8 passed, 전체 232 passed·1 skipped, Ruff 통과 |
| 2 | 완료 | Fake model·순차 ingestion/evolution engine·validation/resume 구현 | targeted 14 passed, 전체 242 passed·1 skipped, Ruff 통과 |
| 3 | 완료 | OpenAI construction/evolution provider·strict schema·mock 검증 구현 | targeted 49 passed, 전체 256 passed·1 skipped, Ruff 통과 |
| 4 | 완료 | Query Top-k·1-hop expansion·budget rendering·read-only trace 구현 | targeted 22 passed, 전체 263 passed·1 skipped·기존 1 failed, Ruff 통과 |
| 5 | 완료 | `cloud_amem` Vehicle build/cache/runner/suite·CLI·Fake E2E 통합 | targeted 47 passed·1 skipped, 전체 268 passed·1 skipped, Ruff 통과 |
| 6 | 완료 | failure/privacy/cache/accounting 안정화·README/설계 문서 완료 | targeted 111 passed·1 skipped, 전체 276 passed·1 skipped, coverage 84%, Ruff 통과 |

각 회차 완료 시 이 표와 해당 회차의 `구현 결과` 절만 갱신한다. 평가 수치는 이
문서에 넣지 않는다.

## 4. 1회차 — Domain model·DB·Repository

### 목표

A-MEM note, metadata version, embedding, link와 실행 trace를 표현하는 저장 계약을
먼저 고정한다. Provider, ingestion algorithm과 Vehicle profile은 구현하지 않는다.

### 변경 파일

새 파일:

```text
ubuntu/src/palmclaw_ubuntu/migrations/019_amem.sql
ubuntu/tests/test_amem.py
```

수정 파일:

```text
ubuntu/src/palmclaw_ubuntu/models.py
ubuntu/src/palmclaw_ubuntu/contracts.py
ubuntu/src/palmclaw_ubuntu/storage.py
ubuntu/tests/test_storage.py
```

구현 시작 시 새 migration이 이미 추가돼 있으면 충돌을 피하도록 다음 사용 가능한
번호로 바꾸고 이 문서에 실제 번호를 기록한다.

### 구현 항목

1. `models.py`에 다음 immutable dataclass를 추가한다.
   - `AMemNote`
   - `AMemNoteVersion`
   - `AMemLink`
   - `AMemNeighbor`
   - `AMemConstructionResponse`
   - `AMemNeighborUpdate`
   - `AMemEvolutionDecision`
   - `AMemRetrievalResult`
2. `contracts.py`에 `AMemModel` protocol을 추가한다.
   - `construct(entry)`
   - `evolve(new_note, neighbors)`
3. migration에 다음 table과 index를 추가한다.
   - `amem_notes`
   - `amem_note_versions`
   - `amem_note_embeddings`
   - `amem_links`
   - `amem_construction_runs`
   - `amem_evolution_events`
   - `amem_retrieval_runs`
   - `amem_retrieval_candidates`
4. `storage.py`에 최소 CRUD와 trace API를 추가한다.
   - source별 note insert/get/list
   - latest/all version 조회와 새 version insert
   - model·dimension·version별 embedding 저장/조회
   - normalized note pair link insert/list
   - construction/evolution/retrieval run begin/complete/fail
   - retrieval candidate 저장과 trace 조회
5. 기존 `clone_in_memory()`, `backup_to()`와 `recover_interrupted_execution()`이
   A-MEM table을 포함하도록 한다.

### Hard constraint

- `amem_notes(session_id, source_message_id)`는 unique다.
- note version은 `(note_id, version)`이 unique다.
- 최신 version은 deterministic하게 하나만 선택할 수 있어야 한다.
- link endpoint는 정규화해 같은 undirected pair를 중복 저장하지 않는다.
- self-link와 cross-session link는 application validation과 DB constraint 중
  가능한 양쪽에서 차단한다.
- embedding은 정확한 `note_version_id`, model과 dimension에 연결한다.

### Targeted test

```bash
cd ubuntu
pytest tests/test_storage.py tests/test_amem.py
ruff check src/palmclaw_ubuntu/models.py \
  src/palmclaw_ubuntu/contracts.py \
  src/palmclaw_ubuntu/storage.py \
  tests/test_storage.py tests/test_amem.py
```

### 완료 조건

- 새 DB와 기존 migration DB를 모두 열 수 있다.
- insert 후 close/reopen, clone과 backup에서 동일 note graph가 조회된다.
- duplicate source, duplicate link, self/cross-session link가 거절된다.
- model/provider/Vehicle 코드 없이 저장 계층 test가 통과한다.

### 이번 회차에서 하지 않는 것

- `providers.py` 수정
- embedding similarity 계산
- ingestion/evolution orchestration
- Vehicle profile·CLI 추가

### 구현 결과

`완료 (2026-08-06)`

- `AMemNote`, metadata version, link, neighbor와 construction/evolution/retrieval
  응답 모델 및 `AMemModel` protocol을 추가했다.
- `019_amem.sql`에 note graph, embedding과 세 종류의 실행 trace table을
  additive migration으로 추가했다.
- source note·version·embedding·정규화된 undirected link CRUD와
  construction/evolution/retrieval begin·complete·fail·trace API를 구현했다.
- source uniqueness, 단일 active version, version별 embedding, self-link·중복
  link·cross-session link 차단을 application과 DB constraint로 검증했다.
- `clone_in_memory()`, `backup_to()`에서 graph 보존을 검증하고 process 재시작 시
  running 상태인 A-MEM 실행을 `interrupted`로 복구하도록 확장했다.
- targeted 검증: `8 passed`, 지정 파일 Ruff 통과.
- 전체 회귀: `232 passed, 1 skipped`, `ruff check src tests` 통과.
- Provider, similarity 계산, ingestion/evolution engine, Vehicle profile은 계획대로
  이번 회차에서 구현하지 않았다.

## 5. 2회차 — 순차 ingestion·evolution engine

### 선행 조건

1회차의 storage·constraint test가 모두 통과해야 한다.

### 목표

Fake A-MEM model과 기존 `FakeEmbeddingModel`만으로 chronological history를
note graph로 만드는 deterministic engine을 구현한다.

### 변경 파일

새 파일:

```text
ubuntu/src/palmclaw_ubuntu/amem.py
```

수정 파일:

```text
ubuntu/src/palmclaw_ubuntu/providers.py
ubuntu/src/palmclaw_ubuntu/storage.py
ubuntu/tests/test_amem.py
```

### 구현 항목

1. `FakeAMemModel`을 추가한다.
   - test가 지정한 construction/evolution response를 순서대로 반환
   - 호출 입력과 usage를 검증할 수 있게 보존
2. `AMemEngine`을 구현한다.
   - history entry를 timestamp·source 순서로 처리
   - note construction validation
   - enriched note text 생성과 embedding
   - 기존 최신 note Top-5 link candidate 계산
   - evolution decision validation
   - note/version/embedding/link를 atomic commit
3. candidate-subset validation을 구현한다.
   - candidate 밖 ID, self ID, duplicate ID 거절
   - neighbor update의 note ID와 배열 길이 검증
   - context/keywords/tags의 개수·문자 길이 제한
4. accepted neighbor update 뒤 최신 metadata embedding을 다시 생성한다.
5. idempotency와 resume를 구현한다.
   - 완료 source는 재호출하지 않음
   - failed run은 상태와 error를 남김
   - transaction 전 실패는 graph를 변경하지 않음
   - process 재시작 후 첫 미완료 source부터 계속
6. note content와 metadata embedding text formatter를 versioned pure function으로
   분리한다.

### Targeted test

```bash
cd ubuntu
pytest tests/test_amem.py
ruff check src/palmclaw_ubuntu/amem.py \
  src/palmclaw_ubuntu/providers.py tests/test_amem.py
```

### 필수 test case

- 빈 graph의 첫 note는 evolution 호출 없이 저장
- 두 번째 note부터 Top-5 후보만 LLM에 노출
- link-only, neighbor-update-only와 둘 다 수행하는 decision
- candidate 밖 update와 dangling link 거절
- correction이 원문을 덮지 않고 metadata version만 추가
- accepted evolution의 neighbor embedding 교체
- 중간 construct/evolve/embed 실패 후 atomic rollback과 resume
- 동일 history 재실행 시 provider 호출 0회
- 다른 처리 순서를 허용하지 않는 chronology 검증

### 완료 조건

- fake history fixture에서 예상 note, link와 version graph가 정확히 생성된다.
- persistent DB를 닫고 다시 열어 ingestion을 재개할 수 있다.
- LLM decision이 source 원문이나 다른 session을 변경할 수 없다.
- query retrieval과 Vehicle 코드 없이 ingestion engine이 독립 동작한다.

### 이번 회차에서 하지 않는 것

- 실제 OpenAI request schema
- query-time linked retrieval rendering
- Vehicle cache/fingerprint/profile
- Cloud 호출

### 구현 결과

`완료 (2026-08-06)`

- `AMemHistoryEntry`, `AMemEngine`과 versioned note-content·metadata-embedding
  pure formatter를 추가했다.
- chronological source만 순차 처리하고 이미 완료된 source는 provider·embedding
  호출 없이 건너뛰도록 구현했다. 저장된 최신 note보다 앞선 backfill과 입력 순서
  위반은 명시적으로 거절한다.
- construction metadata로 새 note embedding을 생성하고, 기존 latest-version
  embedding 중 cosine similarity Top-5만 evolution candidate로 전달한다.
- link-only, neighbor-update-only와 두 동작을 함께 수행하는 evolution을 지원하며,
  neighbor update는 원문을 유지하고 새 metadata version·embedding을 생성한다.
- candidate 밖 ID, self·duplicate ID, `should_evolve=false`의 payload, 빈 값·중복·
  길이·배열 개수 제한을 commit 전에 검증한다.
- model과 embedding 호출 및 검증은 transaction 밖에서 수행하고, 검증된 note,
  version, embedding, link와 construction/evolution 완료 trace는 단일
  `BEGIN IMMEDIATE` transaction으로 atomic commit한다.
- construct/evolve/embed 실패는 construction trace에 `failed`와 error를 남기고
  graph를 변경하지 않는다. 동일 source 재시도, persistent DB close/reopen resume와
  완료 history 전체 재실행 시 provider 0회 동작을 검증했다.
- `FakeAMemModel`에 scripted construction/evolution queue와 request capture를
  추가했다.
- targeted 검증: `14 passed`, 지정 파일 Ruff 통과.
- 전체 회귀: `242 passed, 1 skipped`, `ruff check src tests` 통과.
- 실제 OpenAI schema/request, query-time retrieval, Vehicle profile과 Cloud 호출은
  계획대로 이번 회차에서 구현하지 않았다.

## 6. 3회차 — OpenAI A-MEM provider

### 선행 조건

2회차 fake engine의 validation과 resume test가 통과해야 한다.

### 목표

기존 OpenAI Responses provider 패턴으로 note construction과 evolution structured
output을 구현한다. 모든 검증은 mock client로 수행하며 실제 Cloud를 호출하지
않는다.

### 변경 파일

```text
ubuntu/src/palmclaw_ubuntu/providers.py
ubuntu/tests/test_providers.py
ubuntu/tests/test_amem.py
```

### 구현 항목

1. `OpenAIAMemModel`을 추가한다.
2. construction prompt와 strict JSON schema를 추가한다.
   - keywords
   - tags
   - contextual description
3. evolution prompt와 strict JSON schema를 추가한다.
   - `should_evolve`
   - `linked_note_ids`
   - new-note metadata update
   - candidate neighbor metadata update
4. 다음 version 상수를 분리해 manifest에서 읽을 수 있게 한다.
   - construction prompt/schema
   - evolution prompt/schema
5. 기존 provider의 timeout, output limit, reasoning effort, PII redaction,
   response ID, usage와 latency metadata를 재사용한다.
6. raw history를 instruction이 아닌 untrusted data block으로 직렬화한다.
7. parse 실패나 incomplete output을 빈 metadata로 성공 처리하지 않는다.

### Targeted test

```bash
cd ubuntu
pytest tests/test_providers.py tests/test_amem.py
ruff check src/palmclaw_ubuntu/providers.py \
  tests/test_providers.py tests/test_amem.py
```

### 필수 test case

- construction request의 model, prompt와 strict schema
- evolution request에 후보 밖 note가 포함되지 않음
- response parse와 dataclass 변환
- empty/duplicate/oversized metadata 거절
- malformed JSON, timeout과 API error 전파
- PII redaction 전후 source ID와 speaker mapping 유지
- provider usage·response ID·prompt/schema version 기록
- history 속 prompt injection이 system instruction으로 승격되지 않음

### 완료 조건

- mock Responses client로 두 provider method가 모두 재현된다.
- fake와 OpenAI model이 같은 `AMemModel` contract를 만족한다.
- provider 실패가 engine의 failed run과 resume 경로로 연결된다.
- API key나 network 없이 targeted test가 통과한다.

### 이번 회차에서 하지 않는 것

- 실제 model 품질 판단
- prompt tuning
- query rewrite
- MiniLM 또는 공식 package 설치
- Cloud smoke

### 구현 결과

`완료 (2026-08-06)`

- OpenAI Responses `parse` 기반 `OpenAIAMemModel`을 추가하고 construction과
  evolution을 같은 `AMemModel` contract로 구현했다.
- construction schema는 non-empty contextual description·keywords·tags를
  요구하고 context·배열·item 길이와 duplicate metadata를 제한한다.
- evolution schema는 `should_evolve`, linked candidate IDs, new-note metadata와
  neighbor metadata update를 모두 required field로 선언하고 extra field를
  금지한다.
- construction/evolution prompt·schema version을 각각 독립 상수와 model
  attribute로 노출했으며 engine trace도 operation별 version을 기록한다.
- raw history와 note graph는 JSON의 명시적인 untrusted data block으로만 보내고,
  system instructions와 분리했다. source ID와 note ID는 보존하고 PII는 기존
  cloud privacy redaction 경로를 적용한다.
- model, output limit, reasoning effort, `store=false`, response ID, token usage,
  provider status, latency와 privacy report를 응답 metadata에 보존한다.
- provider 입력 후보를 최대 5개·동일 session·unique ID로 제한하고, 반환된 link와
  neighbor update ID를 실제 전달한 candidate 집합과 다시 대조한다.
- empty·duplicate·oversized metadata, non-candidate ID, malformed JSON, timeout,
  API error, incomplete·failed·cancelled output이 성공 응답으로 변환되지 않음을
  mock client로 검증했다.
- OpenAI incomplete construction이 engine의 failed trace와 graph rollback을 거쳐
  동일 source 재시도에서 정상 완료되는 경로를 검증했다.
- targeted 검증: `49 passed`, 지정 파일 Ruff 통과.
- 전체 회귀: `256 passed, 1 skipped`, `ruff check src tests` 통과.
- 실제 Cloud 호출, model 품질 판단, prompt tuning, query rewrite와 Vehicle profile은
  계획대로 이번 회차에서 수행하지 않았다.

## 7. 4회차 — Query retrieval·graph expansion

### 선행 조건

2회차 graph storage가 안정적이고 3회차 provider metadata가 version contract를
준수해야 한다.

### 목표

완성된 A-MEM graph에서 query embedding Top-k seed와 linked neighbor를 token
budget 안에서 읽는 독립 retriever를 구현한다.

### 변경 파일

새 파일:

```text
ubuntu/src/palmclaw_ubuntu/amem_retrieval.py
ubuntu/tests/test_amem_retrieval.py
```

수정 파일:

```text
ubuntu/src/palmclaw_ubuntu/storage.py
ubuntu/src/palmclaw_ubuntu/models.py
ubuntu/tests/test_amem.py
```

### 구현 항목

1. `AMemRetriever`를 구현한다.
   - query embedding
   - 최신 note version embedding만 ranking
   - cosine Top-10 seed
   - seed와 연결된 direct neighbor expansion
   - note ID deduplication
   - seed 우선, neighbor 후순위 token allocation
2. deterministic tie-break를 고정한다.
   - score descending
   - timestamp descending 또는 명시된 policy
   - note ID ascending
3. Agent rendering을 pure function으로 구현한다.
   - timestamp, speaker, original content
   - current evolved context, keywords와 tags
   - seed/linked 표시는 최소한으로 포함
   - UUID, raw score와 내부 schema는 제외
4. retrieval run/candidate trace를 기록한다.
   - seed/neighbor source
   - rank와 score
   - selected/rejected reason
   - rendered token
5. query-time write는 retrieval trace에만 허용하고 graph fingerprint 대상에서는
   제외한다.

### Targeted test

```bash
cd ubuntu
pytest tests/test_amem_retrieval.py tests/test_amem.py
ruff check src/palmclaw_ubuntu/amem_retrieval.py \
  tests/test_amem_retrieval.py tests/test_amem.py
```

### 필수 test case

- empty graph와 embedding failure
- 최신 metadata version만 검색
- Top-k seed 정확도와 deterministic tie-break
- logical undirected neighbor traversal
- 여러 seed에서 같은 neighbor가 나오는 경우 deduplication
- 작은 token budget에서 seed 우선 보존
- oversized single note의 bounded rendering
- query 전후 graph fingerprint 동일
- close/reopen 뒤 retrieval 결과 동일
- retrieval trace 추가가 content fingerprint를 바꾸지 않음

### 완료 조건

- 고정 graph/query에서 byte-stable Memory content를 반환한다.
- `top_k`, link candidate와 token budget이 metadata/trace에 기록된다.
- retrieval 결과가 note graph를 변경하지 않는다.
- Vehicle 코드 없이 retriever를 독립 사용할 수 있다.

### 이번 회차에서 하지 않는 것

- Tool routing·execution hint
- Agent prompt 변경
- Vehicle profile 등록
- query rewrite 또는 multi-hop graph traversal

### 구현 결과

`완료 (2026-08-06)`

- `AMemRetriever`를 추가해 query embedding과 latest note-version embedding만으로
  cosine Top-k seed를 선택하도록 구현했다. seed 수는 최대 10개다.
- score descending, timestamp descending, note ID ascending tie-break를 stable
  sort로 고정했다.
- 각 seed의 normalized undirected link를 양방향으로 1-hop 확장하고 여러 seed에서
  같은 neighbor가 도달해도 note ID 기준으로 한 번만 포함한다.
- token allocation은 seed 전체를 먼저 처리한 뒤 linked neighbor를 처리한다.
  remaining budget보다 큰 단일 note는 token 단위로 truncate해 bounded rendering을
  만들고 이후 candidate는 `token_budget` 사유로 제외한다.
- Agent rendering은 timestamp, speaker, immutable original content와 current
  context·keywords·tags, 최소한의 seed/linked 표시만 포함한다. 내부 UUID, score와
  schema version은 출력하지 않는다.
- retrieval run/candidate trace에 source, rank, embedding score, selected flag,
  exclusion reason과 rendered token 수를 기록한다.
- `amem_graph_fingerprint()`를 note·version·embedding·link만으로 정의해 retrieval
  trace를 명시적으로 제외했다. query 전후, trace 추가 후와 DB close/reopen 뒤에도
  fingerprint와 rendered content가 동일함을 검증했다.
- empty graph는 embedding을 호출하지 않고 정상 empty result를 반환하며, embedding
  실패는 failed retrieval trace와 error metadata를 남긴다.
- targeted 검증: `22 passed`, 지정 파일 Ruff 통과.
- 전체 회귀: A-MEM을 포함해 `263 passed, 1 skipped`; 기존
  `test_schema_patch_replay_retrieval_agent_and_metrics_are_isolated` 1개는 이번 변경과
  무관한 `vehicle_bench/agent.py`의 `highest-evidence` system-prompt 문구와 기존
  assertion 충돌로 단독 실행에서도 실패했다. `ruff check src tests`는 통과했다.
- Vehicle profile, cache/fingerprint manifest 연결과 실제 benchmark 실행은 계획대로
  이번 회차에서 수행하지 않았다.

## 8. 5회차 — VehicleMemBench profile 통합

### 선행 조건

1~4회차의 engine, provider와 retriever contract가 모두 고정돼야 한다.

### 목표

`cloud_amem`을 기존 VehicleMemBench build/cache/runner/suite에 연결하고 fake
provider로 end-to-end 실행한다.

### 변경 파일

```text
ubuntu/src/palmclaw_ubuntu/cli.py
ubuntu/src/palmclaw_ubuntu/vehicle_bench/__init__.py
ubuntu/src/palmclaw_ubuntu/vehicle_bench/memory.py
ubuntu/src/palmclaw_ubuntu/vehicle_bench/runner.py
ubuntu/src/palmclaw_ubuntu/vehicle_bench/suite.py
ubuntu/tests/test_cli.py
ubuntu/tests/test_vehicle_memory.py
ubuntu/tests/test_vehicle_bench.py
```

필요하면 runner 전용 test를 새 파일로 만들기보다 현재 repository의
`test_vehicle_bench.py` 패턴을 따른다.

### 구현 항목

1. profile과 strategy를 등록한다.
   - `VEHICLE_MEMORY_PROFILES += cloud_amem`
   - `required_memory_strategies()`에 `amem`
   - builder 허용 strategy에 `amem`
2. `VehicleMemoryBuilder`에 A-MEM dependency와 설정을 추가한다.
3. scenario history를 한 번 순차 ingestion하고 cache manifest를 생성한다.
4. `VehicleAMemSnapshot` 또는 명시적 snapshot branch를 구현한다.
   - query별 `AMemRetriever.resolve()`
   - `VehicleMemoryContext.content`와 A-MEM metadata/trace 반환
   - immutable `memory_fingerprint()`
5. cache config에 모든 결과 영향 parameter를 포함한다.
   - model·embedding identity
   - prompt/schema/policy version
   - link candidate `k`
   - retrieval seed `k`와 token budget
6. runner에 A-MEM context만 주입한다.
   - `memory_context.records=()`
   - `fact_records=()`
   - execution hint 없음
   - selector Tool preload 없음
7. CLI option을 추가한다.
   - `--profiles cloud_amem`
   - `--amem-link-candidates`
   - `--amem-retrieval-top-k`
   - debug 전용 `--amem-note-limit`
   - 예상 호출량 dry-run
8. suite에 A-MEM generation, embedding, graph와 retrieval usage를 집계한다.

### Targeted test

```bash
cd ubuntu
pytest tests/test_vehicle_memory.py \
  tests/test_vehicle_bench.py tests/test_cli.py
ruff check src/palmclaw_ubuntu/vehicle_bench \
  src/palmclaw_ubuntu/cli.py \
  tests/test_vehicle_memory.py tests/test_vehicle_bench.py tests/test_cli.py
```

### 필수 test case

- profile validation과 required strategy mapping
- 작은 fake Vehicle history의 graph build와 10 task snapshot 공유
- 동일 cache 재실행 시 construction/evolution 호출 0회
- query와 task 순서를 바꿔도 graph fingerprint 동일
- A-MEM context sources가 `retrieved_memory, query`뿐임
- raw 전체 history, Gold Memory와 Gold call 미노출
- execution hint와 preselected Tool이 비어 있음
- `--amem-note-limit` run에 `partial_history=true`
- partial-history run이 정식 suite aggregate에서 제외됨
- dry-run이 `2N-1` 예상 generation 호출을 보고
- 다른 기존 profile의 cache config/fingerprint 회귀 없음

### 완료 조건

- network 없이 fake `cloud_amem` Vehicle flow가 완료된다.
- scenario snapshot이 task 사이에서 변하지 않는다.
- 결과 artifact에 graph/retrieval/usage trace가 남는다.
- 기존 profile test가 모두 통과한다.

### 이번 회차에서 하지 않는 것

- 실제 VehicleMemBench full history ingestion
- 실제 OpenAI 호출
- A-MEM 기반 Tool route/hint
- 정확도 비교 또는 prompt 변경

### 구현 결과

`완료 (2026-08-06)`

- `cloud_amem`을 Vehicle Memory profile과 `amem` build strategy에 등록하고
  scenario history line을 timestamp/source ID 순서대로 한 번만 ingestion한다.
- `VehicleMemorySnapshot.resolve()`에 A-MEM branch를 추가했다. query embedding,
  Top-k seed와 1-hop linked retrieval 결과만 `VehicleMemoryContext`로 전달하며
  records, Fact records, selector Tool과 execution hint는 생성하지 않는다.
- graph fingerprint에 A-MEM note/version/embedding/link snapshot을 포함하되 retrieval
  trace는 제외했다. 한 snapshot에서 10개 query와 Agent task를 실행한 뒤에도
  fingerprint가 동일함을 검증했다.
- cache config에 construction/evolution model·prompt·schema, embedding identity,
  note/metadata format, graph/retrieval policy, link 후보 수, retrieval Top-k/token
  budget와 debug note limit을 포함했다.
- cache 재실행은 기존 source note를 검증하고 skip하므로 Fake construction/evolution
  호출이 0회다.
- manifest에 graph/generation/embedding/retrieval usage와 `2N-1` 예상 generation
  호출 수를 기록한다. `--amem-dry-run`도 scenario별 동일 예상치를 network 없이
  출력한다.
- `--amem-link-candidates`, `--amem-retrieval-top-k`, debug 전용
  `--amem-note-limit`를 추가했다. note-limit cache는 `partial_history=true`와
  `formal_aggregate_included=false`로 표시하고 suite 정식 합계에서 제외한다.
- Fake Agent E2E에서 context source가 `retrieved_memory, query`뿐이고 raw 전체
  history, Gold Memory/Gold call, preselected Tool과 execution hint가 노출되지 않음을
  검증했다.
- 변경 파일: `storage.py`, `cli.py`, `vehicle_bench/__init__.py`,
  `vehicle_bench/memory.py`, `vehicle_bench/suite.py`, `test_vehicle_memory.py`,
  `test_vehicle_bench.py`, `test_cli.py`.
- targeted 검증: `47 passed, 1 skipped`. 전체 회귀: `268 passed, 1 skipped`.
  `ruff check src tests`와 `git diff --check`도 통과했다.
- 계획대로 실제 full-history Cloud ingestion, OpenAI 호출, 정확도 비교와 A-MEM
  기반 Tool route/hint는 수행하지 않았다.

## 9. 6회차 — 안정화·전체 회귀·문서화

### 선행 조건

fake `cloud_amem` E2E가 통과하고 실제 Cloud 호출 외의 모든 경로가 연결돼야 한다.

### 목표

중단·재개, 개인정보, cache 재현성과 artifact 집계를 강화하고 Cloud smoke를
시작할 수 있는 상태로 구현을 마감한다.

### 변경 파일

필요한 결함 수정 파일과 다음 문서를 갱신한다.

```text
ubuntu/README.md
docs/engineering/README.md
docs/engineering/amem-vehiclemembench-baseline-implementation-plan.md
docs/engineering/amem-implementation-execution-plan.md
```

### 구현 항목

1. A-MEM 전체 failure injection test를 추가한다.
   - construction 실패
   - evolution 실패
   - note/neighbor embedding 실패
   - transaction commit 전 실패
   - manifest write 전후 중단
2. privacy audit을 추가한다.
   - Cloud 전송 전 PII redaction
   - prompt/response/trace/artifact residual span 검사
3. cache compatibility와 fingerprint sensitivity matrix를 검증한다.
4. provider role과 비용 집계 누락을 점검한다.
5. 예상 호출량·token·latency dry-run 결과 형식을 고정한다.
6. 사용자용 CLI 예시, cache 위치와 resume 절차를 `ubuntu/README.md`에 추가한다.
7. 상위 설계 문서의 상태를 `구현 완료 · Cloud smoke 전`으로 갱신한다.
8. 이 실행 계획의 진행 표와 각 회차 구현 결과를 실제 결과로 갱신한다.

### 최종 검증

```bash
cd ubuntu
pytest
pytest --cov=palmclaw_ubuntu --cov-report=term-missing
ruff check src tests
```

coverage 수치는 기존 project gate를 새로 만들기 위한 것이 아니라 A-MEM의
실행되지 않은 branch를 찾는 진단으로 사용한다.

### 완료 조건

- 전체 test와 Ruff가 통과한다.
- 실패 주입 후 재개 시 graph/cache가 정상 run과 동일하다.
- query-time graph mutation과 Gold/raw-history 누출이 0이다.
- 실제 Cloud 호출 없이 모든 profile·CLI·artifact 경로가 검증된다.
- Scenario 6 partial Cloud smoke에 사용할 명령과 예상 비용이 출력된다.
- 구현 범위의 미완료 TODO가 문서에 명시되지 않은 채 남아 있지 않다.

### 이번 회차에서 하지 않는 것

- 실제 Cloud smoke 실행
- Scenario 6 ESM 측정
- ablation 또는 Scenario 1–10 평가
- A-MEM을 제품 runtime 기본 Memory로 연결

### 구현 결과

`완료 (2026-08-06)`

- construction, evolution과 initial embedding의 기존 실패/재개 검증에 더해 neighbor
  re-embedding 실패와 graph transaction commit 직전 실패를 주입했다. 실패 시 새
  note/version/link가 전혀 남지 않고 동일 source retry가 완료됨을 확인했다.
- A-MEM manifest의 ingestion 전 write와 graph commit 후 ready write를 각각 한 번
  실패시켰다. 같은 cache를 다시 build하면 첫 미완료 source에서 재개하며 이미
  생성된 construction/evolution 호출을 반복하지 않는다.
- note, metadata version, embedding과 link 각각에 대한 graph fingerprint sensitivity
  matrix를 추가했다. retrieval/model-call trace는 fingerprint에서 계속 제외된다.
- A-MEM construction/evolution, graph embedding과 query embedding을 공통
  `model_calls` accounting에 `amem_*` role로 기록한다. success/failure, usage,
  latency, privacy metadata와 backend별 예상 비용을 graph/retrieval usage 및 suite
  aggregate에 반영했다. cache reuse 시 과거 build 비용을 이번 실행 비용으로 다시
  더하지 않는다.
- OpenAI A-MEM mock은 source와 candidate PII가 provider 전송 직전에 redaction됨을
  검증한다. Fake Vehicle E2E에는 source note와 generated metadata 양쪽에 email을
  넣어 JSON/JSONL/TSV/Markdown artifact residual span이 0임을 확인했다.
- cache config matrix에서 construction/evolution prompt/schema, embedding identity,
  link candidate, retrieval Top-k/token budget와 partial note limit 변화가 모두 별도
  config/cache identity를 만드는지 검증했다.
- `--amem-dry-run` 출력 계약을 helper/test로 고정했다. scenario별 history/selected
  note, `2N-1` 호출, maximum output-token allowance와 configured output-only cost를
  출력하고 실측 전 input token/latency/full cost는 명시적으로 `null`로 남긴다.
- `ubuntu/README.md`에 Scenario 6 100-note dry-run/build 명령, full 5,395-call bound,
  cache 위치, partial-history 제외와 build/Agent resume 절차를 추가했다.
- 상위 설계 문서를 `구현 완료 · Cloud smoke 전`으로 전환하고 engineering index에
  설계/실행 문서를 등록했다. 실제 Cloud 호출, Scenario 6 ESM과 ablation은 실행하지
  않았다.
- targeted 검증: `111 passed, 1 skipped`. 전체 회귀: `276 passed, 1 skipped`.
  coverage 진단은 전체 `84%`, `amem.py 92%`, `amem_retrieval.py 95%`였으며 새
  coverage gate는 추가하지 않았다. `ruff check src tests`도 통과했다.

## 10. 회차 간 handoff 형식

각 회차 종료 보고에는 다음 다섯 항목만 사용한다.

```text
회차:
완료한 계약:
변경 파일:
검증 결과:
다음 회차 선행 조건:
```

실패나 보류 항목은 다음 회차에 암묵적으로 넘기지 않는다. 현재 회차의 완료
조건에 해당하면 같은 회차에서 해결하고, 범위 밖이면 상위 설계 문서의 후속 작업에
명시한다.

## 11. 구현 완료 후 별도 진행

6회차 이후 작업은 이 문서의 구현 범위가 아니다.

1. bounded Scenario 6 Cloud 구조 smoke와 비용 재산정
2. 전체 Scenario 6 Memory cache 생성과 10-task E2E
3. no-link/no-evolution ablation
4. 고정-cache Agent 반복
5. Scenario 1–10 확장

각 실험의 진입 조건과 판정 기준은
[상위 설계 문서](amem-vehiclemembench-baseline-implementation-plan.md)의
Phase A4~A6을 따른다.

## 12. 후속 구현: `cloud_amem_style`

원본 `cloud_amem`을 변경하지 않고 similarity-gated 효율형을 별도 구현했다.

- profile/strategy: `cloud_amem_style` / `amem_style`
- 기본 gate: top-1 candidate cosine similarity `>= 0.75`
- 호출 계약: note construction `N`회 + gate를 통과한 evolution `E`회
- 원본과 독립된 session, graph fingerprint, cache manifest 사용
- manifest에 threshold, 실제 evolution 호출 수와 skip 수 기록
- CLI: `--amem-style-evolution-threshold`
- dry-run: 최소 `N`, 보수적 최대 `2N-1` 호출 보고
- malformed/incomplete evolution 또는 후보 외 ID는 실패 trace를 보존하고
  해당 note의 evolution만 생략; 인프라 오류는 계속 중단

Fake provider 기반 engine, CLI, Vehicle snapshot/E2E 회귀와 Ruff 검증을
통과했으며 실제 Cloud 호출은 수행하지 않았다.
