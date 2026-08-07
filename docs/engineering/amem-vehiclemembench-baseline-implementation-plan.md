# A-MEM VehicleMemBench Baseline 구현 계획

상태: `구현 완료 · Cloud smoke 전`

작성일: `2026-08-06`

코드 구현의 6회차 실행 순서와 checkpoint는
[A-MEM Baseline 구현 실행 계획](amem-implementation-execution-plan.md)에
분리해 관리한다.

구현 완료 범위는 PalmClaw-native A-MEM domain/storage, 순차
construction·evolution, OpenAI/Fake provider, embedding+linked retrieval,
`cloud_amem` Vehicle profile, cache/resume, CLI dry-run, usage/privacy artifact와
failure injection 검증이다. 실제 OpenAI 호출, Scenario 6 ingestion/ESM과 ablation은
아래 Phase A4~A6의 별도 실험으로 남아 있다.

## 1. 목적

PalmClaw의 VehicleMemBench 평가에 A-MEM 계열의 독립 Memory baseline을
추가한다. A-MEM은 Zettelkasten에서 영감을 받은 원자 note와 동적 link,
과거 note의 metadata evolution을 사용하는 open-schema Memory 방식이다.

새 baseline은 최신 Ours의 Fact-first·Ontology·Tool late binding을 재사용하지
않는다. 같은 Agent, VehicleWorld, Tool discovery와 scorer 위에서 Memory 표현과
검색 방식만 교체해 다음 질문을 검증한다.

1. 원문 중심 note와 동적 관계망이 Ours의 Fact 원자화·관계 손실을 보완하는가.
2. coreference, state shift와 복합 설정에서 linked retrieval이 유효한가.
3. 명시적 Fact versioning이 없는 A-MEM이 correction과 preference conflict를
   얼마나 안정적으로 처리하는가.
4. 정확도 개선이 note별 LLM 생성·evolution 비용을 정당화하는가.

주 참고 자료:

- 논문: [A-MEM: Agentic Memory for LLM Agents](https://papers.neurips.cc/paper_files/paper/2025/file/19909c36f51abc4856b4560aff3d36d6-Paper-Conference.pdf)
- 공식 평가 코드: [WujiangXu/A-mem](https://github.com/WujiangXu/A-mem)
- 공식 라이브러리: [agiresearch/A-mem](https://github.com/agiresearch/A-mem)

## 2. 고정 결정

### 2.1 Profile과 비교 범위

- 최초 profile 이름은 `cloud_amem`으로 한다.
- 기존 profile과 cache는 변경하지 않는다.
- `cloud_amem`은 A-MEM note construction, link generation, memory evolution과
  linked retrieval을 모두 포함한다.
- 첫 비교에서는 A-MEM 결과로 Tool selector나 실행 hint를 만들지 않는다.
  검색된 note context만 Agent에게 주고 기존 Agent Tool discovery를 사용한다.
- 평가 query와 Tool 결과는 Memory를 갱신하지 않는다. 같은 scenario의 10개
  task는 같은 읽기 전용 snapshot을 공유한다.

### 2.2 논문 충실도와 통제 변수

논문, 공식 평가 코드와 공식 라이브러리 사이에는 query rewrite, embedding,
link 표현과 persistence 세부 차이가 있다. 최초 profile은 다음 계약으로 고정한다.

| 항목 | `cloud_amem` 결정 |
| --- | --- |
| note 단위 | Vehicle history 한 줄, 즉 timestamp가 있는 한 speaker turn |
| note metadata | keywords, tags, contextual description |
| note 원문 | timestamp, speaker와 content를 그대로 보존 |
| link 후보 | 새 note embedding 기준 기존 note Top-5 |
| link 판단 | 제한된 후보만 보는 Memory LLM structured decision |
| evolution | 후보 과거 note의 context, keywords, tags만 version update |
| query | 원문 query를 직접 embedding; 별도 query-rewrite LLM 없음 |
| seed retrieval | embedding Top-10 |
| graph expansion | seed와 연결된 note를 공통 token budget 안에서 추가 |
| Memory LLM | 다른 Cloud baseline과 같은 설정값 사용 |
| embedding | 다른 PalmClaw profile과 같은 model·dimension 사용 |
| Agent | 동일 Agent model, prompt, Tool registry와 round 제한 |

논문의 `all-MiniLM-L6-v2`와 공식 평가 코드의 LLM keyword query rewrite는
방법론이 유망한 경우 다음 별도 ablation으로 추가한다.

- `cloud_amem_minilm`: 논문 embedding 재현
- `cloud_amem_query_rewrite`: 공식 평가 코드의 query keyword 생성 추가
- `cloud_amem_style`: top-1 embedding 후보가 설정된 cosine threshold 이상일
  때만 evolution을 실행하는 효율형 변형. 기본 threshold는 `0.75`이며 원본
  `cloud_amem`과 별도 cache/session 및 결과 이름을 사용한다.

따라서 최초 `cloud_amem` 결과를 논문 수치의 exact reproduction이라고 부르지
않는다. 동일 PalmClaw 조건에서 A-MEM 조직 방식의 효과를 측정하는 controlled
baseline으로 정의한다.

`cloud_amem_style`은 호출 수를 `2N-1` 고정 방식에서 실측 `N+E`
(`0 <= E <= N-1`)로 낮추기 위한 PalmClaw 변형이며 논문 방식의 exact
reproduction으로 보고하지 않는다. construction, embedding 후보 선택과 linked
retrieval은 유지하고 evolution gating만 변경한다.

### 2.3 공식 패키지를 직접 포함하지 않는다

공식 라이브러리는 ChromaDB, SentenceTransformers, NLTK, LiteLLM,
scikit-learn 등 현재 Ubuntu runtime에 없는 의존성을 추가한다. 또한 공식
구현의 in-memory object, collection reset과 metadata overwrite 방식은 PalmClaw의
scenario 격리, resume, privacy trace와 맞지 않는다.

초기 구현은 논문 알고리즘과 공개 prompt/schema를 참고해 PalmClaw native
provider·SQLite·embedding abstraction 위에 재현한다. 코드를 직접 전재할 경우
MIT attribution과 third-party notice를 함께 추가한다.

## 3. 방법론 계약

### 3.1 Note construction

각 history entry `h_i`에서 하나의 note `n_i`를 만든다.

```text
n_i = {
  source_message_id,
  timestamp,
  speaker,
  content,
  keywords,
  tags,
  context,
  embedding,
  links
}
```

Memory LLM은 원문을 수정하거나 요약 note로 대체하지 않고 `keywords`, `tags`,
`context`만 생성한다. embedding 대상 text는 다음 순서와 구분자를 고정한다.

```text
timestamp + speaker + content + context + keywords + tags
```

prompt에는 History를 untrusted data로 취급하고 원문에 포함된 instruction을
따르지 않도록 명시한다. 생성 결과는 길이, 배열 크기, 빈 문자열과 제어문자를
deterministic validation한다.

### 3.2 Link generation과 evolution

새 note를 기존 graph에 넣기 직전, 현재 metadata가 포함된 embedding으로 기존
note Top-5를 찾는다. Memory LLM에는 새 note와 이 후보 5개만 제공하고 다음
structured output을 받는다.

```text
should_evolve
linked_note_ids
new_note_keywords
new_note_tags
neighbor_updates[] {
  note_id,
  context,
  keywords,
  tags
}
```

다음 hard rule은 LLM이 변경할 수 없다.

- link와 update 대상은 제공한 Top-5 note ID의 부분집합이어야 한다.
- note의 원문, timestamp, speaker와 source message ID는 변경할 수 없다.
- 다른 scenario/session의 note를 연결할 수 없다.
- 존재하지 않는 ID, 중복 ID와 self-link는 거절한다.
- metadata update는 새 version을 만들고 이전 version을 보존한다.
- accepted evolution 후에는 최신 metadata text를 다시 embedding한다.
- note insert, link insert, neighbor version update는 하나의 transaction으로
  commit한다.

link는 logical undirected relation으로 검색하되 `created_by_note_id`와 생성 시점,
후보 score, model decision을 보존한다. 이는 논문의 “같은 box에 연결된 Memory”
의미를 따른다. directed edge 재현이 필요하면 후속 ablation으로 분리한다.

### 3.3 Retrieval

query를 같은 embedding model로 encode해 최신 note version을 Top-10 검색한다.
그 다음 seed note와 직접 연결된 이웃을 추가한다.

```text
Query
  → embedding Top-10 seed
  → seed별 direct neighbor 후보
  → 중복 제거
  → score·link 근거·최신성으로 deterministic 정렬
  → retrieval token budget까지 rendering
```

seed note를 먼저 보장하고 남은 token budget에 linked neighbor를 넣는다. 전체
출력은 다음 정보를 포함하되 내부 UUID나 score를 Agent 추론에 불필요하게
노출하지 않는다.

```text
- time
- speaker
- original content
- evolved context
- keywords/tags
- linked context 여부
```

retrieval은 완전히 read-only다. 공식 구현에 있는 retrieval count와
last-accessed update는 평가 snapshot을 변이시키므로 trace table에만 기록하고
note 자체에는 쓰지 않는다.

### 3.4 Tool 실행 경계

A-MEM은 Tool ontology, Tool name, required argument schema를 Memory 생성에
사용하지 않는다. 따라서 `cloud_amem`은 다음을 금지한다.

- Tool schema를 note construction·link·evolution prompt에 제공
- query의 reference Tool이나 Gold Memory 제공
- A-MEM metadata에서 Tool name 또는 argument hint 생성
- Gold-derived default, task ID별 rule 또는 Vehicle state 정답 사용

Agent에는 `retrieved_memory + query`만 제공한다. Tool 선택과 argument binding은
기존 공통 Agent loop가 수행한다.

## 4. 저장 모델

기존 `memories`와 Fact table에 A-MEM field를 억지로 추가하지 않고 additive
migration으로 별도 table을 만든다.

### 4.1 `amem_notes`

- `id`, `session_id`, `source_message_id`
- `timestamp`, `speaker`, `content`
- `status`, `created_at`
- scenario 내 `source_message_id` unique constraint

### 4.2 `amem_note_versions`

- `id`, `note_id`, `version`
- `context`, `keywords_json`, `tags_json`
- `supersedes_id`, `created_by_event_id`, `created_at`
- note당 최신 active version unique constraint

### 4.3 `amem_note_embeddings`

- `note_id`, `note_version_id`
- `model_id`, `dimensions`, `vector_json`
- metadata version과 embedding input SHA-256

### 4.4 `amem_links`

- 정규화된 `left_note_id`, `right_note_id`
- `created_by_note_id`, `evolution_event_id`
- 후보 similarity, LLM decision과 생성 시각
- 같은 note pair unique constraint

### 4.5 실행·진단 table

- `amem_construction_runs`
- `amem_evolution_events`
- `amem_retrieval_runs`
- `amem_retrieval_candidates`

각 table은 status, error, model/prompt/schema version, usage와 latency를 남긴다.
중단된 write run은 기존 repository recovery 패턴을 따라 재시도 가능해야 한다.

## 5. 코드 구조

### 5.1 모델과 protocol

`ubuntu/src/palmclaw_ubuntu/models.py`:

- `AMemNote`
- `AMemNoteVersion`
- `AMemLink`
- `AMemConstructionResponse`
- `AMemEvolutionDecision`
- `AMemRetrievalResult`

`ubuntu/src/palmclaw_ubuntu/contracts.py`:

```text
AMemModel.construct(entry) -> AMemConstructionResponse
AMemModel.evolve(new_note, neighbors) -> AMemEvolutionDecision
```

Provider 호출이 두 메서드에서 같은 backend/model을 쓰더라도 prompt version과
schema version은 별도로 기록한다.

### 5.2 Provider

`ubuntu/src/palmclaw_ubuntu/providers.py`에 `OpenAIAMemModel`을 추가한다.

- 기존 Responses API, timeout, retry, PII redaction과 usage 수집 재사용
- strict structured output 사용
- construction과 evolution의 독립 prompt/schema version
- 출력 parse 실패 시 `General` 같은 silent fallback을 만들지 않음
- bounded retry 후 scenario Memory build를 실패 상태로 남기고 resume 허용

초기 version 이름:

```text
vehicle-amem-note-construction-v1
vehicle-amem-note-v1
vehicle-amem-evolution-v1
vehicle-amem-evolution-decision-v1
vehicle-amem-retrieval-v1
```

### 5.3 Repository와 engine

새 모듈 경계:

```text
ubuntu/src/palmclaw_ubuntu/amem.py
ubuntu/src/palmclaw_ubuntu/amem_retrieval.py
```

`amem.py`는 순차 ingestion, decision validation, transaction과 resume를 담당한다.
`amem_retrieval.py`는 embedding cache, seed ranking, graph expansion, token budget과
rendering을 담당한다.

기존 `EmbeddingModel` protocol과 model-call accounting을 재사용한다. 외부 vector
DB는 추가하지 않는다. 초기 benchmark 규모에서는 scenario별 SQLite scan과
in-memory cosine ranking으로 충분하며, 성능 문제가 실측된 뒤 ANN 도입을
검토한다.

### 5.4 VehicleMemBench builder와 snapshot

`ubuntu/src/palmclaw_ubuntu/vehicle_bench/memory.py`:

- `VEHICLE_MEMORY_PROFILES`에 `cloud_amem` 추가
- `required_memory_strategies()`에 `amem` 연결
- `VehicleMemoryBuilder`에 `amem_model`과 A-MEM 설정 추가
- 허용 strategy에 `amem` 추가
- `VehicleAMemSnapshot` 또는 기존 snapshot의 명시적 A-MEM branch 추가
- cache config와 fingerprint에 모든 A-MEM version·parameter 포함

fingerprint는 최소한 다음 내용을 반영한다.

- dataset/scenario hash
- history entry hash
- Memory/Embedding model과 dimension
- construction/evolution prompt·schema version
- link candidate `k`
- retrieval seed `k`와 token budget
- metadata validation·graph expansion policy version

동일 fingerprint로 재실행하면 완료 note, embedding과 evolution event를 재사용하고
새 Cloud 호출을 하지 않아야 한다.

### 5.5 CLI와 runner

`ubuntu/src/palmclaw_ubuntu/cli.py`:

- `--profiles cloud_amem` 허용
- `--amem-link-candidates`, `--amem-retrieval-top-k` 추가
- debug 전용 `--amem-note-limit`를 허용하되 제한된 run에는
  `partial_history=true`를 기록하고 E2E 결과 집계에서 제외
- 실행 전 예상 note 수, LLM 호출 수와 embedding 수를 출력하는 dry-run 제공

`ubuntu/src/palmclaw_ubuntu/vehicle_bench/runner.py`:

- `cloud_amem`을 Memory profile로 처리
- `VehicleMemoryContext.content`만 Agent context에 연결
- Fact/Tool execution hint branch에는 넣지 않음
- raw history와 Gold 정보 미노출 검증 유지
- A-MEM retrieval metadata와 trace를 결과 artifact에 보존

`ubuntu/src/palmclaw_ubuntu/vehicle_bench/suite.py`:

- A-MEM provider role과 usage 집계
- graph/retrieval 품질 및 profile별 비용 표 추가

## 6. 비용과 실행 제한

공식 평가 방식은 history turn마다 note construction과 evolution을 각각 한 번
수행한다. 첫 note의 빈 graph evolution을 생략하면 `N`개 history line에 대해
예상 Memory LLM 호출은 `2N-1`이다.

현재 보존 artifact 기준 history line 수는 다음과 같다.

| 범위 | History line | 예상 A-MEM LLM 호출 |
| --- | ---: | ---: |
| Scenario 6 | 2,698 | 5,395 |
| Scenario 1–10 | 27,313 | 54,625 |
| 전체 Scenario 1–50 | 134,518 | 269,035 |

이는 현재 batch 기반 Ours보다 매우 큰 생성 비용이다. 따라서 다음 원칙을 둔다.

- 구현 검증 전 full Cloud ingestion을 실행하지 않는다.
- dry-run에서 예상 호출·입력 token·최대 비용을 먼저 산출한다.
- scenario cache를 ingestion과 Agent 평가 사이에서 분리 재사용한다.
- embedding과 재생성 대상 neighbor update는 가능한 한 한 요청에 batch한다.
- note construction을 여러 turn에 대해 batch하는 최적화는 최초 profile에 넣지
  않는다. 필요하면 `cloud_amem_batched`라는 별도 방법론으로 평가한다.
- 부분 history smoke 결과는 정식 E2E 성능표에 넣지 않는다.

## 7. 구현 단계

### Phase A0 — 명세·fixture 고정

- 이 문서의 `cloud_amem` 계약을 test fixture로 고정한다.
- 논문과 공식 코드에서 사용하는 construction/evolution prompt의 의미를 보존하되
  PalmClaw의 untrusted-data와 strict JSON 규칙을 추가한다.
- correction, multi-person, conditional preference, compound setting과 unrelated
  conversation을 포함한 작은 chronological fixture를 만든다.

완료 조건:

- 동일 fixture로 note, link와 version expected output을 표현할 수 있다.
- Gold Tool이나 Vehicle task 정답이 fixture Memory 생성 입력에 없다.

### Phase A1 — 저장 계약과 fake engine

- model dataclass, protocol과 additive SQLite migration 구현
- fake A-MEM model로 순차 construction/evolution 구현
- idempotent resume, atomic transaction과 fingerprint 구현
- latest metadata version과 embedding consistency 검사 추가

완료 조건:

- 중간 note에서 강제 실패 후 재개해도 중복 note/link/version이 없다.
- source content와 timestamp가 evolution 전후 동일하다.
- 다른 scenario note 연결이 DB constraint와 runtime validation에서 모두 차단된다.

### Phase A2 — Cloud provider와 graph retrieval

- OpenAI construction/evolution structured output 구현
- 같은 embedding abstraction으로 note와 query encode
- Top-5 link 후보, validated link/evolution과 re-embedding 구현
- Top-10 seed + linked neighbor retrieval과 token cap 구현
- provider usage, privacy와 retrieval trace 연결

완료 조건:

- 잘못된 ID, self-link, 후보 밖 update와 oversized metadata가 거절된다.
- retrieval 전후 snapshot fingerprint가 같다.
- linked neighbor가 seed와 구분되어 trace되고 중복 없이 rendering된다.
- token budget을 초과하지 않는다.

### Phase A3 — Vehicle profile 연결

- builder, cache, CLI, runner와 suite에 `cloud_amem` 추가
- scenario history를 한 번만 ingestion하고 task들이 snapshot을 공유
- A-MEM context에는 raw 전체 history, Gold Memory와 Gold call이 없음 확인
- 기존 profile validation과 resume 경로 회귀 테스트

완료 조건:

- fake provider를 사용한 작은 Vehicle fixture 10/10 runtime 완료
- 동일 cache 재실행 시 Memory provider 호출 0회
- query/task 순서를 바꿔도 fingerprint와 retrieval 결과가 동일함
- 다른 기존 profile의 결과와 cache key가 변하지 않음

### Phase A4 — 구조 smoke와 비용 판정

- Scenario 6 앞부분의 bounded note로 Cloud construction/evolution smoke
- parse 성공률, link/update 비율, 평균 degree, metadata version과 비용 측정
- 전체 Scenario 6 예상 비용을 실측 평균 token/latency로 다시 계산

완료 조건:

- persistent parse failure와 invalid relation이 0이거나 명시적 실패 처리됨
- 개인정보 redaction 후 artifact sensitive span이 0
- full Scenario 6 실행 여부를 비용 산출물로 결정 가능

이 단계의 partial-history 결과는 ESM 비교에 사용하지 않는다.

### Phase A5 — Scenario 6 전체 E2E

- 전체 2,698 history entry를 ingestion해 고정 cache 생성
- 동일 Agent 설정으로 Scenario 6의 10 task 실행
- 다음 profile과 비교
  - `no_memory`
  - `cloud_recursive_summary`
  - `cloud_structured_hybrid`
  - 최신 Post-normalized Recursive-assisted Ours
    (`cloud_schema_informed_recursive_assisted_fact_patch`)
  - `cloud_amem`

완료 조건:

- 10/10 task runtime 완료 또는 실패 원인이 artifact에 명시됨
- ESM, State/Tool F1, argument exact와 비용이 같은 report에 기록됨
- note/link/evolution trace에서 대표 성공·실패를 재구성 가능

### Phase A6 — Ablation과 Scenario 1–10

Scenario 6에서 pipeline이 안정적일 때만 다음 고정-cache ablation을 수행한다.

1. note metadata + embedding retrieval, link/evolution 없음
2. `+ link generation`, neighbor metadata evolution 없음
3. full `cloud_amem`

그 후 Scenario 1–10으로 확대하고 reasoning type별 차이를 본다. Agent 실행
변동을 분리하기 위해 각 방법의 Memory cache를 고정한 뒤 Agent를 3회 반복한다.

확대 조건:

- scenario cache 재사용과 resume가 검증됨
- Scenario 6에서 graph expansion이 실제 선택 context를 바꾼 표본이 존재함
- invalid link/update와 privacy violation이 0
- 예상 Cloud 호출량과 비용이 승인된 실험 예산 안에 있음

## 8. 테스트 계획

### 단위 테스트

- note construction schema와 metadata limit
- exact history source 보존과 PII redaction
- candidate-subset link validation
- self/cross-session/dangling link 거절
- logical undirected traversal과 pair deduplication
- neighbor metadata version과 re-embedding
- deterministic seed/neighbor ordering
- token cap, empty graph와 no-result query
- build failure recovery와 idempotency
- fingerprint sensitivity와 query-time immutability

### 통합 테스트

- fake A-MEM provider의 chronological ingestion
- SQLite close/reopen 후 동일 retrieval
- cache reuse와 interrupted manifest recovery
- runner context source가 `retrieved_memory, query`로 제한됨
- A-MEM profile에 Tool hint가 생성되지 않음
- scenario·task 격리와 task order independence
- results JSONL/TSV/Markdown 및 privacy audit

### 회귀 검증

```bash
cd ubuntu
pytest
ruff check src tests
```

관련 테스트 파일은 최소한 다음으로 분리한다.

```text
tests/test_amem.py
tests/test_amem_retrieval.py
tests/test_providers.py
tests/test_vehicle_memory.py
tests/test_vehicle_agent_runner.py
tests/test_vehicle_suite.py
tests/test_cli.py
```

## 9. 평가 지표

기존 E2E 지표:

- Exact State Match
- State/value F1
- Tool F1
- Argument exact
- unnecessary Tool call과 execution error
- Agent input/output token, latency와 cost

A-MEM 전용 구조 지표:

- note construction/evolution 호출 성공률
- note 수와 metadata version 수
- link 수, isolated note 비율, 평균·최대 degree
- evolution 적용률과 neighbor update 수
- seed/linked selected note 수
- linked neighbor가 없을 때와 있을 때의 taskwise ESM
- retrieval context token과 중복 제거 수
- Memory 생성/embedding 호출·token·latency·cost

retrieval recall은 기존 Fact record Recall@k와 직접 비교하지 않는다. A-MEM은
Fact key가 없으므로 다음 두 값을 별도 표기한다.

- reviewed task의 정답 evidence source-line recall
- 검색 context에 필요한 literal value가 존재하는지에 대한 value recall

자동 label이 없는 task는 사후 Gold call에서 source note를 역생성하지 않고
`not reviewed`로 남긴다.

## 10. 실패 가설과 판정

### 기대 가설

- linked note가 같은 사건의 복합 설정을 함께 복구한다.
- evolved context가 멀리 떨어진 인물·상황 관계의 검색 표현을 개선한다.
- coreference와 state shift에서 Ours보다 retrieval coverage가 높다.

### 주요 실패 위험

- 오래된 원문 note와 수정 note가 함께 검색돼 correction을 혼동한다.
- LLM evolution이 근거 없는 관계나 context를 과잉 생성한다.
- link expansion이 관련성보다 context 크기만 늘린다.
- Tool knowledge가 없어 memory recall이 높아도 Tool/argument 실행에서 실패한다.
- note별 두 번의 LLM 호출로 비용·시간이 baseline 가치보다 커진다.

### Baseline 채택 조건

A-MEM은 Ours보다 높은 ESM을 내야만 baseline으로 채택되는 것이 아니다. 다음을
모두 만족하면 비교 baseline으로 유지한다.

1. 논문 핵심인 note construction, link, evolution과 linked retrieval이 trace로
   검증된다.
2. query/Gold 누출 없이 같은 VehicleWorld·Agent·scorer를 사용한다.
3. snapshot과 cache가 재현 가능하고 query-time mutation이 없다.
4. 실패·비용과 privacy 지표가 완전하게 기록된다.
5. full과 no-link/no-evolution ablation을 구분할 수 있다.

제품 runtime 채택은 별도 결정이다. 제품 후보가 되려면 정확도 외에도 note별
생성 호출을 줄이는 batch/on-device 설계와 evidence-grounded evolution이 추가로
필요하다.

## 11. 예상 변경 파일

새 파일:

```text
ubuntu/src/palmclaw_ubuntu/amem.py
ubuntu/src/palmclaw_ubuntu/amem_retrieval.py
ubuntu/src/palmclaw_ubuntu/migrations/019_amem.sql
ubuntu/tests/test_amem.py
ubuntu/tests/test_amem_retrieval.py
```

수정 파일:

```text
ubuntu/src/palmclaw_ubuntu/models.py
ubuntu/src/palmclaw_ubuntu/contracts.py
ubuntu/src/palmclaw_ubuntu/providers.py
ubuntu/src/palmclaw_ubuntu/storage.py
ubuntu/src/palmclaw_ubuntu/cli.py
ubuntu/src/palmclaw_ubuntu/vehicle_bench/__init__.py
ubuntu/src/palmclaw_ubuntu/vehicle_bench/memory.py
ubuntu/src/palmclaw_ubuntu/vehicle_bench/runner.py
ubuntu/src/palmclaw_ubuntu/vehicle_bench/suite.py
ubuntu/tests/test_providers.py
ubuntu/tests/test_vehicle_memory.py
ubuntu/tests/test_vehicle_bench.py
ubuntu/tests/test_cli.py
ubuntu/README.md
docs/engineering/README.md
```

실제 구현은 additive `019_amem.sql` migration을 사용한다. 구현 및 network-free
검증 결과는
[A-MEM Baseline 구현 실행 계획](amem-implementation-execution-plan.md)의
1~6회차 결과가 source of truth다.
