# PalmClaw Ubuntu PoC 소스 기반 상세 참조

> 검증 기준일: 2026-07-27  
> 대상: `ubuntu/`의 Python Runtime, Tool-schema incremental memory, 평가 코드  
> 저장소 기준: PalmClaw Git `29a8724`; 본 문서는 현재 workspace의 `ubuntu/` 구현을 직접 대조해 작성

## 1. 문서 목적과 신뢰 수준

이 문서는 PalmClaw 원본 Android 앱을 설명하는 일반 문서가 아니라, 지금까지
별도로 구현한 Ubuntu Agent Runtime과 Memory 연구 PoC의 **현재 코드 동작을
질문·리뷰·수정 시 빠르게 확인하기 위한 기준 문서**다.

기능 설명은 다음 순서로 신뢰 근거를 확인했다.

1. 실제 Python source와 SQL migration을 구현의 최종 근거로 사용했다.
2. public API의 호출 경로를 `CLI → Runtime assembly → Agent/Memory/Tool →
   SQLite` 순서로 대조했다.
3. 해당 경로를 검증하는 테스트 파일을 기능별로 연결했다.
4. 실제 Cloud 실행 결과가 있는 항목은 재현 artifact를 별도로 연결했다.
5. 계획만 존재하거나 부분 구현된 항목은 `미구현` 또는 `제한`으로 명시했다.

문서의 상태 표기는 다음 의미다.

- **소스 검증**: 현재 source와 migration에 구현이 존재한다.
- **테스트 검증**: 관련 자동 테스트가 통과한다.
- **실행 검증**: 실제 OpenAI 또는 공식 VehicleMemBench checkout으로 생성한
  artifact가 있다.
- **제한**: 코드가 보장하지 않는 경계 또는 아직 구현되지 않은 사항이다.

이는 formal verification이나 보안 인증을 뜻하지 않는다. 특히 외부 API,
운영체제 race condition, 모델의 의미적 정확성은 테스트만으로 완전 보장할 수
없다. 코드가 바뀌면 이 문서도 같은 변경에서 갱신해야 한다.

현재 snapshot에서 직접 재검증한 결과:

- `pytest -q`: 수집 122개, 121 passed, 1 skipped
- skip 1개: `VEHICLEMEMBENCH_ROOT`를 명시할 때만 실행되는 공식 checkout
  integration smoke
- `ruff check src tests`: 통과
- 이 문서의 source/test/artifact link: 144개 모두 존재
- `#L...` source anchor: 57개 모두 현재 파일 line 범위 안

공식 VehicleMemBench 실행 여부는 위 pytest skip과 별개로, 저장된 Cloud live
artifact를 16.7절에서 근거로 제시한다.

## 2. 저장소와 구현 경계

| 경로 | 역할 | 이 문서의 범위 |
|---|---|---|
| `app/` | 원본 PalmClaw Android/Kotlin 앱 | 참고 대상이지만 이번 Ubuntu PoC 구현에는 연결되지 않음 |
| `ubuntu/` | Ubuntu용 독립 Python CLI Agent Runtime | 상세 참조 대상 |
| `ubuntu/src/palmclaw_ubuntu/vehicle_bench/` | VehicleMemBench adapter와 평가 전용 Agent | 상세 참조 대상 |
| `docs/engineering/` | 설계, 계획, 방법론, 결과 해설 | 보조 근거 |

현재 `ubuntu/`는 Android 앱의 Kotlin class를 호출하거나 Android DB를 공유하지
않는다. Python package, SQLite DB, workspace, Skill root가 모두 독립적이다.
따라서 Ubuntu에서 검증한 기능이 Android 앱에 자동으로 반영되지는 않는다.

패키지 정의는
[`ubuntu/pyproject.toml`](../../ubuntu/pyproject.toml)에 있다. Python
`>=3.12,<3.13`, 실행 명령은 `palmclaw`, 주요 dependency는 `openai`,
`httpx`, `pydantic`, `jsonschema`, `tiktoken`이다.

## 3. 현재 지원 범위 요약

| 기능 | 상태 | 실제 범위 |
|---|---|---|
| Ubuntu CLI 대화 | 소스·테스트 검증 | 단일 turn `ask`, interactive `chat` |
| OpenAI AgentModel | 소스·테스트 검증 | Responses API 기반 Tool calling |
| Fake AgentModel | 소스·테스트 검증 | network 없는 개발·테스트용 Echo/Script |
| Session 영속 저장 | 소스·테스트 검증 | SQLite session/turn/message/active session |
| Agent loop | 소스·테스트 검증 | LLM → Tool → Tool result → LLM, 최대 round 제한 |
| Context Builder | 소스·테스트 검증 | 정책, standard memory, Tool memory, Skill, 최근 완전한 대화 |
| 기본 Tool | 소스·테스트 검증 | `file_read`, `file_write`, `web_fetch` |
| Tool JSON schema 검증 | 소스·테스트 검증 | JSON Schema Draft 2020-12 |
| Workspace 경계 | 소스·테스트 검증 | session/shared 상대 경로, escape 차단 |
| Cloud summary memory | 소스·테스트 검증 | 누적 Markdown summary version |
| Cloud structured memory | 소스·테스트 검증 | evidence, immutable version, gate, retrieval |
| Local summary/structured memory | 소스·테스트 검증 | loopback OpenAI-compatible Chat Completions |
| BM25/embedding/hybrid retrieval | 소스·테스트 검증 | standard memory와 Tool memory 각각 지원 |
| ADD/UPDATE/MERGE/DELETE patch | 소스·테스트 검증 | Tool-schema record에 transaction 적용 |
| Cloud patch MemoryModel | 소스·테스트·실행 검증 | OpenAI structured output |
| Background patch queue | 소스·테스트 검증 | turn 후 queue, CLI로 bounded batch 실행 |
| Tool-schema routing/retrieval | 소스·테스트·실행 검증 | lexical route + scope/condition + top-k |
| PII/secret redaction | 소스·테스트 검증 | 규칙 기반 탐지·Cloud 전송 직전 redaction |
| Generic synthetic 평가 | 소스·테스트 검증 | offline fixture 또는 live MemoryModel |
| VehicleMemBench 평가 | 소스·테스트·실행 검증 | 공식 simulator/scorer, 격리된 평가 Agent |
| Local patch SLM | **미구현** | Local summary/structured와 달리 patch backend는 OpenAI/Fake만 가능 |
| Android 통합 | **미구현** | Python PoC와 Kotlin 앱 간 adapter 없음 |
| 자동 idle scheduler/daemon | **미구현** | `memory patch run`을 명시적으로 실행해야 함 |
| Shell/browser/범용 자동화 Tool | **미구현** | 일반 Runtime에는 파일 2종과 HTTPS GET만 있음 |
| LLM route classifier | **부분 구현** | protocol과 fallback hook은 있으나 Runtime assembly가 classifier를 주입하지 않음 |

## 4. 전체 구조와 실행 흐름

### 4.1 Composition root

Runtime 생성의 단일 진입점은
[`create_runtime()`](../../ubuntu/src/palmclaw_ubuntu/application.py#L74)이다.
이 함수가 다음 object를 조립한다.

- [`Settings`](../../ubuntu/src/palmclaw_ubuntu/config.py#L46)
- [`RuntimeLease`](../../ubuntu/src/palmclaw_ubuntu/runtime_lock.py#L8)
- [`SQLiteRepository`](../../ubuntu/src/palmclaw_ubuntu/storage.py#L90)
- [`SkillsLoader`](../../ubuntu/src/palmclaw_ubuntu/skills.py#L19)
- [`ToolRegistry`](../../ubuntu/src/palmclaw_ubuntu/tools.py#L47)
- [`MemoryEngine`](../../ubuntu/src/palmclaw_ubuntu/memory.py#L46)
- [`ContextBuilder`](../../ubuntu/src/palmclaw_ubuntu/context.py#L26)
- [`AgentLoop`](../../ubuntu/src/palmclaw_ubuntu/agent.py#L37)
- [`SessionTurnCoordinator`](../../ubuntu/src/palmclaw_ubuntu/coordinator.py#L11)
- 선택적 [`BackgroundMemoryPatchWorker`](../../ubuntu/src/palmclaw_ubuntu/background_memory.py#L37)
- 선택적 [`ToolMemoryRetriever`](../../ubuntu/src/palmclaw_ubuntu/memory_router.py#L233)

Runtime object의 실제 소유 관계는
[`Runtime`](../../ubuntu/src/palmclaw_ubuntu/application.py#L47)에 고정되어 있다.
종료할 때 repository를 닫고 process-level lease를 해제한다.

### 4.2 한 사용자 turn의 End-to-End 순서

```text
CLI ask/chat
  → active session 결정
  → SessionTurnCoordinator
      → 같은 session은 직렬화, 전체 session 동시성 제한
      → AgentLoop.run
          1. turn 생성, user message의 secret을 redaction 후 저장
          2. standard memory 검색
          3. 선택적으로 Tool-schema memory route/retrieval
          4. ContextBuilder로 system + memory + Skill + history 조립
          5. AgentModel 호출 및 model trace 저장
          6. Tool call이 있으면 schema 검증 → 실행 → 결과 저장
          7. Tool result를 포함해 다음 model round 반복
          8. Tool call이 없으면 turn 완료
          9. standard memory consolidation 시도
     10. 성공한 foreground turn에 patch queue job enqueue
```

핵심 구현은
[`AgentLoop.run()`](../../ubuntu/src/palmclaw_ubuntu/agent.py#L61)과
[`SessionTurnCoordinator.run()`](../../ubuntu/src/palmclaw_ubuntu/coordinator.py#L24)에
있다. Patch 추출은 foreground 답변 경로에서 실행하지 않고 성공 turn을 durable
queue에 넣기만 한다.

### 4.3 Agent loop의 terminal 상태

Agent loop는 응답 완료 외에도 timeout, 사용자 취소, 예외, 최대 Tool round 초과를
terminal turn으로 저장한다. Model call과 Tool call에는 latency, token usage,
backend/model, error, side-effect 분류가 남는다.

안전하지 않은 write Tool이 같은 turn에서 동일한 canonical argument로 다시
호출되면 fingerprint로 중복 실행을 차단한다. fingerprint는 Tool 이름과
정렬된 JSON argument의 SHA-256이다.

검증 근거:

- [`test_agent_loop_e2e.py`](../../ubuntu/tests/test_agent_loop_e2e.py)
- [`test_runtime_safety.py`](../../ubuntu/tests/test_runtime_safety.py)
- [`test_parallel_sessions_keep_history_isolated`](../../ubuntu/tests/test_runtime_safety.py#L365)

### 4.4 Timeout의 정확한 의미

Model과 Tool 호출은 worker thread future에 timeout을 적용한다. timeout 발생 시
future를 취소하고 Agent turn은 terminal 상태로 정리한다. 다만 Python thread에서
이미 실행 중인 blocking operation을 강제 종료하는 OS sandbox는 아니다.
즉, 사용자 관점의 기다림과 DB 상태는 제한되지만 underlying third-party call이
즉시 중단된다고 보장하지 않는다.

## 5. 설정, backend와 데이터 위치

### 5.1 기본 경로

[`Settings.from_env()`](../../ubuntu/src/palmclaw_ubuntu/config.py#L110)의 기본값:

- data: `ubuntu/.data`
- DB: `<data>/palmclaw.db`
- session workspace: `<data>/workspaces`
- shared workspace: `<data>/shared`
- user Skill: `<data>/skills`
- built-in Skill: Python package의 `builtin_skills/`
- runtime lock: `<data>/.runtime.lock`

`PALMCLAW_DATA_DIR`, `PALMCLAW_WORKSPACE_ROOT`,
`PALMCLAW_SHARED_WORKSPACE_ROOT`, `PALMCLAW_SKILLS_ROOT`로 바꿀 수 있다.

### 5.2 Model 기본값과 역할

코드 기본값은
[`config.py`](../../ubuntu/src/palmclaw_ubuntu/config.py#L7)에 선언되어 있다.

| 역할 | 기본 model ID | API |
|---|---|---|
| Agent | `gpt-5.6-terra` | OpenAI Responses |
| summary/structured/patch Memory | `gpt-5.6-luna` | OpenAI Responses/structured parse |
| Embedding | `text-embedding-3-small` | OpenAI Embeddings |
| Local memory | `local-memory` | loopback OpenAI-compatible Chat Completions |

이는 **코드에 저장된 default string**이며 계정별 model 접근 가능성을 보장하는
문장은 아니다. 실제 ID는 `PALMCLAW_AGENT_MODEL`,
`PALMCLAW_MEMORY_MODEL`, `PALMCLAW_PATCH_MEMORY_MODEL`,
`PALMCLAW_EMBEDDING_MODEL`로 지정한다. API key는
`OPENAI_API_KEY` 환경변수에서만 읽으며 Runtime이 DB나 문서로 저장하지 않는다.

### 5.3 Backend 조합

- `PALMCLAW_BACKEND=openai|fake`: foreground Agent를 선택한다.
- `PALMCLAW_MEMORY_BACKEND=openai|local|fake`: standard summary/structured
  MemoryModel을 독립 선택한다.
- `PALMCLAW_PATCH_MEMORY_BACKEND=auto|openai|fake`: patch MemoryModel을
  선택한다.

중요한 현재 동작:

1. `PALMCLAW_MEMORY_BACKEND=local`은 summary와 structured consolidation에만
   적용된다.
2. patch에는 Local backend가 없다.
3. 전체 Agent backend가 `openai`이면 standard memory backend가 local이어도
   embedding은 기본적으로 OpenAI를 사용한다.
4. local endpoint는 `localhost`, `127.0.0.1`, `::1`만 허용하며 credential이
   포함된 URL과 원격 host를 거부한다.

구현 근거:

- backend 조립:
  [`_assemble_runtime()`](../../ubuntu/src/palmclaw_ubuntu/application.py#L109)
- provider contract:
  [`contracts.py`](../../ubuntu/src/palmclaw_ubuntu/contracts.py)
- provider 구현:
  [`providers.py`](../../ubuntu/src/palmclaw_ubuntu/providers.py)
- Local endpoint 제한:
  [`_local_chat_endpoint()`](../../ubuntu/src/palmclaw_ubuntu/providers.py#L1147)
- provider test:
  [`test_providers.py`](../../ubuntu/tests/test_providers.py)

### 5.4 주요 기본 limit

| 설정 | 기본값 |
|---|---:|
| model timeout | 60초 |
| Tool timeout | 15초 |
| 최대 Tool round | 8 |
| 전체 동시 session | 4 |
| history message 후보 | 40 |
| 전체 context | 16,000 token |
| standard memory context | 2,000 token |
| Skill context | 3,000 token |
| Tool memory context | 1,000 token |
| Tool result | 20,000 character |
| consolidation trigger | 미처리 message 6개 |
| file | 1,000,000 byte |
| web response | 500,000 byte |
| redirect | 3 |
| standard memory top-k | 5 |
| Tool memory top-k | 5 |
| Tool memory 최대 route | 3 |
| patch worker batch | 4 job |
| patch 최대 attempt | 3 |
| patch retry delay | 30초 |
| patch lease | 180초 |

모든 필드, 환경변수 parsing, 상호 budget 검사는
[`Settings`](../../ubuntu/src/palmclaw_ubuntu/config.py#L46)와
[`Settings.validate()`](../../ubuntu/src/palmclaw_ubuntu/config.py#L380)이
최종 근거다.

## 6. SQLite 영속 계층과 복구

### 6.1 연결과 transaction

[`SQLiteRepository`](../../ubuntu/src/palmclaw_ubuntu/storage.py#L90)는:

- `foreign_keys=ON`
- `busy_timeout=5000`
- file DB일 때 WAL mode
- 한 process 안에서 `RLock`
- 일반 write의 명시적 transaction/commit/rollback
- patch batch의 `BEGIN IMMEDIATE`

를 사용한다. 시작 시 packaged SQL을 파일명 순서로 적용하고
`schema_migrations`로 완료 migration을 추적한다.

### 6.2 Migration별 책임

| Migration | 핵심 table/변경 |
|---|---|
| [`001_initial.sql`](../../ubuntu/src/palmclaw_ubuntu/migrations/001_initial.sql) | session, turn, message, Tool call/result, summary memory, consolidation/model call, runtime state |
| [`002_provider_continuation.sql`](../../ubuntu/src/palmclaw_ubuntu/migrations/002_provider_continuation.sql) | provider continuation item |
| [`003_runtime_safety.sql`](../../ubuntu/src/palmclaw_ubuntu/migrations/003_runtime_safety.sql) | Tool side-effect, retry safety, fingerprint |
| [`004_structured_memory_retrieval.sql`](../../ubuntu/src/palmclaw_ubuntu/migrations/004_structured_memory_retrieval.sql) | structured memory field, evidence, status event, embedding, retrieval trace |
| [`005_memory_validation_privacy.sql`](../../ubuntu/src/palmclaw_ubuntu/migrations/005_memory_validation_privacy.sql) | gate decision과 review queue |
| [`006_evaluation_runs.sql`](../../ubuntu/src/palmclaw_ubuntu/migrations/006_evaluation_runs.sql) | evaluation run/case |
| [`007_tool_schema_memory_patching.sql`](../../ubuntu/src/palmclaw_ubuntu/migrations/007_tool_schema_memory_patching.sql) | Tool-memory record/source, patch run/proposal |
| [`008_tool_memory_patch_engine.sql`](../../ubuntu/src/palmclaw_ubuntu/migrations/008_tool_memory_patch_engine.sql) | proposal sequence/validation, record status event |
| [`009_background_memory_patch_jobs.sql`](../../ubuntu/src/palmclaw_ubuntu/migrations/009_background_memory_patch_jobs.sql) | durable patch job, lease/retry, model-call linkage |
| [`010_tool_memory_retrieval.sql`](../../ubuntu/src/palmclaw_ubuntu/migrations/010_tool_memory_retrieval.sql) | Tool-memory embedding, routed retrieval run/candidate |

### 6.3 저장되는 audit 정보

- session, active session, turn 상태와 terminal reason
- user/assistant/tool message와 Tool call ID
- Tool argument, schema/side-effect/retry class, result/error
- Agent/Memory/Embedding model call의 token, latency, privacy metadata, cost
- summary/structured memory의 version, evidence, gate decision
- retrieval의 전체 candidate score, rank, 선택/제외 이유
- patch job/run/proposal, validation code, record lifecycle와 evidence
- evaluation run, case 입력/출력/지표와 artifact 위치

DB 읽기 API의 중심은
[`get_trace()`](../../ubuntu/src/palmclaw_ubuntu/storage.py#L3344),
standard memory detail/retrieval API, Tool-memory detail/retrieval API,
evaluation detail API다.

### 6.4 재시작 복구와 idempotency

[`recover_interrupted_execution()`](../../ubuntu/src/palmclaw_ubuntu/storage.py#L330)은
startup 때:

- running turn과 Tool call을 `interrupted`
- 미완료 consolidation을 복구 가능한 상태
- 만료된 running patch job을 retry 가능한 상태
- DB에 patch 적용이 commit됐으나 job 완료 checkpoint 전 종료된 경우 완료 복구
- running Tool-memory retrieval을 failed/empty

로 정리한다.

process-level [`RuntimeLease`](../../ubuntu/src/palmclaw_ubuntu/runtime_lock.py)
때문에 같은 data directory를 두 Runtime이 동시에 foreground runtime으로 열 수
없다. SQLite 자체는 WAL과 busy timeout을 사용하지만 이 lease가 모든 외부
process writer를 일반적으로 통제하는 보안 lock은 아니다.

검증 근거:

- [`test_storage.py`](../../ubuntu/tests/test_storage.py)
- [`test_runtime_safety.py`](../../ubuntu/tests/test_runtime_safety.py)
- [`test_background_memory.py`](../../ubuntu/tests/test_background_memory.py)

## 7. Context와 Skill

### 7.1 Context Builder

[`ContextBuilder`](../../ubuntu/src/palmclaw_ubuntu/context.py#L26)는 다음을
조립한다.

1. 고정 system policy
2. 검색된 standard memory
3. 검색된 compact Tool-schema memory
4. 선택된 Skill 본문
5. 최근 대화

각 memory/Skill component를 독립 budget으로 자른 뒤 전체 context budget을
확인한다. History는 단순 message tail이 아니라 **완전한 대화 단위**로 고른다.
Assistant Tool call과 대응하는 Tool result를 원자적으로 보존하고 orphan 또는
미완성 Tool chain은 제외한다. 현재 turn이 budget에 들어가지 않으면 조용히
삭제하지 않고 `ContextBudgetError`를 발생시킨다.

System policy는 검색 memory를 지시가 아닌 untrusted data로 취급하고, 등록된
Tool만 사용하며, file path는 workspace 안으로 제한하도록 Agent에 명시한다.
선택 Skill, token 사용량, drop 수는 context metadata와 model trace에서 확인할
수 있다.

검증:

- [`test_context.py`](../../ubuntu/tests/test_context.py)
- Tool memory만 compact하게 들어가는지:
  [`test_tool_memory_retrieval.py`](../../ubuntu/tests/test_tool_memory_retrieval.py#L405)

### 7.2 Skill loader

[`SkillsLoader`](../../ubuntu/src/palmclaw_ubuntu/skills.py#L19)는:

- built-in root와 workspace root의 `*/SKILL.md`만 읽는다.
- 같은 `name`이면 workspace Skill이 built-in을 덮어쓴다.
- symlink가 Skill root 밖으로 나가면 무시한다.
- frontmatter의 `name`, `description`, `always`를 읽는다.
- `always` Skill과 최신 user text의 token overlap으로 최대 3개를 선택한다.

기본 package에는
[`file-workspace/SKILL.md`](../../ubuntu/src/palmclaw_ubuntu/builtin_skills/file-workspace/SKILL.md)가
있다.

이 구현은 최소 PoC loader다. 임의 dependency, nested resource, executable
script, marketplace install, full Codex Skill routing 규칙을 지원하지 않는다.

검증: [`test_skills.py`](../../ubuntu/tests/test_skills.py)

## 8. Tool Registry와 실행 경계

### 8.1 Registry

[`ToolRegistry`](../../ubuntu/src/palmclaw_ubuntu/tools.py#L47)는 Tool 등록 시:

- 이름 중복 차단
- JSON Schema Draft 2020-12 schema 자체 검증
- 실행 전 argument schema 검증
- Tool timeout 적용
- Tool 결과의 최대 character 제한
- 예외를 구조화된 `ToolResult` error로 변환

을 수행한다. Runtime에 실제 등록되는 Tool은
[`application.py`](../../ubuntu/src/palmclaw_ubuntu/application.py#L129)에
있는 세 가지뿐이다.

### 8.2 `file_read`

[`FileReadTool`](../../ubuntu/src/palmclaw_ubuntu/tools.py#L186):

- 상대/session/shared 경로만 허용
- 일반 file만 읽음
- 기존 hard link(`st_nlink > 1`) 차단
- 최대 byte 검사
- UTF-8 text 반환

### 8.3 `file_write`

[`FileWriteTool`](../../ubuntu/src/palmclaw_ubuntu/tools.py#L251):

- content의 UTF-8 byte 크기 검사
- 기존 hard link 차단
- 기존 file은 `overwrite=true`가 있어야 변경
- 필요한 parent directory 생성
- unsafe side-effect Tool로 분류

제한: 현재 write는 temporary file + atomic replace가 아니라
`Path.write_text()`를 사용한다. resolve/check와 실제 write 사이의 TOCTOU를
제거한 OS sandbox도 아니다. 따라서 신뢰하지 않는 local process가 동시에
workspace를 조작하는 공격 모델까지 보장하지 않는다.

### 8.4 Workspace path

[`WorkspaceResolver`](../../ubuntu/src/palmclaw_ubuntu/workspace.py)는:

- 일반 상대 경로 → 현재 session root
- `session://...` → 현재 session root
- `shared://...` → shared root
- absolute path 거부
- canonical resolve 결과가 허용 root 밖이면 거부
- 다른 session ID를 직접 경로로 사용해 접근하는 것을 차단

해 `..` 및 symlink escape를 방어한다.

검증:

- [`test_workspace_tools.py`](../../ubuntu/tests/test_workspace_tools.py)
- schema validation:
  [`test_invalid_arguments_are_rejected_before_execution`](../../ubuntu/tests/test_workspace_tools.py#L94)

### 8.5 `web_fetch`

[`WebFetchTool`](../../ubuntu/src/palmclaw_ubuntu/tools.py#L358)은 다음으로
제한된다.

- HTTPS `GET` only
- URL credential 거부
- 최초 URL과 모든 redirect target 재검증
- DNS 응답 주소가 전부 public/global이어야 함
- 검증한 첫 IP에 TLS connection을 pin해 단순 DNS rebinding 완화
- redirect 수와 response byte 제한
- 임의 header, auth, request body 미지원

이 Tool은 browser가 아니며 JavaScript rendering, form submit, download
automation을 하지 않는다.

검증: [`test_web_tool.py`](../../ubuntu/tests/test_web_tool.py)

## 9. Provider contract와 OpenAI 호출

Protocol은 [`contracts.py`](../../ubuntu/src/palmclaw_ubuntu/contracts.py)에
분리되어 있다.

- `AgentModel`: message와 Tool definition → assistant response/Tool calls
- `MemoryModel`: summary 생성
- `StructuredMemoryModel`: typed memory candidate 생성
- `PatchMemoryModel`: Tool-memory patch batch 생성
- `EmbeddingModel`: 고정 dimension vector 생성

### 9.1 Foreground Agent

[`OpenAIResponsesAgentModel`](../../ubuntu/src/palmclaw_ubuntu/providers.py#L349)은
OpenAI Responses API를 사용하고 `store=False`를 설정한다. Tool을 function
schema로 넘기고, provider의 opaque continuation item과 encrypted reasoning
item을 다음 round에 재사용할 수 있게 DB message에 보존한다.

### 9.2 Cloud Memory

- [`OpenAIMemoryModel`](../../ubuntu/src/palmclaw_ubuntu/providers.py#L468):
  별도 memory system prompt로 compact summary를 생성한다.
- [`OpenAIStructuredMemoryModel`](../../ubuntu/src/palmclaw_ubuntu/providers.py#L570):
  Pydantic strict schema로 candidate/evidence를 parse한다.
- [`OpenAIPatchMemoryModel`](../../ubuntu/src/palmclaw_ubuntu/providers.py#L702):
  strict ADD/UPDATE/MERGE/DELETE payload를 parse한다.
- [`OpenAIEmbeddingModel`](../../ubuntu/src/palmclaw_ubuntu/providers.py#L930):
  float vector와 설정 dimension을 요청한다.

Model이 반환한 structured/patch object는 DB에 직접 쓰이지 않는다. Runtime의
evidence 검증과 validation gate, transaction engine을 거친다.

### 9.3 Local Memory

[`LocalMemoryModel`](../../ubuntu/src/palmclaw_ubuntu/providers.py#L863)과
[`LocalStructuredMemoryModel`](../../ubuntu/src/palmclaw_ubuntu/providers.py#L893)은
loopback OpenAI-compatible `/chat/completions`를 호출한다. `trust_env=False`,
temperature 0을 사용하고 structured 경로는 JSON object를 요구한다.

Local **Patch** model class는 현재 없다. 방법론의 최종 on-device SLM 목표는
[`tool-schema-on-device-memory-implementation-plan.md`](tool-schema-on-device-memory-implementation-plan.md)의
P6 이후 계획이며, 지금 완료된 P1–P5 PoC는 Cloud patch model 기준이다.

### 9.4 Fake/Scripted provider

`Echo`, `Fake`, `Scripted` provider는 deterministic test와 offline harness용이다.
Fake backend 결과를 실제 Cloud model 품질로 해석하면 안 된다.

## 10. Standard 장기 Memory

Standard memory는 `summary`와 `structured` 두 전략을 가진다. 구현 owner는
[`MemoryEngine`](../../ubuntu/src/palmclaw_ubuntu/memory.py#L46)이다.

### 10.1 Consolidation trigger와 watermark

Agent turn이 정상 완료되면 `MemoryEngine.consolidate()`를 호출한다. 마지막으로
완료한 message cursor 이후 user/assistant message가 기본 6개 이상 쌓였을 때
새 consolidation run을 시작한다.

실패는 foreground Agent 답변을 되돌리지 않는다. 실패 run과 error를 기록하고
미처리 message window를 다음 retry 대상으로 남긴다.

### 10.2 Summary 전략

Cloud/Local model에 이전 summary와 새 message window를 주고 하나의 compact
Markdown summary를 받는다. 매 consolidation마다 version이 증가하며 retrieval은
가장 최신 summary 전체를 반환한다.

Summary는 fact별 evidence/record retrieval을 하지 않는다. 길이를 context
budget에 맞게 자를 뿐이다.

### 10.3 Structured 전략

각 candidate의 주요 필드는:

- `subject`, `predicate`, `value`
- `scope`: `session` 또는 `global`
- `memory_type`, `confidence`, `sensitivity`
- source `message_id`와 exact evidence quote
- lifecycle status, version, `supersedes_id`

`fact_key`는 scope owner, 정규화한 subject/predicate로 만든다. value는 identity에
포함되지 않으므로 같은 사실의 correction은 immutable 새 version으로 저장하고
기존 active record를 supersede할 수 있다.

`working`은 현재 turn context 개념이며 장기 DB record scope가 아니다. 실제 장기
visibility는 session 또는 같은 local DB의 global이다.

### 10.4 Standard validation gate

[`MemoryValidationGate`](../../ubuntu/src/palmclaw_ubuntu/validation.py#L81)는:

- source message 존재와 evidence exact span
- candidate가 evidence에 지지되는지 token/substring 기반 확인
- confidence threshold
- secret/PII
- global scope의 높은 minimum confidence
- high sensitivity
- 기존 fact와 duplicate/contradiction
- explicit update 여부

를 기준으로 `verified`, `rejected`, `review`를 결정한다. PII, unresolved conflict,
high sensitivity는 자동 overwrite 대신 review queue로 보낼 수 있다.

이 gate는 독립 LLM judge나 NLI model이 아니라 **규칙 기반 heuristic**이다.
따라서 의미가 같은 paraphrase 또는 복잡한 부정을 완전히 판별한다고 보장하지
않는다.

### 10.5 Structured retrieval

지원 mode:

- `none`: memory 없음
- `full`: active visible memory를 token budget까지 넣는 baseline
- `bm25`: 내부 BM25 lexical score
- `embedding`: cosine similarity
- `hybrid`: BM25와 embedding score를 각각 정규화 후 결합

대상은 현재 session의 verified active memory와 global verified active memory다.
top-k, token budget을 모두 적용하며 embedding은 DB cache를 사용한다.
retrieval failure는 Agent turn을 실패시키지 않고 빈 memory와 failed trace를
남긴다.

각 retrieval은 query, candidate별 BM25/embedding/hybrid score, rank,
selected/excluded reason, context token, latency를 저장한다.

검증:

- evidence/version:
  [`test_memory_phase3.py`](../../ubuntu/tests/test_memory_phase3.py)
- gate/review/privacy:
  [`test_memory_phase4.py`](../../ubuntu/tests/test_memory_phase4.py)
- Agent context selection:
  [`test_agent_context_contains_only_retrieved_memory`](../../ubuntu/tests/test_memory_phase3.py#L426)

## 11. Tool-schema incremental Memory 방법론 구현

이 경로가 현재 연구 PoC의 핵심이다. Standard structured memory와 별도의
`tool_memory_*`, `memory_patch_*` table과 실행 경로를 사용한다.

### 11.1 Ontology: Tool schema를 저장 partition의 상한으로 사용

[`build_tool_memory_ontology()`](../../ubuntu/src/palmclaw_ubuntu/tool_memory_schema.py#L93)은
등록된 `ToolDefinition`에서 다음을 유도한다.

- namespace/domain
- action
- topic
- argument slot과 JSON type

Tool 이름과 schema identifier는 NFKC/casefold 기반으로 정규화한다. Patch
provider가 ontology에 없는 domain/topic을 임의 생성해 저장하는 것을 gate가
차단한다.

### 11.2 Record identity

[`normalize_tool_memory_identity()`](../../ubuntu/src/palmclaw_ubuntu/tool_memory_schema.py#L105)와
[`tool_memory_record_key()`](../../ubuntu/src/palmclaw_ubuntu/tool_memory_schema.py#L135)이
record identity를 만든다.

```text
user_id
+ tool_domain
+ topic
+ scope
+ scope_key
+ canonical conditions
→ SHA-256 record_key
```

`value`는 key에서 제외된다. 따라서 같은 사용자/Tool partition/scope/condition의
값이 바뀌면 별도 무관 record를 계속 추가하기보다 UPDATE version으로 처리할 수
있다.

지원 scope:

- `global`: 해당 `user_id` 전체
- `vehicle`: 특정 vehicle ID
- `session`: 특정 session ID
- `conditional`: 정규화 conditions가 질문 context와 맞을 때

Record에는 value JSON, memory type, confidence, status, version, evidence,
created/superseded/deleted lineage가 저장된다.

### 11.3 Patch extraction 입력

[`BackgroundMemoryPatchWorker`](../../ubuntu/src/palmclaw_ubuntu/background_memory.py#L37)는
job 하나마다:

- 완료된 source turn의 user/assistant message
- 고정 `user_id`
- 전체 Tool ontology
- 해당 사용자의 모든 active Tool-memory record
- patch policy/prompt version

을 `PatchMemoryModel`에 보낸다. 일반 Runtime은 한 완료 turn 단위,
VehicleMemBench는 history 한 줄 단위다.

현재 효율성 선택은 “foreground 추론을 막지 않고 idle-time batch로 추출”이다.
추출 비용 자체를 최소화한 구조는 아니며, Vehicle live run에서는 전체 ontology와
active record 반복 전송으로 input token이 컸다.

### 11.4 Patch operation

strict schema는
[`memory_patch.py`](../../ubuntu/src/palmclaw_ubuntu/memory_patch.py)에 있다.

| Operation | 의미 | DB 결과 |
|---|---|---|
| `ADD` | 같은 identity의 active record가 없을 때 생성 | version 1 active record |
| `UPDATE` | 명시적 correction | 기존 record superseded, 새 version active |
| `MERGE` | 동등한 중복 record 통합 | target들을 supersede하고 lineage를 가진 새 record |
| `DELETE` | 명시적 철회/무효화 | hard delete 없이 tombstone `deleted` |

[`ToolMemoryPatchEngine`](../../ubuntu/src/palmclaw_ubuntu/memory_patch.py#L191)은
batch 전체를 `BEGIN IMMEDIATE` transaction에서 적용한다. 중간 patch 하나가
실패하면 앞서 적용한 변경도 rollback하고 proposal rejection audit을 남긴다.
`run_id + sequence_index`가 proposal idempotency key이며, 완료 run 재실행은
동일 payload인지 확인한 뒤 기존 결과를 반환한다.

### 11.5 Patch validation

[`ToolMemoryPatchValidator`](../../ubuntu/src/palmclaw_ubuntu/validation.py#L302)는
다음을 검사한다.

- 모든 필드와 operation별 필수/금지 필드
- 고정 user와 ontology domain/topic
- scope owner와 `scope_key`
- target record가 active이고 같은 owner/domain/topic인지
- source message와 turn/session 일치
- exact contiguous evidence quote
- value의 모든 leaf가 evidence에 실제 등장하는지
- confidence threshold
- secret/PII
- duplicate/conflict
- UPDATE/DELETE를 지지하는 명시적 change marker

Model은 이 검사를 우회해 record table에 쓸 수 없다.

현재 알려진 처리 한계: validation에서 거절될 patch를 provider가 반복 반환하면
같은 job이 최대 attempt까지 재시도된 뒤 `failed`가 될 수 있다. “유효 memory가
없음”을 안전한 completed no-op로 분류하는 정책은 아직 개선 대상이다.

### 11.6 Durable background queue

성공 foreground turn마다 source turn unique job 하나를 enqueue한다. Job은
`queued → running → completed|retryable|failed` 상태, attempt, lease,
`next_attempt_at`, source message cursor를 가진다.

`palmclaw memory patch run`은 설정한 개수만 claim해 bounded cycle을 수행한다.
provider timeout/error는 attempt 한도 전까지 retryable로 남는다. process가
중단되면 lease와 DB commit 상태를 이용해 재개한다.

**Background라는 이름은 자동 daemon을 의미하지 않는다.** 현재 CLI process가
idle을 감지하거나 주기 실행하지 않는다. 외부 scheduler 또는 사용자의 명시적
`memory patch run`이 필요하다.

검증:

- patch operation/transaction:
  [`test_memory_patch.py`](../../ubuntu/tests/test_memory_patch.py)
- queue/retry/restart:
  [`test_background_memory.py`](../../ubuntu/tests/test_background_memory.py)
- schema/identity/persistence:
  [`test_tool_memory_schema.py`](../../ubuntu/tests/test_tool_memory_schema.py)

## 12. Tool-schema Memory routing과 추론

### 12.1 Route

[`ToolSchemaRouter`](../../ubuntu/src/palmclaw_ubuntu/memory_router.py#L94)는
질문 token과 ontology의 domain/topic/action/slot/tool token 및 일부
한·영 alias를 비교한다. 가중치는 domain 6, topic 5, slot 5, action 3,
일반 tool token 1이다.

상위 route의 score가 없으면 route를 만들지 않는다. 2위가 1위의 80% 이상이면
ambiguous로 표시할 수 있다. `ToolMemoryRouteClassifier` protocol과
missing/ambiguous 시 classifier fallback 경로는 존재하지만
[`application.py`](../../ubuntu/src/palmclaw_ubuntu/application.py#L268)는
classifier를 주입하지 않는다. 현재 일반 Runtime과 Vehicle PoC의 실동작은
deterministic lexical routing이다.

### 12.2 Scope와 condition filter

[`ToolMemoryRetriever`](../../ubuntu/src/palmclaw_ubuntu/memory_router.py#L233)는:

1. route된 domain/topic에 속한 fixed user의 active record만 SQL 조회
2. global/session/vehicle owner 일치 확인
3. conditional record의 모든 condition leaf가 query와 condition context에
   나타나는지 확인
4. full/BM25/embedding/hybrid ranking
5. top-k와 token budget 적용

순으로 처리한다.

조건 판정도 lexical leaf match다. 의미적 동의어, 상식적 조건, 복잡한
coreference를 일반적으로 해결하지 않는다.

### 12.3 Context에 주입되는 정보

선택된 record만 compact text로 만든다.

```text
record id, domain, topic, scope, conditions,
version, confidence, value
```

Evidence 원문, 제외 record, 다른 partition의 record는 Agent prompt에 넣지 않는다.
route가 없거나 embedding/provider/DB retrieval이 실패하면 임의 full-memory
fallback을 하지 않고 빈 Tool memory를 반환한다. 실패/empty 이유는 trace에
남긴다.

이것이 “memory update는 idle-time에 충분히 수행하되 실제 Tool 추론에서는
관련 record top-k만 넣어 효율화한다”는 현재 가설의 코드화다.

검증: [`test_tool_memory_retrieval.py`](../../ubuntu/tests/test_tool_memory_retrieval.py)

## 13. 개인정보와 secret 처리

구현 owner는 [`privacy.py`](../../ubuntu/src/palmclaw_ubuntu/privacy.py)다.

### 13.1 탐지 대상

- OpenAI 형태 API key와 bearer token
- common secret assignment
- email
- 일부 정부 ID 형식
- Luhn-valid credit card
- 휴대전화 형태
- 일부 주소 형태
- 중첩 object의 민감한 key 이름

`PALMCLAW_PII_ALLOWLIST`의 exact string은 PII redaction 예외로 둘 수 있다.

### 13.2 적용 지점

- user message DB 저장 전: `redact_secrets()`로 secret 제거
- Cloud Agent/Memory/Patch/Embedding 호출 직전:
  `redact_data_for_cloud()`로 설정에 따라 PII와 secret 제거
- evaluation artifact 작성 전: 중첩 payload redaction과 잔여 privacy audit
- model call trace: 전송 character, 탐지/redaction 수, category, 전후
  sensitive character 수

`PALMCLAW_LOCAL_PII_STORAGE=redacted|raw`는 standard structured memory의
local 저장 정책이다. Cloud 전송 redaction과 local storage 정책은 별개다.

### 13.3 보장하지 않는 것

탐지는 regex와 checksum 중심 heuristic이다. 임의 자연어 속 이름, 간접 식별자,
모든 국가의 주소/ID, 난독화된 credential을 완전하게 찾지 못할 수 있다.
privacy artifact의 residual 0은 구현 detector가 찾은 잔여가 0이라는 뜻이지,
실제 개인정보가 절대 없다는 formal 보장은 아니다.

검증:

- [`test_privacy.py`](../../ubuntu/tests/test_privacy.py)
- Cloud 호출 직전 redaction:
  [`test_providers.py`](../../ubuntu/tests/test_providers.py#L408)
- evaluation artifact:
  [`test_vehicle_bench.py`](../../ubuntu/tests/test_vehicle_bench.py#L413)

## 14. CLI 기능

parser의 최종 근거는
[`build_parser()`](../../ubuntu/src/palmclaw_ubuntu/cli.py#L17)다.

### 14.1 Runtime과 session

```bash
palmclaw doctor
palmclaw session new --title "PoC"
palmclaw session list
palmclaw session use SESSION_ID
palmclaw session show [SESSION_ID]
palmclaw ask [--session SESSION_ID] "질문"
palmclaw chat [--session SESSION_ID]
palmclaw history [--session SESSION_ID] [--limit 100]
palmclaw trace TURN_ID
```

### 14.2 Standard memory

```bash
palmclaw memory show [--session ID] [--all-versions]
palmclaw memory inspect MEMORY_ID
palmclaw memory versions FACT_KEY
palmclaw memory search [--mode MODE] [--top-k K] "query"
palmclaw memory compare [--top-k K] "query"
palmclaw memory retrievals [--session ID]
palmclaw memory retrieval-trace RUN_ID
palmclaw memory review [--session ID]
palmclaw memory resolve MEMORY_ID --decision accept|reject [--reason TEXT]
palmclaw memory backend-report [--session ID]
```

### 14.3 Tool-schema memory

```bash
palmclaw memory patch pending [--session ID]
palmclaw memory patch run [--session ID] [--limit N]
palmclaw memory patch trace RUN_ID
palmclaw memory tool search [--mode MODE] [--top-k K] "query"
palmclaw memory tool retrievals [--session ID]
palmclaw memory tool trace RUN_ID
```

Patch 실행에는 `PALMCLAW_PATCH_MEMORY_ENABLED=1`, Tool retrieval에는
`PALMCLAW_TOOL_MEMORY_RETRIEVAL_ENABLED=1`이 필요하다. Patch를 켜면 후자는
명시하지 않았을 때 기본적으로 함께 켜진다.

### 14.4 검사와 평가

```bash
palmclaw skill list
palmclaw tool list
palmclaw eval profiles
palmclaw eval run ...
palmclaw eval vehicle ...
palmclaw eval list
palmclaw eval show RUN_ID
```

CLI는 `KeyError`, `RuntimeError`, `TimeoutError`, `ValueError`를 사용자 오류로
출력하고 exit code 2, `KeyboardInterrupt`는 130을 반환한다.

## 15. Generic synthetic Memory 평가

평가 구현은
[`evaluation.py`](../../ubuntu/src/palmclaw_ubuntu/evaluation.py),
기본 dataset은
[`synthetic_memory_v1.json`](../../ubuntu/src/palmclaw_ubuntu/evaluation_datasets/synthetic_memory_v1.json)이다.

### 15.1 Profile

- `no_memory`
- `cloud_summary`
- `cloud_structured_bm25`
- `cloud_structured_embedding`
- `local_structured_bm25`
- `local_structured_embedding`
- `local_full`
- `ablation_no_retrieval`
- `ablation_no_gate`
- `ablation_no_redaction`

### 15.2 Offline과 live

- `offline`: logical cloud/local profile도 case별 deterministic fixture
  MemoryModel과 Fake embedding을 사용한다. Harness, schema, metric,
  ablation 방향을 확인하는 용도이며 모델 품질 평가가 아니다.
- `live`: memory-enabled profile의 logical backend에 따라 실제 OpenAI 또는
  local MemoryModel을 조립한다. Foreground Agent는 계속 fake이며, 이 generic
  harness는 Agent Tool-calling 품질보다 memory extraction/retrieval/gate를
  평가한다.

### 15.3 지표와 artifact

task success, schema/value-level memory F1, unsupported memory, gate decision,
conflict/overwrite, Recall@k, MRR, nDCG, irrelevant injection, PII F1,
Cloud 노출, latency, token, 설정된 단가 기준 cost를 계산한다.

각 run은 SQLite에 저장되고 `metrics.json`, redacted `cases.jsonl`,
`results.tsv`, `results.md`, SVG graph를 생성한다.

검증: [`test_evaluation.py`](../../ubuntu/tests/test_evaluation.py)

## 16. VehicleMemBench 통합

Vehicle 코드는 일반 Runtime에서 분리된
[`vehicle_bench/`](../../ubuntu/src/palmclaw_ubuntu/vehicle_bench/) package다.

### 16.1 Dataset loader와 pin

[`dataset.py`](../../ubuntu/src/palmclaw_ubuntu/vehicle_bench/dataset.py)는
기본적으로 upstream commit
`5ef3c48a4dbb446e6bb84a91dcc3632e9b1d203b`를 요구한다.

Loader는 전체 checkout에서:

- 50 scenario history/QA pair
- 500 task
- 111 Tool JSON schema
- gold Tool call과 argument
- fixture와 schema digest

를 검증한다. Gold call parsing은 Python `eval()`이 아니라 `ast`와
`literal_eval()`을 사용한다.

### 16.2 공식 simulator와 scorer

[`VehicleWorldRuntime`](../../ubuntu/src/palmclaw_ubuntu/vehicle_bench/scoring.py)은
trusted checkout의 VehicleWorld와 upstream `eval_utils.py`를 import해 공식
state/Tool scoring 함수를 호출한다.

이 namespace 격리는 일반 Runtime namespace 오염을 줄이기 위한 것이지
untrusted Python sandbox가 아니다. `--allow-unpinned`는 adapter 개발용이며
논문/보고 평가에는 pinned trusted checkout만 사용해야 한다.

### 16.3 평가 전용 Agent 격리

[`VehicleAgentLoop`](../../ubuntu/src/palmclaw_ubuntu/vehicle_bench/agent.py)은
일반 session/Skill/file/web/standard memory를 사용하지 않는다.

- 최초 제공 Tool은 `list_module_tools` 하나
- Agent가 module을 요청하면 해당 Vehicle Tool을 다음 round부터 동적 등록
- 각 task마다 fresh initial/reference/predicted VehicleWorld 생성
- task query는 memory snapshot을 갱신하지 않음
- host file, web, shell Tool 없음

Profile별 prompt source:

- `no_memory`: query만
- `gold_memory`: benchmark gold memory + query
- `cloud_summary`: history에서 Cloud로 만든 latest summary + query
- `cloud_structured_bm25|embedding|hybrid`: history에서 만든 structured
  record의 해당 retrieval 결과 + query
- `cloud_schema_patch`: history turn별 patch record의 routed hybrid top-k + query

Gold memory와 retrieved memory를 동시에 넣지 않는다.

### 16.4 History snapshot과 cache

[`VehicleMemoryBuilder`](../../ubuntu/src/palmclaw_ubuntu/vehicle_bench/memory.py#L404)는:

- timestamp/speaker를 보존해 history를 parse
- summary/structured는 날짜 경계를 우선한 token batch로 consolidation
- schema patch는 history 한 줄을 immutable completed turn 하나로 replay
- dataset/history/model/prompt/schema/budget 설정을 hash한 cache directory 생성
- `manifest.json`과 `memory.db`로 중단 지점 재개
- 모든 요청 task 전에 memory snapshot을 완성·고정
- retrieval 전후 fingerprint를 비교해 task가 memory를 변경하지 않았는지 검사

같은 cache key를 재사용하면 완료된 Cloud memory generation을 다시 호출하지
않는다.

### 16.5 Vehicle 평가 실행과 resume

[`runner.py`](../../ubuntu/src/palmclaw_ubuntu/vehicle_bench/runner.py)는 task별
model/Tool trace와 공식 scorer input/output을 기록한다.
[`suite.py`](../../ubuntu/src/palmclaw_ubuntu/vehicle_bench/suite.py)는 여러
scenario를 묶고 child checkpoint를 재개하며 SQLite evaluation table과 report를
완료한다.

```bash
# Gold call adapter/scorer smoke
palmclaw eval vehicle \
  --benchmark-root /trusted/VehicleMemBench \
  --scenario 1 --task-limit 10

# 실제 Agent + schema patch memory
palmclaw eval vehicle \
  --benchmark-root /trusted/VehicleMemBench \
  --mode live \
  --profiles cloud_schema_patch \
  --scenario 1 --scenario-limit 1 --task-limit 10
```

`--resume-run`, `--memory-cache-dir`, `--memory-batch-tokens`,
`--memory-top-k`, `--memory-token-budget`, model override를 지원한다.

### 16.6 Vehicle 지표

- **Exact State Match (ESM)**: predicted final VehicleWorld가 reference와
  state difference/false positive 없이 정확히 같은 task 비율
- **State F1**: 변경되어야 할 positive state를 얼마나 정확히 변경했는지의 F1
- **Value F1**: state 변화의 값까지 맞춘 정도
- **Tool F1**: predicted Tool 이름 sequence/set과 reference Tool의 precision/recall F1
- **Argument exact match**: Tool 이름, 개수, 각 JSON argument가 전부 정확한 task 비율
- **Retrieval Recall@k**: gold Tool argument에 필요한 값이 선택 memory에 포함된 비율
- completion, latency, token, model/Tool call 수
- retrieval empty/mismatch 등 failure taxonomy
- patch proposal acceptance, validation code, version/evidence 지표
- Cloud 전송/redaction과 설정 단가 기준 추정 비용

Tool F1이 높아도 argument가 틀리면 state가 틀릴 수 있다. 따라서 ESM, State F1,
Tool F1, argument exact를 함께 봐야 한다.

### 16.7 현재 `cloud_schema_patch` live gate

실행 artifact:
[`9128b98f-107d-4124-951f-197efcaeb75d`](../../ubuntu/evaluation/vehiclemembench/9128b98f-107d-4124-951f-197efcaeb75d/)

조건:

- pinned 공식 scenario 1
- history 2,438 turn
- task 10개
- `cloud_schema_patch`

핵심 결과:

| 지표 | 값 |
|---|---:|
| 완료 task | 10/10 |
| ESM | 0.300 |
| State F1 | 0.350 |
| Tool F1 | 0.367 |
| Argument exact | 0.300 |
| Retrieval Recall@k | 0.000 |
| Retrieval empty | 8 task |
| Active patch record | 19 |
| Patch job | completed 2,435 / failed 3 |
| Proposal | applied 19 / rejected 20 |
| Patch model call | 2,456 |
| Patch input/output token | 24,675,484 / 103,602 |

Artifact privacy scan에서 구현 detector 기준 잔여 sensitive span은 0이었다.
비용 단가 환경변수를 설정하지 않아 USD cost는 0으로 기록됐으며 “무료”를
뜻하지 않는다.

이 결과는 실행 경로가 완성됐음을 보이지만 현재 방법의 성능 성공을 보이지는
않는다. 주된 병목은 query 표현과 Tool schema lexical term이 다를 때 route가
비거나 틀리는 현상이다. 5-scenario 확대 전에 semantic routing, invalid patch의
safe no-op, 반복 ontology token 절감이 남아 있다.

검증:

- loader/scorer/agent/suite:
  [`test_vehicle_bench.py`](../../ubuntu/tests/test_vehicle_bench.py)
- memory cache/schema patch isolation:
  [`test_vehicle_memory.py`](../../ubuntu/tests/test_vehicle_memory.py)
- 상세 P1–P5 기록:
  [`tool-schema-on-device-memory-implementation-plan.md`](tool-schema-on-device-memory-implementation-plan.md)

## 17. 추적성과 질문 시 확인 위치

| 질문 | 먼저 볼 위치 |
|---|---|
| “한 turn에서 무엇이 일어났나?” | `palmclaw trace TURN_ID`, `agent.py`, `storage.get_trace()` |
| “왜 이 memory가 저장됐나?” | `memory inspect`, evidence, gate decision, status event |
| “기존 memory가 어떻게 바뀌었나?” | `memory versions FACT_KEY`, `supersedes_id` |
| “왜 관련 memory가 검색되지 않았나?” | `memory retrieval-trace` 또는 `memory tool trace` |
| “Patch가 왜 거절/실패했나?” | `memory patch trace`, proposal `validation_code` |
| “어떤 model에 무엇이 전송됐나?” | model call role/metadata/privacy/usage |
| “Tool이 왜 실행되지 않았나?” | turn trace의 schema error, fingerprint, timeout |
| “재시작 후 무엇을 복구했나?” | Runtime `recovery`, turn/job/run status |
| “평가 수치는 어디서 나왔나?” | run `metrics.json`, `cases.jsonl`, `manifest.json` |
| “현재 기능이 테스트됐나?” | 아래 기능-테스트 matrix |

## 18. 기능-소스-테스트 검증 Matrix

| 기능 | Source owner | 자동 검증 |
|---|---|---|
| Agent loop/Tool round | `agent.py` | `test_agent_loop_e2e.py`, `test_runtime_safety.py` |
| Session/trace/migration | `storage.py`, migrations | `test_storage.py`, `test_cli.py` |
| 동시성/lock/recovery | `coordinator.py`, `runtime_lock.py`, `storage.py` | `test_runtime_safety.py` |
| Context atomic history/budget | `context.py` | `test_context.py` |
| Skill load/override/path | `skills.py` | `test_skills.py` |
| Tool schema/file boundary | `tools.py`, `workspace.py` | `test_workspace_tools.py` |
| HTTPS/DNS/redirect/size | `tools.py` | `test_web_tool.py` |
| OpenAI/Local/Fake provider | `providers.py` | `test_providers.py`, `test_config.py` |
| summary/structured memory | `memory.py` | `test_memory_phase3.py` |
| gate/review/privacy | `validation.py`, `privacy.py` | `test_memory_phase4.py`, `test_privacy.py` |
| Tool ontology/identity | `tool_memory_schema.py` | `test_tool_memory_schema.py` |
| Patch schema/transaction | `memory_patch.py`, `validation.py` | `test_memory_patch.py` |
| Background queue/retry | `background_memory.py`, `storage.py` | `test_background_memory.py` |
| Routed Tool retrieval | `memory_router.py` | `test_tool_memory_retrieval.py` |
| Generic evaluation | `evaluation.py` | `test_evaluation.py` |
| Vehicle adapter/scorer/report | `vehicle_bench/*` | `test_vehicle_bench.py`, `test_vehicle_memory.py` |

## 19. 명시적 미구현·위험·해석 제한

향후 질문에서 다음을 이미 구현된 기능으로 전제하면 안 된다.

1. **Android port 없음**: Ubuntu code는 Kotlin 앱에 배포되지 않았다.
2. **Local patch SLM 없음**: Local summary/structured provider만 있다.
3. **자동 background service 없음**: patch queue 처리 명령 또는 외부 scheduler가
   필요하다.
4. **범용 automation 없음**: shell, desktop automation, browser, cron Tool이
   일반 Runtime에 없다.
5. **semantic/classifier routing 없음**: current production assembly는 lexical
   router만 쓴다.
6. **완전한 PII 검출 없음**: 규칙 detector 범위만 보호한다.
7. **독립 semantic verifier 없음**: memory gate는 evidence exact span과
   heuristic entailment 중심이다.
8. **thread 강제 종료 없음**: timeout은 Runtime 상태를 끝내지만 underlying
   blocking thread 즉시 종료를 보장하지 않는다.
9. **file write atomicity 없음**: hostile local race까지 막는 sandbox가 아니다.
10. **Vehicle checkout sandbox 없음**: trusted pinned upstream Python을 실제
    실행한다.
11. **현재 schema-patch 성능 미달**: 1-scenario live gate Recall@k가 0이므로
    성능 주장은 금지한다.
12. **비용 자동 가격 조회 없음**: 사용자가 단가 환경변수를 넣을 때만 USD를
    계산한다.

## 20. 로컬 재현과 문서 갱신 절차

환경:

```bash
source /mnt/data/miniconda3/bin/activate
conda activate /home/hj153lee/PalmClaw/.conda/ubuntu-agent
python -m pip install -e "/home/hj153lee/PalmClaw/ubuntu[dev]"
```

기본 검증:

```bash
cd /home/hj153lee/PalmClaw/ubuntu
pytest -q
ruff check src tests
```

CLI smoke:

```bash
PALMCLAW_BACKEND=fake palmclaw doctor
PALMCLAW_BACKEND=fake palmclaw session new --title "Reference smoke"
PALMCLAW_BACKEND=fake palmclaw ask "hello"
PALMCLAW_BACKEND=fake palmclaw history
```

문서를 갱신해야 하는 변경:

- Settings field/default/environment variable
- Tool definition/schema/security boundary
- DB migration, lifecycle/status
- context source 또는 budget 정책
- provider request/response contract
- memory identity, patch validation, retrieval score
- evaluation profile, scorer, artifact schema
- 이 문서의 capability/limit 상태

갱신 시에는 source symbol link, 관련 test, 필요한 경우 새 실행 artifact를 함께
변경한다. 계획 문서나 한 번의 실험 결과만으로 기능을 `구현 완료`로 바꾸지 않는다.

## 21. Source module 색인

향후 코드 질문에서 owner를 빠르게 찾기 위한 전체 Python module 색인이다.

### 21.1 일반 Runtime

| Module | 책임 |
|---|---|
| [`__init__.py`](../../ubuntu/src/palmclaw_ubuntu/__init__.py) | package metadata |
| [`__main__.py`](../../ubuntu/src/palmclaw_ubuntu/__main__.py) | `python -m palmclaw_ubuntu` 진입점 |
| [`config.py`](../../ubuntu/src/palmclaw_ubuntu/config.py) | 환경변수, default, 설정 검증 |
| [`models.py`](../../ubuntu/src/palmclaw_ubuntu/models.py) | message, Tool, memory, patch, embedding 관련 dataclass |
| [`contracts.py`](../../ubuntu/src/palmclaw_ubuntu/contracts.py) | Agent/Memory/Patch/Embedding provider protocol |
| [`application.py`](../../ubuntu/src/palmclaw_ubuntu/application.py) | composition root와 Runtime lifecycle |
| [`agent.py`](../../ubuntu/src/palmclaw_ubuntu/agent.py) | foreground Agent loop와 trace |
| [`coordinator.py`](../../ubuntu/src/palmclaw_ubuntu/coordinator.py) | session 직렬화, global concurrency, patch enqueue |
| [`runtime_lock.py`](../../ubuntu/src/palmclaw_ubuntu/runtime_lock.py) | data-dir process lease |
| [`context.py`](../../ubuntu/src/palmclaw_ubuntu/context.py) | system/memory/Skill/history context 조립과 budget |
| [`skills.py`](../../ubuntu/src/palmclaw_ubuntu/skills.py) | 최소 `SKILL.md` load/select |
| [`tokens.py`](../../ubuntu/src/palmclaw_ubuntu/tokens.py) | tiktoken 기반 token count/truncate |
| [`tools.py`](../../ubuntu/src/palmclaw_ubuntu/tools.py) | Registry, file read/write, HTTPS fetch |
| [`workspace.py`](../../ubuntu/src/palmclaw_ubuntu/workspace.py) | session/shared path resolution |
| [`providers.py`](../../ubuntu/src/palmclaw_ubuntu/providers.py) | OpenAI, Local, Fake, Scripted model adapter |
| [`storage.py`](../../ubuntu/src/palmclaw_ubuntu/storage.py) | SQLite repository, migration, recovery, audit query |
| [`privacy.py`](../../ubuntu/src/palmclaw_ubuntu/privacy.py) | secret/PII 탐지·redaction·통계 |
| [`validation.py`](../../ubuntu/src/palmclaw_ubuntu/validation.py) | standard memory gate와 Tool-memory patch validator |
| [`memory.py`](../../ubuntu/src/palmclaw_ubuntu/memory.py) | summary/structured consolidation과 standard retrieval |
| [`tool_memory_schema.py`](../../ubuntu/src/palmclaw_ubuntu/tool_memory_schema.py) | Tool ontology, record identity/key |
| [`memory_patch.py`](../../ubuntu/src/palmclaw_ubuntu/memory_patch.py) | strict patch parser와 transactional operation engine |
| [`background_memory.py`](../../ubuntu/src/palmclaw_ubuntu/background_memory.py) | durable patch queue worker |
| [`memory_router.py`](../../ubuntu/src/palmclaw_ubuntu/memory_router.py) | schema route, scope/condition filter, ranked Tool retrieval |
| [`evaluation.py`](../../ubuntu/src/palmclaw_ubuntu/evaluation.py) | generic synthetic memory evaluation |
| [`cli.py`](../../ubuntu/src/palmclaw_ubuntu/cli.py) | 모든 CLI parser/handler |

### 21.2 VehicleMemBench

| Module | 책임 |
|---|---|
| [`vehicle_bench/__init__.py`](../../ubuntu/src/palmclaw_ubuntu/vehicle_bench/__init__.py) | public Vehicle adapter export |
| [`vehicle_bench/dataset.py`](../../ubuntu/src/palmclaw_ubuntu/vehicle_bench/dataset.py) | pinned checkout, dataset/schema/gold call 검증 |
| [`vehicle_bench/tools.py`](../../ubuntu/src/palmclaw_ubuntu/vehicle_bench/tools.py) | 111 simulator Tool adapter와 dynamic discovery |
| [`vehicle_bench/agent.py`](../../ubuntu/src/palmclaw_ubuntu/vehicle_bench/agent.py) | 평가 전용 격리 Agent loop |
| [`vehicle_bench/memory.py`](../../ubuntu/src/palmclaw_ubuntu/vehicle_bench/memory.py) | history parse/batch, Cloud memory cache/snapshot |
| [`vehicle_bench/scoring.py`](../../ubuntu/src/palmclaw_ubuntu/vehicle_bench/scoring.py) | VehicleWorld 실행과 공식 scorer adapter |
| [`vehicle_bench/runner.py`](../../ubuntu/src/palmclaw_ubuntu/vehicle_bench/runner.py) | task 실행, metric, checkpoint, artifact |
| [`vehicle_bench/suite.py`](../../ubuntu/src/palmclaw_ubuntu/vehicle_bench/suite.py) | multi-scenario orchestration, resume, usage/cost 집계 |

### 21.3 Packaged data

| 경로 | 책임 |
|---|---|
| [`builtin_skills/`](../../ubuntu/src/palmclaw_ubuntu/builtin_skills/) | Runtime built-in Skill |
| [`evaluation_datasets/`](../../ubuntu/src/palmclaw_ubuntu/evaluation_datasets/) | versioned generic synthetic dataset |
| [`migrations/`](../../ubuntu/src/palmclaw_ubuntu/migrations/) | SQLite schema 001–010 |
| [`tests/`](../../ubuntu/tests/) | 122개 수집 test의 전체 source |
