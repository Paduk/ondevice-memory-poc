# PalmClaw Ubuntu Agent Runtime PoC 계획

상태: `Phase 5 implementation complete — Offline reference and OpenAI live evaluation complete`

최종 검토: 2026-07-24

## 1. 목적

이 문서는 PalmClaw의 Android Agent Runtime에서 플랫폼 독립적인 설계와 동작을 추출하여, Ubuntu에서 실행되는 CLI 기반 연구용 PoC를 구현하기 위한 계획이다.

PoC의 핵심 목표는 다음과 같다.

- CLI에서 세션을 생성·선택하고 대화를 실행한다.
- Cloud LLM이 구조화된 Tool call을 생성하고, 런타임이 검증·실행·저장한 결과를 다음 LLM 호출에 반영한다.
- 대화, Tool trace, 장기 메모리와 세션별 history를 프로세스 재시작 이후에도 복구한다.
- `SKILL.md`, 메모리, 시스템 지침과 최근 대화를 토큰 예산 안에서 조립한다.
- Phase 1에서는 Cloud `MemoryModel`로 consolidation을 구현하고, Phase 4에서 동일 계약의 로컬 SLM 구현을 추가한다.
- top-k 검색, 충돌·환각 검증, 개인정보 redaction과 메모리 품질 평가를 실험할 수 있게 한다.

이 PoC는 Android 앱을 Linux UI 애플리케이션으로 그대로 옮기는 작업이 아니다. PalmClaw의 Agent Loop와 계약을 기준으로 별도의 headless runtime을 구축하는 작업이다.

## 2. 완료 상태의 사용자 시나리오

최종 PoC는 다음 흐름을 재현해야 한다.

1. 사용자가 CLI에서 새 세션을 생성하거나 기존 세션을 선택한다.
2. 사용자의 입력이 SQLite에 먼저 저장된다.
3. 런타임이 관련 장기 메모리, 활성 Skill, 최근 대화를 선택해 context를 만든다.
4. Cloud LLM이 응답 또는 Tool call을 반환한다.
5. Tool call은 schema와 정책 검증을 통과한 경우에만 실행된다.
6. Tool 결과와 trace가 저장되고 다음 LLM 호출에 반영된다.
7. Tool call이 더 이상 없거나 최대 round에 도달하면 turn이 종료된다.
8. consolidation 조건이 충족되면 설정된 `MemoryModel`이 미처리 대화에서 메모리 또는 메모리 후보를 생성한다.
9. 개인정보·evidence·충돌 검증을 통과한 후보만 versioned memory로 반영된다.
10. 프로세스를 종료하고 다시 시작해도 세션, history, Tool trace와 메모리가 복구된다.

## 3. 범위

### 3.1 PalmClaw에서 이식할 기능

| 영역 | Ubuntu PoC 처리 | 근거가 되는 현재 소스 |
| --- | --- | --- |
| Agent model/tool 반복 | 제어 흐름과 종료 조건을 재현 | [`AgentLoop.kt`](../../app/src/main/java/com/palmclaw/agent/AgentLoop.kt) |
| Provider 추상화 | 정규화된 message, Tool spec, usage, error 계약 유지 | [`LlmProvider.kt`](../../app/src/main/java/com/palmclaw/providers/LlmProvider.kt) |
| Context 조립 | system, runtime, memory, Skill, history의 명시적 조립 | [`ContextBuilder.kt`](../../app/src/main/java/com/palmclaw/agent/ContextBuilder.kt) |
| Tool 등록과 실행 | 등록, JSON argument 검증, timeout, 구조화된 실패 유지 | [`ToolRegistry.kt`](../../app/src/main/java/com/palmclaw/tools/ToolRegistry.kt) |
| Skill 로딩과 선택 | Android assets 대신 파일시스템 catalog 사용 | [`SkillsLoader.kt`](../../app/src/main/java/com/palmclaw/skills/SkillsLoader.kt) |
| 메모리 consolidation | threshold 기반 background 작업 개념 유지 | [`MemoryConsolidator.kt`](../../app/src/main/java/com/palmclaw/agent/MemoryConsolidator.kt) |
| Workspace 경계 | session/shared 경로와 canonical path 검증 유지 | [`WorkspacePathResolver.kt`](../../app/src/main/java/com/palmclaw/workspace/WorkspacePathResolver.kt) |
| 세션별 turn 직렬화 | 동일 세션은 직렬화하고 서로 다른 세션은 제한적 병렬 실행 | [`SessionTurnCoordinator.kt`](../../app/src/main/java/com/palmclaw/runtime/SessionTurnCoordinator.kt) |

### 3.2 Ubuntu에서 교체할 기능

| Android 구현 | Ubuntu PoC 구현 |
| --- | --- |
| Jetpack Compose | 대화형 CLI와 비대화형 명령 |
| Room | SQLite repository와 명시적 migration |
| Android `Context`와 assets | 설정 가능한 Linux 디렉터리와 package resources |
| SharedPreferences/Android secure storage | 환경 변수, 권한 제한 설정 파일, secret reference |
| WorkManager/AlarmManager | 런타임 job queue와 Linux scheduler adapter |
| Android permission dialog | Tool policy와 CLI 승인 프롬프트 |
| `android.util.Log` | 구조화된 표준 logging |
| Foreground Service | CLI process, 선택적 systemd service |

### 3.3 초기 범위에서 제외할 기능

- Android Compose UI 이식
- Bluetooth, 연락처, Android 알림, 기기 설정과 미디어 제어
- 모바일 attachment URI와 Android document provider
- Telegram, Feishu 등 원격 channel adapter
- 임의 shell 명령 실행
- 다중 사용자 서버와 네트워크 공개 API
- 분산 vector database와 production 규모의 clustering
- 완전한 OS sandbox 구현

원격 channel과 daemon 운영은 CLI PoC가 안정화된 이후 별도 단계로 다룬다.

## 4. 기술 방향

### 4.1 구현 언어

연구용 PoC는 Python 3.12를 기본안으로 한다.

선택 이유:

- 로컬 SLM, embedding, NLI, PII 탐지와 평가 도구를 쉽게 교체할 수 있다.
- 메모리 pipeline의 ablation과 실험 자동화가 용이하다.
- PalmClaw의 Kotlin 소스를 직접 복사하기보다, 공개된 runtime 계약과 검증된 동작을 독립 인터페이스로 재현하기 적합하다.

예상 디렉터리 구조:

```text
ubuntu/
├── pyproject.toml
├── src/palmclaw_ubuntu/
│   ├── cli/
│   ├── agent/
│   ├── providers/
│   ├── tools/
│   ├── skills/
│   ├── context/
│   ├── memory/
│   ├── privacy/
│   ├── storage/
│   ├── workspace/
│   └── evaluation/
├── migrations/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── security/
│   └── e2e/
└── README.md
```

Kotlin/JVM 구현이 추후 필요해지더라도 Provider, Tool, Storage, Memory 계약과 SQLite schema는 언어에 종속되지 않게 문서화한다.

### 4.2 주요 구성 요소

```text
CLI
 └─ Application Service
     ├─ Session Service
     └─ Session Turn Coordinator
         └─ Agent Loop
             ├─ Context Builder
             │   ├─ System Policy
             │   ├─ Retrieved Memory
             │   ├─ Selected Skills
             │   └─ Recent History
             ├─ Cloud AgentModel
             └─ Tool Registry
                 ├─ Argument Validator
                 ├─ Tool Policy Engine
                 └─ Tool Executors

SQLite
 ├─ Conversation and trace
 ├─ Turn lifecycle
 ├─ Memory and evidence
 └─ Consolidation and evaluation

Memory Pipeline
 ├─ MemoryModel
 │   ├─ Cloud MemoryModel (Phase 1 baseline)
 │   └─ Local SLM MemoryModel (Phase 4)
 ├─ Memory candidate extraction
 ├─ Conflict and evidence gate
 └─ PII and secret redaction
```

각 구성 요소는 protocol/interface 뒤에 두어 fake 구현으로 단위 테스트할 수 있어야 한다.

## 5. CLI와 세션

### 5.1 계획 명령

```text
palmclaw session new [--title TITLE]
palmclaw session list
palmclaw session use SESSION_ID
palmclaw session show [SESSION_ID]
palmclaw chat [--session SESSION_ID]
palmclaw ask [--session SESSION_ID] "message"
palmclaw history [--session SESSION_ID]
palmclaw trace TURN_ID
palmclaw memory list [--scope global|session]
palmclaw memory review
palmclaw skill list
palmclaw tool list
palmclaw doctor
```

`chat`은 대화형 REPL이고, `ask`는 스크립트에서 사용할 수 있는 단일 turn 명령이다. 출력 형식은 사람이 읽는 기본 형식과 자동화용 JSON 형식을 지원한다.

### 5.2 세션 규칙

- 세션 ID는 런타임이 생성하는 불투명 UUID를 사용한다.
- 활성 세션 선택은 사용자별 로컬 설정에 저장한다.
- 하나의 세션에서는 동시에 한 turn만 실행한다.
- 다른 세션 간 병렬 실행은 설정된 최대 동시성으로 제한한다.
- 세션 삭제는 history, session memory, workspace와 연결된 job을 사전 점검한다.
- PoC에서는 삭제를 기본적으로 hard delete하지 않고 tombstone 또는 별도 확인 절차로 처리한다.

## 6. Agent Loop

한 round의 순서는 다음과 같이 고정한다.

1. 사용자 메시지 저장
2. turn lifecycle을 `running`으로 전환
3. 관련 Skill 선택과 memory retrieval
4. Context Builder 실행
5. Cloud LLM 요청과 원본 응답 metadata 저장
6. assistant content와 Tool call 저장
7. Tool argument schema 검증
8. Tool 정책과 승인 필요 여부 판정
9. Tool 실행, timeout 적용, 결과 크기 제한
10. Tool 결과 저장
11. 다음 round 또는 terminal 상태 결정

종료 조건:

- 모델이 Tool call 없는 최종 응답을 반환
- terminal Tool이 성공
- 최대 Tool round 도달
- 사용자 취소
- provider 또는 runtime의 복구 불가능한 오류

Agent Loop는 Tool 결과를 추측하거나 실패를 성공으로 변환하지 않는다. 모든 provider와 Tool 오류는 구조화된 code, message, retry safety를 가진다.

### 6.1 Turn 상태

```text
queued → running → completed
                 ↘ failed
                 ↘ cancelled
                 ↘ timed_out
                 ↘ interrupted
```

프로세스 시작 시 `running` 상태로 남아 있는 turn은 `interrupted`로 정정한다. 외부 side effect가 있는 Tool은 자동 재실행하지 않는다.

## 7. AgentModel과 MemoryModel

### 7.1 공통 계약

`AgentModel`은 사용자 응답과 Tool planning을 담당하며 다음 정규화된 데이터를 주고받는다.

- role/content/reasoning/tool-call을 표현하는 chat message
- JSON Schema를 포함하는 Tool spec
- assistant content와 구조화된 Tool call
- input/output token과 latency 같은 usage
- retry 가능 여부가 포함된 오류

`MemoryModel`은 대화 consolidation과 memory 후보 생성을 담당한다. Phase 1의 Cloud 구현과 Phase 4의 로컬 구현이 같은 입력 범위, prompt version과 출력 schema를 사용할 수 있도록 AgentModel과 별도 계약으로 정의한다.

### 7.2 구현 순서와 교체 규칙

1. Fake provider
2. OpenAI-compatible Cloud AgentModel
3. Cloud MemoryModel
4. OpenAI Responses 및 Anthropic-compatible AgentModel adapter
5. Local SLM MemoryModel

Cloud API key는 데이터베이스나 trace에 저장하지 않는다. 환경 변수 또는 별도의 secret file reference로만 읽고, logging과 예외 메시지에서 redaction한다.

Phase 1에서는 AgentModel과 MemoryModel이 모두 Cloud를 사용한다. Phase 4에서 MemoryModel backend만 설정으로 로컬 SLM으로 교체할 수 있게 하며 Cloud MemoryModel은 평가 baseline으로 계속 유지한다.

로컬 SLM을 AgentModel provider 인터페이스에 억지로 결합하지 않는다. 로컬 모델 장애가 일반 대화를 중단시키지 않도록 consolidation job만 재시도 가능한 상태로 남긴다.

Cloud와 Local 비교 시 입력 message 범위, 출력 JSON schema, prompt version과 후속 gate를 동일하게 유지한다. 각 호출에는 역할, backend, model ID, prompt/schema version, token usage와 latency를 기록한다.

## 8. 영속 저장

SQLite는 WAL mode와 foreign key enforcement를 사용한다. schema migration은 순번이 지정된 SQL 파일로 관리한다.

### 8.1 핵심 테이블

| 테이블 | 핵심 데이터 |
| --- | --- |
| `sessions` | ID, title, lifecycle, 생성·수정 시각 |
| `turns` | session, 상태, round, 시작·종료 시각, terminal reason |
| `messages` | session, turn, role, content, 순서, model metadata |
| `tool_calls` | call ID, name, arguments, validation·approval·실행 상태 |
| `tool_results` | call ID, bounded content, error, metadata, duration |
| `memories` | scope, type, content, confidence, sensitivity, 상태, version |
| `memory_sources` | memory와 근거 message의 연결 및 source span |
| `memory_embeddings` | model/version, vector, 생성 시각 |
| `consolidation_runs` | 처리 message 범위, model/prompt version, 결과 |
| `audit_events` | 정책 판정, redaction, 승인, 거부 사건 |
| `evaluation_runs` | dataset/version, configuration, metric 결과 |

### 8.2 저장 원칙

- 사용자 입력은 Agent Loop 실행 전에 저장한다.
- assistant Tool call과 각 Tool 결과는 별도 record로 저장한다.
- trace에는 API key, authorization header와 원문 secret을 저장하지 않는다.
- Tool 결과는 설정된 글자·byte 제한을 적용하고 원본 보관 여부를 명시한다.
- 시간은 UTC epoch와 timezone metadata를 사용한다.
- memory update는 기존 record를 덮어쓰지 않고 새 version과 `supersedes_id`를 만든다.

## 9. Context Builder

Context Builder의 입력은 다음과 같다.

- system policy
- 현재 시각, session ID와 workspace 같은 runtime context
- Tool 사용 정책 요약
- 선택된 장기 메모리
- 활성 Skill 본문
- 최근 대화와 완결된 Tool call/result chain

조립 순서:

```text
System Policy
Runtime and Workspace Context
Retrieved Memory
Active Skills
Recent Conversation
```

### 9.1 Context 예산

전체 context token budget을 구성 요소별로 제한한다.

- system/runtime: 항상 포함되는 예약 영역
- memory: retrieval 결과가 사용하는 최대 budget
- Skill: 우선순위와 관련도에 따른 최대 budget
- history: 남은 budget에서 최근 완결 turn부터 역순 선택
- provider output: 응답과 Tool call을 위한 별도 예약

불완전한 assistant Tool call만 history에 포함하거나 대응 Tool result만 고립시키지 않는다. PalmClaw의 현재 `ContextBuilder`가 보존하는 Tool chain 일관성을 동일하게 검증한다.

메모리와 외부 Skill 본문은 신뢰할 수 없는 데이터일 수 있으므로 명확한 delimiter와 출처 metadata를 사용한다. 사용자·Tool 콘텐츠가 system instruction으로 합쳐지지 않게 한다.

## 10. Skill 시스템

### 10.1 탐색 규칙

- 설정된 builtin Skill root와 workspace Skill root에서 `<skill-name>/SKILL.md`를 탐색한다.
- workspace Skill이 같은 이름의 builtin Skill을 재정의할 수 있으나 출처를 기록한다.
- frontmatter에서 name, description, always, requirements와 compatibility를 파싱한다.
- 경로 탈출, 순환 symlink와 허용 root 밖의 참조를 차단한다.
- 선택되지 않은 Skill 전체 본문을 context에 넣지 않고 catalog summary만 제공한다.

### 10.2 선택 단계

1. `always`이고 compatibility 검사를 통과한 Skill 선택
2. name, description과 keyword를 이용한 deterministic 후보 생성
3. embedding 유사도를 이용한 후보 보강
4. 중복 제거와 최대 Skill 수·token budget 적용

PoC 첫 단계는 deterministic 선택으로 시작하고, embedding selector는 평가 가능한 별도 전략으로 추가한다.

### 10.3 Skill 신뢰 수준

- `builtin`: repository에서 제공되고 검토된 Skill
- `workspace`: 현재 workspace가 제공한 Skill
- `external`: 별도 설치 출처의 Skill

Skill 신뢰 수준은 Tool 권한을 자동으로 상승시키지 않는다. Skill은 지침일 뿐이며 실제 실행 권한은 항상 Tool Policy Engine이 결정한다.

## 11. Tool Registry와 권한

### 11.1 Tool 계약

각 Tool은 다음 정보를 제공한다.

- 고유 name과 description
- JSON argument schema
- 위험 등급
- 필요한 capability
- timeout과 최대 결과 크기
- side-effect 및 retry 특성
- 실행 함수

Tool 실행 상태:

```text
proposed → invalid
         → denied
         → approval_pending → denied
                            → executing → succeeded
                                        → failed
                                        → timed_out
```

### 11.2 초기 Tool

| Tool | 초기 지원 | 정책 |
| --- | --- | --- |
| 파일 목록·읽기 | 지원 | session/shared workspace 내부 |
| 파일 생성·수정 | 지원 | 쓰기 승인 정책과 크기 제한 |
| 파일 삭제·이동 | 제한 지원 | destructive 확인 필수 |
| HTTP fetch | 지원 | scheme, redirect, DNS/IP, 크기 제한 |
| Web search | provider adapter가 있을 때 지원 | query와 결과 trace redaction |
| 예약 작업 | 제한 지원 | 위험 Tool의 무인 실행 금지 |
| shell/process 실행 | 제외 | 별도 sandbox 설계 전 비활성화 |

### 11.3 Workspace 경계

- 모든 파일 경로는 lexical 검사 후 canonical path로 다시 검사한다.
- `..`, 절대 경로, symlink와 hard-link 기반 경계 우회를 테스트한다.
- session workspace가 다른 session workspace를 참조하지 못하게 한다.
- shared workspace는 명시적으로 허용된 경로만 노출한다.
- Tool은 raw host path 대신 `session://`과 `shared://` 경로를 우선 사용한다.
- service 계정과 디렉터리 permission을 사용해 애플리케이션 논리 경계 밖에서도 피해를 제한한다.

### 11.4 Web 경계

- 기본 허용 scheme은 `https`로 제한한다.
- loopback, link-local, private network와 cloud metadata endpoint 접근을 차단한다.
- redirect마다 목적지를 다시 검증한다.
- DNS rebinding을 고려해 연결 대상 IP를 검증한다.
- 요청·응답 크기와 전체 시간을 제한한다.
- authorization, cookie와 민감 header를 trace에서 제거한다.

## 12. 장기 메모리

### 12.1 메모리 범위

| 범위 | 용도 |
| --- | --- |
| `working` | 현재 turn 안에서만 사용하는 임시 상태 |
| `session` | 해당 세션에서만 유효한 사건·결정 |
| `global` | 검증된 사용자 선호와 장기 사실 |

Context에는 전체 global memory를 그대로 넣지 않는다. 현재 질의와 관련된 `session` 및 `global` memory만 retrieval한다.

### 12.2 메모리 record

각 메모리는 최소한 다음 필드를 가진다.

```json
{
  "subject": "user",
  "predicate": "preferred_language",
  "value": "Korean",
  "scope": "global",
  "memory_type": "preference",
  "confidence": 0.94,
  "sensitivity": "low",
  "status": "proposed",
  "evidence_message_ids": ["message-id"],
  "model_id": "configured-memory-model",
  "extractor_version": "v1"
}
```

Phase 1의 Cloud Summary Memory는 end-to-end baseline으로 별도 보존한다. Phase 3 이후의 Structured Memory는 자유 텍스트 summary만 저장하지 않고 구조화 필드와 사람이 읽는 rendering을 함께 둔다.

### 12.3 Consolidation trigger

다음 조건 중 하나가 충족되면 처리되지 않은 message 범위에 대해 job을 생성한다.

- 설정된 사용자/assistant message 수 도달
- 설정된 token 양 도달
- 세션 종료 또는 명시적 `/consolidate`

동일 message 범위를 중복 처리하지 않도록 watermark와 consolidation run을 transaction으로 관리한다.

### 12.4 MemoryModel 단계별 역할

Phase 1의 Cloud MemoryModel은 전체 흐름을 검증하기 위해 대화 window를 요약한 단순 memory artifact를 생성한다. 이 구현은 `Cloud Summary Memory` baseline으로 보존한다.

Phase 3에서는 Cloud MemoryModel이 다음과 같은 구조화 memory 후보를 생성하도록 확장한다.

- 대화에서 메모리 후보 추출
- 후보 유형, 범위, confidence와 sensitivity 제안
- 기존 메모리와의 entailment/contradiction/unknown 분류 보조
- 명시된 schema에 맞는 JSON 출력

Phase 4에서는 같은 계약과 JSON schema를 구현하는 Local SLM MemoryModel을 추가한다. Cloud MemoryModel은 삭제하지 않고 품질·속도·비용·Cloud 노출량 비교를 위한 baseline으로 유지한다.

어떤 MemoryModel도 기존 memory를 직접 수정하거나 삭제하지 못하게 한다. MemoryModel은 후보만 제안하고, 실제 반영은 schema validation과 gate가 결정한다. 검증 실패 시 원문 대화는 보존하고 후보만 거부한다.

로컬 inference adapter는 localhost의 OpenAI-compatible endpoint 또는 llama.cpp server를 우선 지원한다. backend, model과 prompt/schema version을 모든 consolidation run에 기록한다.

## 13. Memory 검증 Gate

후보는 다음 순서로 검증한다.

1. JSON Schema 및 허용 필드 검증
2. evidence message 존재 여부 확인
3. evidence span과 후보 내용의 lexical/semantic 연관성 확인
4. 추측, 미래 계획과 일회성 상태의 장기 사실 오인 여부 확인
5. PII·secret 분류
6. 동일 subject/predicate의 기존 memory 검색
7. entailment/contradiction/unknown 판정
8. confidence와 scope 정책 적용
9. accept, review, reject 또는 supersede 결정

결정 원칙:

- evidence가 없는 후보는 저장하지 않는다.
- 모델 추론을 사용자가 말한 사실로 승격하지 않는다.
- `unknown`은 기존 값을 덮어쓰지 않는다.
- 충돌 시 시간과 명시성만으로 자동 승자를 정할 수 없으면 review queue로 보낸다.
- 민감도가 높은 후보는 정확하더라도 global memory 자동 저장을 금지할 수 있다.
- update는 기존 record 변경이 아니라 새 version 생성으로 표현한다.

## 14. Top-k Memory Retrieval

Phase 3에서는 다음 retrieval mode를 같은 query와 memory snapshot에서 비교한다.

- `full`: 전체 memory 주입 실험 baseline
- `bm25`: lexical relevance 기반 top-k
- `embedding`: semantic similarity 기반 top-k
- `hybrid`: BM25와 embedding 후보의 결합 및 재정렬

`full`은 연구 비교용이며 기본 runtime 경로는 top-k retrieval을 사용한다.

retrieval 단계:

1. 현재 사용자 입력과 최근 대화에서 검색 query 생성
2. session/global scope와 sensitivity 정책으로 후보 필터링
3. 선택된 mode에 따라 BM25, embedding 또는 양쪽 후보 생성
4. relevance, confidence, recency, scope와 사용 이력으로 재정렬
5. 동일 사실 중복과 서로 충돌하는 version 제거
6. top-k 및 memory token budget 적용

PoC 초기 데이터 규모에서는 SQLite에 embedding을 저장하고 애플리케이션에서 exact cosine search를 수행한다. 데이터가 충분히 커진 뒤 `sqlite-vec` 같은 index adapter를 검토한다.

retrieval 점수는 실험 가능하도록 각 항의 값을 trace에 남긴다.

```text
score =
  lexical_or_semantic_relevance
  + confidence_weight
  + recency_weight
  + scope_weight
  - sensitivity_penalty
  - redundancy_penalty
```

가중치는 코드 상수로 숨기지 않고 versioned evaluation configuration으로 관리한다.

## 15. 개인정보와 Redaction

### 15.1 데이터 등급

| 등급 | 예 | 기본 처리 |
| --- | --- | --- |
| public | 일반 선호와 비민감 설정 | 정책 범위에서 저장·전송 |
| personal | 이름, 이메일, 전화번호, 주소 | 목적과 scope에 따라 masking |
| sensitive | 건강, 금융, 신원 식별 정보 | 자동 global memory 제한 |
| secret | API key, token, password, cookie | 저장·전송·trace 금지 |

### 15.2 적용 지점

- message 수신 직후 분류 metadata 생성
- memory 후보 저장 전 sensitivity gate
- Cloud LLM context 생성 직전 최종 redaction
- Tool argument와 결과 trace 저장 전 redaction
- log와 exception rendering 시 redaction
- 평가 데이터 export 시 별도 redaction

원문을 로컬에 저장해야 하는 실험에서는 redacted 값과 원문을 같은 필드에 섞지 않는다. 원문 보관 여부, 암호화와 retention은 명시적 설정으로 둔다.

PII detector는 deterministic pattern, NER/SLM detector와 allowlist를 조합한다. secret pattern은 recall을 우선하고, 일반 PII는 precision과 recall을 함께 평가한다.

기능은 단계별로 적용한다.

- Phase 1: API key, authorization, token, password와 cookie를 대상으로 한 최소 secret redaction
- Phase 2: log, trace, Tool argument/result와 예외 경로 전반의 secret redaction 검증
- Phase 4: 이메일, 전화번호, 주소와 민감정보를 포함한 PII 분류, masking 정책과 정량 평가

PII 연구가 Phase 4에 있더라도 secret의 저장·Cloud 전송·trace 차단은 Phase 1부터 fail-closed로 적용한다.

## 16. 평가 계획

### 16.1 Agent Runtime

- 다중 Tool round 성공률
- 잘못된 Tool argument 거부율
- Tool timeout·실패 후 상태 일관성
- 재시작 후 session/trace 복구율
- 동일 세션 직렬화와 다른 세션 동시성
- context token budget 준수율

### 16.2 Memory 품질

- 후보 extraction precision, recall, F1
- evidence 없는 memory 생성률
- scope 분류 정확도
- conflict 탐지 precision과 recall
- 잘못된 자동 overwrite 비율
- retrieval Recall@k, MRR 또는 nDCG
- memory를 사용한 답변 정확도와 무관 memory 주입률

### 16.3 개인정보

- PII category별 precision, recall, F1
- Cloud AgentModel prompt의 PII 노출량
- Cloud MemoryModel consolidation prompt의 PII 노출량
- secret 노출 건수
- redaction 이후 과도한 정보 손실률
- Tool trace와 log의 민감정보 잔존량

“개인정보 노출량”은 최소한 다음 두 값으로 보고한다.

- `exposed_sensitive_spans / total_sensitive_spans`
- `exposed_sensitive_characters / total_sensitive_characters`

secret의 허용 노출 목표는 0이다.

### 16.4 Ablation

먼저 다음 핵심 configuration을 동일한 task와 dataset에서 비교한다.

1. No Memory
2. Cloud Summary Memory
3. Cloud Structured Memory + BM25
4. Cloud Structured Memory + Embedding
5. Local Structured Memory + BM25
6. Local Structured Memory + Embedding
7. Local Memory + Retrieval + Gate + Redaction

이후 retrieval, conflict gate와 PII redaction을 하나씩 제거하는 targeted ablation을 수행한다. 모든 조합을 전수 실행하기보다 핵심 configuration을 먼저 비교하고 필요한 조합만 확장한다.

추가 비교 항목:

- 전체 memory 주입 대 top-k retrieval
- lexical Skill selection 대 embedding 보강
- confidence/recency/scope 가중치 변경
- Cloud MemoryModel 대 Local MemoryModel의 품질, latency, 비용과 Cloud 노출량

평가 fixture는 합성 데이터 또는 명시적으로 허가된 redacted 데이터만 repository에 포함한다. 실제 사용자 trace는 커밋하지 않는다.

## 17. 구현 단계

### Phase 1 — End-to-End 러프 구현

목표는 Ubuntu 터미널 입력부터 Cloud LLM, Tool, 저장과 memory consolidation까지 전체 흐름을 한 번 관통시키는 것이다. 프로젝트 골격과 핵심 protocol 구성도 이 Phase에 포함한다.

산출물:

- `ubuntu/` Python package, CLI entry point와 설정 loader
- AgentModel, MemoryModel, Tool, Repository protocol
- SQLite migration, session/message/Tool trace 저장
- OpenAI-compatible Cloud AgentModel
- 단순 summary를 생성하는 Cloud MemoryModel
- bounded Agent Loop와 기본 Context Builder
- 파일 조회·작성 Tool
- 기본 `SKILL.md` 탐색과 deterministic 선택
- Fake AgentModel, Fake MemoryModel과 Fake Tool
- model 역할, backend, model ID, prompt/schema version, token usage, latency와 experiment ID 기록

Phase 1에서도 다음 최소 안전 경계를 적용한다.

- session/shared workspace 밖 파일 접근 금지
- 기본 JSON argument 검증
- Tool timeout과 최대 round
- 파일 삭제와 임의 shell Tool 제외
- API key, authorization, token, password와 cookie의 저장·log·trace 차단

완료 조건:

- Ubuntu 터미널에서 새 세션을 만들고 Agent와 여러 번 대화한다.
- Cloud AgentModel이 파일 Tool을 호출하고 결과를 다음 round에 반영한다.
- 대화와 Tool trace가 SQLite에 저장된다.
- 설정된 대화량 이후 Cloud MemoryModel이 summary memory를 생성한다.
- 정상 종료 후 다시 실행해 기존 session, history, trace와 memory를 조회한다.
- Fake model 기반 핵심 흐름이 network 없이 재현된다.

### Phase 2 — Runtime 안정화

목표는 Phase 1의 vertical slice가 실패, 재시작, 장시간 실행과 동시 실행에서도 일관된 상태와 보안 경계를 유지하게 하는 것이다.

산출물:

- turn lifecycle과 in-flight turn 재시작 복구
- 동일 세션 직렬화와 다중 세션 제한 병렬화
- provider·Tool timeout, 취소와 최대 Tool round
- 구성 요소별 token budget
- Tool JSON Schema와 구조화된 오류
- canonical path, symlink와 session workspace 격리
- Web Tool의 redirect, private network와 SSRF 방어
- Tool side-effect와 retry safety metadata
- log, trace, Tool argument/result와 예외 경로의 secret redaction
- 단위·통합·보안·end-to-end 자동 테스트

완료 조건:

- 실행 중 강제 종료 후 기존 `running` turn이 `interrupted`로 복구된다.
- 완료된 외부 side effect를 자동으로 중복 실행하지 않는다.
- 잘못된 Tool 인자와 허용되지 않은 파일 경로가 실행 전에 차단된다.
- 긴 응답, 무한 Tool 호출, provider·Tool timeout이 명시적 terminal 상태를 남긴다.
- 동일 세션은 직렬 실행되고 여러 세션은 설정된 범위에서 동시에 실행된다.
- 불완전한 Tool chain이 provider context에 들어가지 않으며 token budget을 초과하지 않는다.
- 자동 회귀 테스트로 Phase 1의 전체 흐름을 검증한다.

### Phase 3 — Memory 연구 기반

목표는 Cloud MemoryModel을 유지한 상태에서 memory를 구조화하고, 생성 근거와 version을 추적하며, retrieval 전략을 비교할 수 있게 하는 것이다.

산출물:

- Cloud MemoryModel의 구조화 memory candidate 출력
- `working`, `session`, `global` memory scope
- source message와 evidence span 연결
- `proposed`, `verified`, `rejected`, `superseded` 상태
- immutable version과 `supersedes_id`
- consolidation watermark와 중복 방지
- 전체 memory 주입 baseline
- BM25 retrieval
- embedding exact cosine retrieval
- top-k와 memory token budget
- retrieval query, candidate score와 최종 선택 trace

완료 조건:

- 어떤 대화가 어떤 memory 후보와 최종 memory를 만들었는지 추적한다.
- 오래되거나 충돌하는 memory의 version 변화를 조회한다.
- 현재 질문과 관련된 memory만 top-k로 context에 포함할 수 있다.
- 전체 memory, BM25와 embedding retrieval 결과를 같은 query에서 비교한다.
- 누락되거나 잘못 검색된 memory 사례를 source evidence와 score까지 분석한다.
- Cloud Summary Memory와 Cloud Structured Memory를 모두 평가 baseline으로 유지한다.

### Phase 4 — 로컬 SLM·검증·개인정보

목표는 Cloud MemoryModel과 같은 계약의 Local SLM MemoryModel을 추가하고, memory validation과 개인정보 보호를 강화하는 것이다.

산출물:

- localhost OpenAI-compatible 또는 llama.cpp 기반 Local MemoryModel
- Cloud/Local 공통 입력 범위, prompt와 JSON schema
- schema, evidence와 source-span validation
- entailment·contradiction·unknown gate
- confidence와 scope 정책
- conflict review queue
- PII·secret detector와 redactor
- Cloud 전송 직전 최종 redaction
- Cloud/Local MemoryModel backend 선택 설정

완료 조건:

- 설정만으로 Cloud와 Local MemoryModel을 교체한다.
- 동일한 대화에서 두 MemoryModel의 후보와 최종 memory를 비교한다.
- 로컬 SLM 장애 시 Agent 대화는 유지되고 consolidation만 재시도 상태가 된다.
- evidence가 없거나 근거가 부족한 후보가 memory에 반영되지 않는다.
- `unknown` 또는 해결되지 않은 충돌이 기존 memory를 자동으로 덮어쓰지 않는다.
- 개인정보·API key fixture가 Cloud prompt, trace와 log에 노출되지 않는다.
- Cloud와 Local MemoryModel의 품질, latency, 비용과 Cloud 노출량을 비교한다.

Local MemoryModel은 consolidation을 위해 과거 대화를 Cloud로 다시 보내는 노출을 줄인다. 메인 AgentModel이 Cloud인 동안 현재 대화와 Agent context의 Cloud 노출까지 제거되는 것은 아니므로 두 노출을 별도로 측정한다.

### Phase 5 — 평가·Ablation

목표는 동일한 task와 dataset에서 memory backend, retrieval, gate와 redaction의 기여도를 재현 가능하게 측정하는 것이다.

산출물:

- versioned 평가 dataset과 task runner
- No Memory, Cloud Summary, Cloud Structured와 Local Structured configuration
- BM25/embedding retrieval 비교
- Gate/Redaction targeted ablation
- runtime, memory, retrieval와 privacy metric runner
- JSON/TSV/Markdown 결과표와 그래프 생성
- model, prompt, schema, policy, dataset와 seed version 기록

측정 항목:

- task 성공률
- memory extraction precision, recall과 F1
- unsupported-memory rate와 conflict detection F1
- Recall@k, MRR 또는 nDCG
- latency, token usage와 Cloud API 비용
- consolidation에 따른 Cloud 노출량
- redaction 전후 민감정보 노출량과 과도한 정보 손실률

완료 조건:

- 동일한 task를 각 Memory configuration으로 반복 실행한다.
- Retrieval, Gate와 Redaction을 하나씩 제거해 기여도를 분석한다.
- 동일 seed, model, prompt, schema, policy와 dataset version으로 재실행할 수 있다.
- baseline과 각 연구 기능의 성능 차이를 별도로 보고한다.
- 실패 사례를 source evidence, retrieval score와 gate 결정까지 추적한다.
- 결과표와 그래프를 논문 실험 결과에 사용할 수 있는 형태로 생성한다.
- 보고서에 실제 secret 또는 비식별화되지 않은 사용자 데이터가 포함되지 않는다.

## 18. 테스트 전략

### 단위 테스트

- context ordering과 token trimming
- Tool schema와 policy decision
- path normalization과 workspace 경계
- Skill frontmatter와 selection
- memory candidate schema와 versioning
- PII detector와 redaction
- retrieval scoring과 tie-breaking

### 통합 테스트

- Fake AgentModel과 Fake MemoryModel을 이용한 전체 Agent Loop
- SQLite transaction 실패와 재시작 복구
- Cloud/Local MemoryModel timeout과 invalid JSON
- Tool approval, timeout, result truncation
- consolidation watermark와 중복 job

### 보안 테스트

- `..`, symlink, absolute path와 다른 session 접근
- HTTP redirect, localhost, IPv4/IPv6 private range와 DNS rebinding
- prompt/Skill/Tool result에 포함된 instruction injection
- Tool arguments와 exception에 포함된 API key
- malformed JSON, oversized payload와 decompression bomb

### End-to-end 테스트

최소 시나리오:

1. 세션 생성
2. 사용자의 파일 작성 요청
3. LLM Tool call
4. 승인과 파일 생성
5. Tool 결과를 반영한 최종 응답
6. consolidation
7. 프로세스 재시작
8. 관련 메모리 retrieval
9. history, trace와 memory provenance 조회

## 19. 관찰 가능성과 감사

구조화 log와 audit event는 다음 ID를 공통으로 사용한다.

- session ID
- turn ID
- provider request ID
- Tool call ID
- consolidation run ID
- experiment ID
- evaluation run ID

기본 log에는 model의 숨겨진 reasoning 원문을 기록하지 않는다. 저장 가능한 assistant reasoning metadata가 있더라도 명시적 설정과 redaction을 적용한다.

주요 측정값:

- AgentModel/MemoryModel 역할, backend, model ID와 latency
- prompt/schema/policy version과 token usage
- turn별 round 수
- Tool 실행 시간, 실패·거부·timeout 수
- context 구성 요소별 token 수
- retrieval 후보 수와 최종 top-k
- consolidation 후보, accept, reject, review 수
- redacted span 수와 유형

## 20. 위험과 완화

| 위험 | 완화 |
| --- | --- |
| Cloud 또는 Local MemoryModel이 잘못된 메모리를 생성 | evidence 필수, proposed 상태, conservative gate |
| Cloud consolidation이 과거 대화 노출을 증가 | 입력 window 최소화, 별도 노출량 측정, Phase 4 Local backend 비교 |
| Cloud/Local 비교 조건이 달라짐 | 동일 입력 범위, prompt/schema, gate와 versioned configuration |
| 기존 memory 자동 오염 | versioning, supersede link, review queue |
| 전체 memory가 context를 오염 | scope filter, top-k와 token budget |
| Tool이 host 파일에 접근 | canonical path, 전용 OS 사용자, workspace allowlist |
| Web Tool을 통한 SSRF | DNS/IP/redirect 검증과 private range 차단 |
| 비밀정보가 Cloud로 전송 | context 직전 최종 redaction과 secret fail-closed |
| Tool 재시도로 side effect 중복 | idempotency metadata와 retry safety 분류 |
| 모델별 Tool protocol 차이 | normalized Provider adapter와 protocol fixture |
| 평가 결과 재현 불가 | model/prompt/policy/dataset version 저장 |
| Android 코드와 동작 괴리 | source-mapped contract test와 차이 문서화 |

## 21. 라이선스와 공개 경계

PalmClaw repository는 AGPL-3.0과 상용 라이선스 정보를 제공한다. Ubuntu runtime을 같은 repository에 구현하거나 기존 소스를 직접 사용해 배포할 때에는 해당 라이선스 의무를 따른다.

문서와 테스트 fixture에는 다음을 포함하지 않는다.

- API key, token, cookie와 실제 endpoint credential
- 사용자 원문 대화와 비식별화되지 않은 memory
- 개인 파일 경로와 machine-specific 절대 경로
- 실제 계정의 Tool trace

배포 또는 외부 서비스화 전에 dependency license와 local model license를 별도로 검토한다.

## 22. PoC Definition of Done

다음 조건을 모두 충족할 때 Ubuntu PoC를 완료로 본다.

- CLI에서 세션 생성, 선택, 대화와 history 조회가 가능하다.
- Cloud AgentModel과 Tool을 포함한 다중 round Agent Loop가 동작한다.
- 대화, Tool call/result, turn 상태가 재시작 이후 복구된다.
- `SKILL.md`가 안전하게 로드되고 관련 Skill만 context에 포함된다.
- 파일과 Web Tool이 schema, 승인, timeout과 workspace/network 경계를 지킨다.
- session/global memory가 분리되고 top-k retrieval을 사용한다.
- Cloud Summary, Cloud Structured와 Local Structured MemoryModel을 같은 평가 계약으로 실행한다.
- Cloud와 Local MemoryModel이 evidence가 있는 memory 후보만 제안한다.
- 충돌 후보가 검증 없이 기존 memory를 덮어쓰지 않는다.
- Cloud prompt, log와 trace에 secret fixture가 노출되지 않는다.
- memory 품질과 개인정보 노출량을 재현 가능한 평가로 보고한다.
- 각 단계의 단위·통합·보안·end-to-end 테스트가 통과한다.

## 23. 구현 시작 전 확인할 결정

아래 항목은 해당 기능을 구현하기 전에 짧은 ADR로 확정한다. Phase 1의 외부 Cloud 호출에 필요한 결정부터 우선 처리한다.

- 최초 지원 Cloud AgentModel/MemoryModel provider와 API protocol
- 두 Cloud 역할에 같은 model을 사용할지 별도 model을 사용할지
- 최초 local SLM과 embedding model
- 원문 대화의 retention 및 암호화 범위
- PII를 local memory에도 저장하지 않을지, Cloud 전송만 차단할지
- destructive Tool의 승인 UX
- 예약 작업을 PoC 필수 범위에 포함할지
- systemd service와 비대화형 실행을 어느 단계에서 허용할지
- 평가용 정답 dataset의 생성·검수 방식

Cloud provider 결정 전에도 Fake AgentModel, Fake MemoryModel과 합성 fixture를 이용한 Phase 1 골격 작업은 시작할 수 있다.
