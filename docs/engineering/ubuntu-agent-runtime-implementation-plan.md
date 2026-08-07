# Ubuntu Agent Runtime 구현 계획

상태: `Phase 5 implementation complete — Offline reference and OpenAI live evaluation complete`

최종 검토: 2026-07-24

상세 설계와 연구 배경은 [Ubuntu Agent Runtime PoC 상세 계획](ubuntu-agent-runtime-poc-plan.md)을 참고한다. 이 문서는 실제 구현 순서, 산출물과 완료 조건만 정의한다.

## 1. 목표와 고정 결정

목표는 PalmClaw의 Agent Runtime을 Ubuntu CLI 환경에서 재구현하고, Cloud/Local memory 전략을 비교할 수 있는 연구 PoC를 만드는 것이다.

- 구현 언어: Python 3.12
- 인터페이스: `AgentModel`과 `MemoryModel` 분리
- Phase 1: 두 모델 모두 Cloud LLM 사용
- Phase 4: `MemoryModel`만 로컬 SLM 구현 추가
- 저장소: SQLite, WAL, versioned migration
- 실행 경계: session/shared workspace 내부 Tool만 허용
- 초기 Tool: 파일 조회·작성, bounded web fetch
- 제외: Android UI·기기 Tool, 임의 shell, 원격 channel, 다중 사용자 서버

```text
CLI → Session Coordinator → Agent Loop
                           ├─ Context Builder
                           ├─ Cloud AgentModel
                           └─ Tool Registry

Memory Pipeline → MemoryModel
                  ├─ Cloud baseline
                  └─ Local SLM

SQLite → Session / Message / Tool Trace / Memory / Evaluation
```

## 2. Phase 요약

| Phase | 목표 | 완료 결과 |
| --- | --- | --- |
| 1 | End-to-End 러프 구현 | CLI부터 Tool·Cloud memory·저장까지 전체 흐름 동작 |
| 2 | Runtime 안정화 | 실패·재시작·동시성·보안 경계를 자동 테스트로 검증 |
| 3 | Memory 연구 기반 | 구조화 memory, provenance, BM25/embedding retrieval |
| 4 | 로컬 SLM·검증·개인정보 | Local MemoryModel, validation gate, PII redaction |
| 5 | 평가·Ablation | 각 memory 구성을 동일 task에서 비교하고 결과 생성 |

## 3. Phase 1 — End-to-End 러프 구현

구현:

- `ubuntu/` Python package, CLI와 설정 loader
- `AgentModel`, `MemoryModel`, `Tool`, `Repository` protocol
- SQLite migration과 session/message/Tool trace 저장
- OpenAI-compatible Cloud AgentModel
- summary memory를 생성하는 Cloud MemoryModel
- bounded Agent Loop와 기본 Context Builder
- 기본 `SKILL.md` 탐색·선택
- workspace 내부 파일 조회·작성 Tool
- Fake model과 Fake Tool

최소 안전장치:

- 기본 Tool JSON argument 검증
- workspace 밖 파일 접근 금지
- 최대 Tool round와 timeout
- 파일 삭제와 shell Tool 제외
- API key, token, password, cookie의 log·trace 저장 금지

완료 조건:

- CLI에서 세션을 생성하고 여러 번 대화한다.
- Cloud AgentModel이 파일 Tool을 호출하고 결과를 다음 round에 반영한다.
- 대화량 기준으로 Cloud MemoryModel consolidation이 실행된다.
- 정상 종료 후 session, history, trace와 memory를 다시 조회한다.
- Fake model 기반 전체 흐름을 network 없이 테스트한다.

## 4. Phase 2 — Runtime 안정화

구현:

- turn lifecycle과 강제 종료 후 `interrupted` 복구
- 동일 세션 직렬화와 다중 세션 제한 병렬화
- provider·Tool timeout, 취소와 최대 round
- 구성 요소별 context token budget
- Tool JSON Schema와 구조화된 오류
- canonical path, symlink와 session workspace 격리
- Web Tool redirect·private network·SSRF 방어
- side-effect와 retry safety metadata
- log·trace·Tool result의 secret redaction 검증
- 단위·통합·보안·end-to-end 테스트

완료 조건:

- 강제 종료 후 실행 중이던 turn이 안전한 상태로 복구된다.
- 완료된 side effect를 자동 재실행하지 않는다.
- 잘못된 Tool 인자와 경계 밖 경로가 실행 전에 차단된다.
- 무한 Tool 호출, 긴 응답과 timeout이 명시적 terminal 상태를 남긴다.
- 여러 세션을 동시에 실행해도 데이터가 섞이지 않는다.
- Phase 1 전체 흐름의 자동 회귀 테스트가 통과한다.

## 5. Phase 3 — Memory 연구 기반

구현:

- Cloud MemoryModel의 구조화 memory candidate
- `working`, `session`, `global` scope
- source message와 evidence span 연결
- `proposed`, `verified`, `rejected`, `superseded` 상태
- immutable version과 `supersedes_id`
- consolidation watermark와 중복 방지
- 전체 memory 주입 baseline
- BM25, embedding exact cosine와 hybrid retrieval
- top-k와 memory token budget
- retrieval query, score와 최종 선택 trace

완료 조건:

- 모든 memory의 source와 생성 과정을 조회한다.
- memory 변경과 충돌 이력을 version chain으로 확인한다.
- 전체 memory, BM25와 embedding 결과를 같은 query에서 비교한다.
- 관련 memory만 top-k로 context에 넣을 수 있다.
- 검색 누락·오류 사례를 evidence와 score까지 추적한다.
- Cloud Summary와 Cloud Structured Memory를 모두 baseline으로 유지한다.

## 6. Phase 4 — 로컬 SLM·검증·개인정보

구현 상태:

- `PALMCLAW_MEMORY_BACKEND`으로 Cloud/Fake/Local MemoryModel 독립 선택
- localhost OpenAI-compatible Chat Completions summary/structured adapter
- Cloud/Local 공통 prompt, structured schema와 입력 범위
- evidence entailment·contradiction·unknown 및 conflict gate
- confidence·scope·sensitivity 정책과 review queue/수동 판정
- 이메일·전화번호·주소·정부 ID·카드·secret 탐지 및 redaction
- Cloud Agent/Memory/embedding 호출 직전 최종 outbound redaction
- Local 장애의 `retryable` 격리와 backend latency/token/cost/privacy report
- 67개 offline test로 Phase 1–4 회귀 및 Phase 4 정책 검증

남은 검증:

- 현재 설치된 Ollama에는 로컬 모델이 없어 실제 SLM 품질·속도 E2E는
  모델 선택 및 로딩 후 수행한다.

구현:

- Cloud와 같은 계약의 Local SLM MemoryModel
- localhost OpenAI-compatible 또는 llama.cpp adapter
- Cloud/Local 공통 입력, prompt와 JSON schema
- schema·evidence·source-span validation
- entailment·contradiction·unknown gate
- confidence·scope 정책과 review queue
- PII·secret detector와 redactor
- Cloud 전송 직전 최종 redaction
- 설정 기반 Cloud/Local MemoryModel 전환

완료 조건:

- 설정만 바꿔 Cloud와 Local MemoryModel을 실행한다.
- 같은 대화에서 두 모델의 후보와 최종 memory를 비교한다.
- 근거가 없거나 충돌이 해결되지 않은 후보를 자동 반영하지 않는다.
- Local MemoryModel 장애가 Agent 대화를 중단시키지 않는다.
- 개인정보와 API key fixture가 Cloud prompt, trace와 log에 노출되지 않는다.
- 품질, latency, 비용과 Cloud 노출량을 backend별로 기록한다.

Local MemoryModel은 consolidation을 위해 과거 대화를 Cloud로 다시 보내는 노출을 줄인다. Cloud AgentModel에 전달되는 현재 대화와 context 노출은 별도로 측정한다.

## 7. Phase 5 — 평가·Ablation

구현 상태:

- 버전·SHA-256이 고정된 8개 합성 memory 평가 case
- No Memory, Summary, Structured+BM25/Embedding 및
  Retrieval/Gate/Redaction ablation을 포함한 10개 profile
- 재현 가능한 offline fixture mode와 실제 OpenAI를 호출하는 live mode
- task success, exact/value memory F1, unsupported memory, conflict,
  Recall@k/MRR/nDCG, PII, Cloud 노출량, latency/token/cost 측정
- 평가 run과 case trace의 SQLite 영속 저장
- JSON/JSONL/TSV/Markdown 및 SVG 그래프 자동 생성
- case 단위 실패 격리, seed·dataset·prompt/schema/policy version 기록
- 평가 DB와 산출물의 secret·PII 재-redaction

기준 실행:

- deterministic offline: 10 profiles × 8 cases, 80/80 완료
- OpenAI live: 4 profiles × 8 cases, 32/32 완료
- live structured value F1: BM25 0.750, embedding 0.667;
  두 retrieval profile의 Recall@k 0.500
- redaction 활성 profile의 합성 PII Cloud 노출률 0.000,
  redaction 제거 ablation은 1.000

Offline mode의 `cloud`/`local` 표시는 구성 profile을 뜻하며 동일한 deterministic
fixture를 사용한다. 따라서 모델 품질 비교 근거는 live mode 결과만 사용한다.
현재 기본 MemoryModel은 OpenAI이며 로컬 모델을 자동으로 다운로드하거나 실행하지
않는다.

비교 구성:

1. No Memory
2. Cloud Summary Memory
3. Cloud Structured Memory + BM25
4. Cloud Structured Memory + Embedding
5. Local Structured Memory + BM25
6. Local Structured Memory + Embedding
7. Local Memory + Retrieval + Gate + Redaction

측정:

- task 성공률
- memory precision, recall과 F1
- unsupported-memory rate와 conflict detection F1
- Recall@k, MRR 또는 nDCG
- latency, token usage와 Cloud API 비용
- Cloud Agent/Memory 호출별 개인정보 노출량
- redaction에 따른 정보 손실률

완료 조건:

- 동일 task를 모든 핵심 구성에서 반복 실행한다.
- Retrieval, Gate와 Redaction을 하나씩 제거해 기여도를 분석한다.
- model, prompt, schema, policy, dataset와 seed version을 기록한다.
- 실패 사례를 evidence, retrieval score와 gate 결정까지 추적한다.
- JSON/TSV 결과표와 논문용 그래프를 재현 가능하게 생성한다.

위 완료 조건은 충족됐다. 이후 논문 실험에서는 task 수와 반복 횟수를 늘리고,
비용 단가를 설정한 뒤 confidence interval과 통계 검정을 추가한다.

## 8. 공통 구현 규칙

- 사용자 메시지는 Agent 실행 전에 저장한다.
- assistant Tool call과 Tool result는 별도 record로 저장한다.
- MemoryModel은 후보만 생성하고 memory를 직접 수정하지 않는다.
- memory update는 overwrite 대신 새 version으로 기록한다.
- Cloud/Local 비교는 동일 입력 범위, prompt, schema와 gate를 사용한다.
- API key와 secret은 DB, log, trace와 평가 결과에 저장하지 않는다.
- model 역할, backend, model ID, prompt/schema version, latency와 usage를 Phase 1부터 기록한다.
- 외부 side effect가 있는 Tool은 안전성이 확인되지 않으면 자동 재시도하지 않는다.

## 9. 최소 데이터 모델

| 테이블 | 용도 |
| --- | --- |
| `sessions` | 세션 정보 |
| `turns` | 실행 상태와 terminal reason |
| `messages` | user/assistant/tool 대화 |
| `tool_calls`, `tool_results` | 검증·승인·실행 trace |
| `memories`, `memory_sources` | versioned memory와 evidence |
| `memory_embeddings` | embedding model/version과 vector |
| `consolidation_runs` | 처리 범위와 MemoryModel 결과 |
| `audit_events` | 정책·redaction·승인 기록 |
| `evaluation_runs` | 실험 구성과 metric |

## 10. 구현 착수 순서

Phase 1 시작 시 아래 순서로 작업한다.

1. Python package와 configuration
2. SQLite schema와 migration
3. Fake AgentModel/MemoryModel 기반 Agent Loop
4. CLI session·chat·history
5. Cloud AgentModel
6. 파일 Tool과 workspace guard
7. Cloud MemoryModel consolidation
8. trace·secret 검증과 end-to-end test

각 Phase는 해당 완료 조건과 자동 테스트가 충족된 뒤 다음 Phase로 진행한다.
