# Ubuntu Agent Runtime E2E 개요

이 문서는 Ubuntu PoC의 실행 흐름과 Phase 5 평가 데이터셋을 빠르게 파악하기
위한 코드 안내서다.

## 1. 평가 데이터셋

평가에는 외부 공개 벤치마크가 아닌 저장소 내 합성 데이터셋
`palmclaw-synthetic-memory v1.0.0`을 사용했다.

- 위치:
  [`synthetic_memory_v1.json`](../../ubuntu/src/palmclaw_ubuntu/evaluation_datasets/synthetic_memory_v1.json)
- case 수: 8
- SHA-256:
  `8ea95c89fb1e9fea2287a8d7aef63fdf8cda803b5b843656db07b5320d50a065`
- 개인정보 case도 `alice@example.test` 같은 식별 불가능한 합성 값만 사용

| Case | 검증 대상 |
| --- | --- |
| `editor_preference` | global 선호 추출과 retrieval |
| `project_decision` | session 범위 결정 추출과 retrieval |
| `unsupported_hallucination` | 근거 없는 memory 후보 차단 |
| `unresolved_conflict` | 기존 memory 충돌과 review 전환 |
| `explicit_update` | 명시적 변경과 version supersede |
| `synthetic_contact_pii` | PII 탐지·redaction·review |
| `retrieval_distractors` | 여러 memory 중 top-k 선택 |
| `transient_request` | 일회성 요청을 장기 memory로 저장하지 않음 |

Offline 평가는 fixture model을 사용해 평가 코드와 ablation 방향을 검증한다.
Live 평가는 같은 합성 대화를 실제 OpenAI `MemoryModel`에 전달한다. 현재
데이터셋은 작은 기능 검증용이므로 실제 대화 분포나 통계적으로 유효한 모델
성능을 대표하지 않는다.

## 2. 일반 대화 E2E 흐름

```text
CLI 입력
  → Runtime/SQLite 초기화 및 중단 상태 복구
  → 세션별 Turn 직렬화
  → 사용자 메시지 저장
  → 관련 Memory retrieval
  → System + Memory + Skill + 최근 History 조립
  → OpenAI AgentModel 호출
  → Tool call schema·권한 검증
  → Tool 실행 및 결과 저장
  → 결과를 포함해 AgentModel 재호출
  → 최종 응답·Turn 상태 저장
  → OpenAI MemoryModel consolidation
  → evidence·충돌·PII gate
  → versioned Memory 저장
```

핵심 처리 순서는 다음과 같다.

1. [`cli.py`](../../ubuntu/src/palmclaw_ubuntu/cli.py)의 `ask`/`chat`이 설정과
   활성 세션을 확인하고 runtime을 생성한다.
2. [`application.py`](../../ubuntu/src/palmclaw_ubuntu/application.py)가
   SQLite, OpenAI provider, Context Builder, Tool Registry와 Memory Engine을
   조립한다.
3. [`coordinator.py`](../../ubuntu/src/palmclaw_ubuntu/coordinator.py)는 같은
   세션의 turn을 직렬화하고 서로 다른 세션의 동시 실행 수를 제한한다.
4. [`agent.py`](../../ubuntu/src/palmclaw_ubuntu/agent.py)는 사용자 메시지를
   먼저 저장하고 `LLM → Tool → 결과 → LLM` loop를 최대 round와 timeout
   안에서 실행한다.
5. [`context.py`](../../ubuntu/src/palmclaw_ubuntu/context.py)와
   [`skills.py`](../../ubuntu/src/palmclaw_ubuntu/skills.py)가 관련 memory,
   `SKILL.md`, 완전한 Tool chain과 최근 history를 token budget 안에서
   조립한다.
6. [`tools.py`](../../ubuntu/src/palmclaw_ubuntu/tools.py)가 JSON Schema를
   검증하고 파일·웹 Tool을 실행한다. 파일 경계와 symlink 검사는
   [`workspace.py`](../../ubuntu/src/palmclaw_ubuntu/workspace.py)가 담당한다.
7. Tool call/result는 SQLite에 저장되고 다음 LLM 호출의 history에 포함된다.
   Tool call이 없으면 응답과 turn을 완료한다.
8. [`memory.py`](../../ubuntu/src/palmclaw_ubuntu/memory.py)가 대화량 기준으로
   OpenAI MemoryModel을 호출하고 summary 또는 structured candidate를 만든다.
9. [`validation.py`](../../ubuntu/src/palmclaw_ubuntu/validation.py)가
   evidence·confidence·충돌을 판정한다. 승인된 structured memory는 새
   version으로 저장되고 충돌 후보는 reject/review/supersede 처리된다.
10. [`privacy.py`](../../ubuntu/src/palmclaw_ubuntu/privacy.py)는 Cloud 호출,
    trace와 평가 산출물에서 secret·PII를 탐지하고 redaction한다.

모든 session, turn, message, Tool trace, model call, memory와 retrieval 기록은
[`storage.py`](../../ubuntu/src/palmclaw_ubuntu/storage.py)와 versioned SQL
migration에 저장되므로 프로세스를 다시 실행해도 복구할 수 있다.

## 3. Phase 5 평가 E2E 흐름

```text
Dataset + Profiles + Seed
  → case/profile/repetition 스케줄 생성
  → case별 격리 Runtime 생성
  → 초기 Memory와 합성 대화 입력
  → consolidation + retrieval 실행
  → gold Memory/Gate/PII와 비교
  → case trace와 metric 저장
  → JSON·JSONL·TSV·Markdown·SVG 생성
```

[`evaluation.py`](../../ubuntu/src/palmclaw_ubuntu/evaluation.py)가 전체 평가를
실행한다. `offline`은 deterministic fixture, `live`는 설정된 OpenAI
MemoryModel을 사용한다. 각 run에는 dataset version/hash, profile, seed,
반복 횟수와 prompt/schema/policy version이 기록된다.

주요 지표는 task success, exact/value Memory F1, unsupported-memory rate,
gate/conflict 품질, Recall@k·MRR·nDCG, PII 탐지·Cloud 노출량, latency,
token과 추정 비용이다. 평가 run과 case trace는
[`006_evaluation_runs.sql`](../../ubuntu/src/palmclaw_ubuntu/migrations/006_evaluation_runs.sql)
스키마에 저장된다.

```bash
# 전체 deterministic 평가
palmclaw eval run --mode offline

# OpenAI MemoryModel 평가
palmclaw eval run \
  --mode live \
  --profiles cloud_summary,cloud_structured_bm25,cloud_structured_embedding

# 저장된 실행 조회
palmclaw eval list
palmclaw eval show RUN_ID
```

현재 기준 결과:

- [Offline 10-profile 결과](../../ubuntu/evaluation/results/912e041a-2a1e-4b79-bb62-8d97e94a085f/results.md)
- [OpenAI live 결과](../../ubuntu/evaluation/results/c4c613ce-63df-42b8-af99-3919c781e814/results.md)

자동 회귀 검증은
[`tests/`](../../ubuntu/tests)에서 담당하며 현재 72개 테스트와 87% line
coverage를 통과한다.
