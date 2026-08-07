# Joint Memory–Tool Planning 구현 계획

상태: `후보 reranker 개선 및 고정-cache 재검증 완료, live 재평가 전`

## 현재 구현

- 제품 runtime opt-in: `PALMCLAW_JOINT_MEMORY_TOOL_PLANNING_ENABLED=1`
- VehicleMemBench profile: `cloud_joint_planned_fact_patch`
- Tool/schema 상한: `PALMCLAW_JOINT_MEMORY_TOOL_MAX_TOOLS`,
  `PALMCLAW_JOINT_MEMORY_TOOL_SCHEMA_TOKENS`
- 기존 route + Fact별 hint fallback과 기존 Ours profile은 그대로 유지한다.
- Top-k Fact와 같은 source-event의 관련 Fact는 planner 전용 입력으로 최대 12개까지
  보강한다. 기존 Memory prompt 본문과 Top-k 선정 결과는 바꾸지 않는다.
- 1차 구현은 별도 planner LLM 호출 없이 bounded 공동 selector와 deterministic
  Desired-state/binder를 사용한다. 최종 Agent LLM은 Fact와 schema-valid 복합 계획을
  함께 판단한다. 별도 구조화 planner LLM은 deterministic 단계에서 남은 ambiguity가
  평가상 병목일 때만 후속으로 추가한다.

## 목표

현재의 분리된 `질문 기반 Tool route`와 `Fact별 실행 hint`를 다음 두 단계로
교체한다.

```text
질문 + 검색 Fact/Bundle + 제한된 Tool Schema
  → 공동 capability/Tool 선택
  → Desired-state Plan
  → schema-valid Tool Plan
```

목표는 selector miss, Fact별 필수 인자 부족, 복합 Tool 누락을 줄이는 것이다.
Fact 저장·Recursive-assisted 추출 방식은 유지하며 Gold call이나 task별 규칙은
사용하지 않는다.

## 핵심 계약

- `MemoryBundle`: 같은 source event, entity, applicability에 속하는 관련 Fact와
  correction/version 관계를 묶는다.
- `DesiredStateItem`: capability, operation, target, desired value, runtime selector,
  evidence record ID, confidence와 unresolved slot을 가진다.
- `ToolPlan`: 하나 이상의 Tool call, 호출 순서·의존성, 사용한 Desired State와
  schema validation 결과를 가진다.
- 각 계획의 상태는 `ready`, `ambiguous`, `blocked` 중 하나다. 근거 없는 target,
  zone, enum 또는 현재 상태는 생성하지 않는다.

## 구현 단계

### 0. 새 profile과 기준선 고정 — 완료

- 기존 `cloud_schema_informed_recursive_assisted_fact_patch`는 변경하지 않는다.
- 새 평가 profile을 추가해 동일 Memory cache로 matcher/planner만 비교한다.
- 현재 selector, hint rejection, context token, Tool discovery trace를 기준선으로
  보존한다.

### 1. Bundle provenance 추가 — 완료

- Fact candidate/record에 source event를 나타내는 `bundle_id`와 evidence message ID를
  보존한다.
- 기존 DB에는 additive migration을 사용하고, bundle 정보가 없는 각 record는
  서로 합쳐지지 않는 독립 singleton bundle로 안전하게 취급한다.
- entity, applicability, evidence event가 다른 Fact는 자동 병합하지 않는다.

### 2. 개선 1 — 공동 Tool 선택 — 완료

- 초기 Fact 검색은 Tool route를 hard filter로 사용하지 않는 현재 memory-first
  동작을 유지한다.
- `query route ∪ Fact capability ∪ bundle capability`로 Tool 후보를 구성한다.
- 후보 Tool의 schema slice만 사용해 질문, Fact, capability를 함께 rerank한다.
- 결과에는 선택 Tool뿐 아니라 선택 근거와 누락 required slot을 기록한다.
- 후보 수와 schema token budget을 설정 가능하게 제한하고, 초과 시 점수순으로
  자른다.

### 3. 개선 2 — Desired-state와 복합 Tool Plan — Core 완료

- 공동 선택 결과의 관련 Fact를 `bundle_id`, entity, applicability 기준으로 묶는다.
- 질문과 선택적 current-state snapshot을 함께 사용해 구조화된 Desired State를
  생성한다.
- 현재는 deterministic planner가 bounded 구조를 만들고 기존 Agent LLM이 복합
  계획을 판단한다. 별도 LLM refinement를 추가하더라도 최종 required argument,
  enum/range/type 검사는 deterministic binder가 수행한다.
- binder는 여러 Fact와 query runtime selector를 한 호출의 slot에 결합하고,
  `switch + setting`, `source + play` 같은 호출 묶음의 순서와 의존성을 만든다.
- current state가 필요한 상대 연산인데 snapshot이 없으면 `blocked` 또는
  `ambiguous`로 남긴다.

### 4. Runtime 연결과 안전한 fallback — 완료

- Agent에는 raw 전체 schema나 전체 Fact DB 대신 선택된 Bundle, Desired State,
  validated Tool Plan만 제공한다.
- `ready` call만 preload하고, `ambiguous` 항목은 Agent가 확인하거나 Tool discovery로
  보완한다.
- planner 실패·timeout·invalid JSON이면 기존 route + Fact별 hint 경로로 fallback한다.
- 제품 Agent와 VehicleMemBench runner가 같은 planner/binder 모듈을 사용하게 한다.

### 5. 검증과 채택 조건 — 고정-cache audit 완료, E2E 보류

- 단위 테스트: bundle 격리, correction/version, multi-Fact slot 결합, 복합 호출
  순서, schema validation, missing-state abstention, token cap과 fallback.
- 고정-cache offline audit: reference Tool boundary recall, required-slot completion,
  invalid plan, ambiguous/blocked 비율을 기존 방식과 비교한다.
- VehicleMemBench 1–10: ESM, State/Tool F1, argument exact, selector miss, mapping
  rejection, discovery call, 불필요 call, Agent/planner token을 측정한다.
- Gold-derived default, task별 literal, 다른 entity/event의 Fact 혼합, schema-invalid
  call이 하나라도 생기면 채택하지 않는다.
- 고정 Memory cache에서 Agent 반복 평가 후 정확도 개선이 실행 변동보다 크고,
  planner 추가 token/latency가 설정한 예산 안일 때만 기본 경로로 승격한다.

## 2026-08-06 고정-cache audit 결과

VehicleMemBench S1–S10의 기존 Ours 결과 100 task를 같은 Fact cache로 재생했다.
Gold call은 생성에 사용하지 않고 사후 채점에만 사용했으며 Memory, embedding,
Agent 호출은 수행하지 않았다.

| 지표 | 기존 | 공동 planner |
| --- | ---: | ---: |
| reference Tool을 모두 포함한 selector | 51/100 | 58/100 |
| reference Tool을 모두 포함한 initial boundary | 57/100 | 64/100 |
| 평균 선택 Tool 수 | 2.60 | 4.10 |
| 평균 선택 schema token 추정치 | - | 261.53 |
| 실제 safety fallback 포함 exact reference hint/call | 19/112 | 19/112 |

- 초기 후보 union 방식은 평균 12 Tool과 non-reference plan 134개로 과도해 폐기했다.
  현재는 route 후보 또는 `query와 Fact가 함께 지지하는 후보`만 확장한다.
- mapping rejection은 311에서 243으로 줄었다. 기존 binder와 충돌하는 single-Fact
  plan은 기존 hint를 유지하고, 완전한 multi-Fact aggregation 또는 기존 hint를 모두
  보존하는 plan만 적용하는 safety gate를 추가해 exact 비열화를 제거했다. 이 gate로
  100 task 중 88 task는 기존 hint를 유지했다.
- 기존 cache에서 planner 관련 Fact 보강은 평균 0.06개뿐이었고, entity와
  applicability 경계를 적용한 multi-Fact bundle은 0개였다. 따라서 개선 2의 실제
  효과는 복합 Fact fixture와 새 cache/live run에서 별도로 검증해야 한다.
- audit artifact는
  `ubuntu/evaluation/vehiclemembench-joint-planning-audit/s1-s10-v4`에 있다.
- 실행 hint/call 비열화 방지와 불필요 call safety gate는 충족했다. 복합 Fact bundle
  단위·retrieval fixture도 통과하지만 기존 S1–S10 cache에는 해당 표본이 없다.
  새 cache 또는 별도 복합 표본에서 개선 2가 실제 적용되는 것을 확인하기 전에는
  기본 profile 승격을 진행하지 않는다.

동일 component cache로 수행한 Scenario 6–10 live 비교는
[별도 결과 문서](joint-tool-selection-s6-s10-results.md)에 기록한다. 개선 1은 selector
miss와 discovery call을 줄였지만 전체 ESM은 동일했고 State/Tool F1은 소폭 낮아,
현재는 반복 평가 전 opt-in 상태를 유지한다.

Scenario 1–5의 [동일-cache 비교](joint-tool-selection-s1-s5-results.md)에서도 selector
miss와 discovery call은 줄었지만 ESM은 0.52에서 0.50으로 낮아지고 불필요 호출이
늘었다. 후보 확장과 Agent context를 조정하기 전까지 현재 형태의 기본 채택을 보류한다.

## 후보 reranker 후속 개선

초기 합집합 확장을 다음의 보수적 reranking으로 교체했다.

- 기존 query route는 non-regression floor로 보존한다.
- Fact가 직접 지지하는 추가 Tool은 전체에서 최고 점수 1개만 허용한다.
- 같은 domain은 route와 추가 후보를 합쳐 기본 최대 3개로 제한한다. 단, 기존 route가
  이미 이를 초과하면 route 자체는 삭제하지 않는다.
- identifier를 underscore 단위로 토큰화해 `seat_leg_support_height`와
  `leg_support_height` 같은 직접 관계를 우선한다.
- route membership은 약한 prior로만 사용하고 query intent와 Fact predicate 일치를
  주 점수로 사용한다.
- Agent prompt에는 후보별 query term, Fact predicate, route 여부를 compact basis로
  제공한다.

S1–10 고정-cache audit에서 평균 후보 수는 이전 개선판 4.10에서 3.23으로 줄었고,
reference Tool 완전 포함은 기존 route 51/100에서 60/100, initial boundary는
57/100에서 66/100으로 증가했다. safety fallback을 포함한 exact hint/call은
19/112로 기존과 동일하다. 결과 artifact는
`ubuntu/evaluation/vehiclemembench-joint-planning-audit/s1-s10-v8`에 있다.

후보 reranker의 Scenario 6–10 동일-cache live 재평가에서는 ESM 0.52→0.58,
State F1 0.6558→0.7358, selector miss 16→12, discovery call 0.40→0.22로
개선됐다. 반면 불필요 호출은 21→26으로 증가했다. 실제 selector membership이 바뀐
task에서는 ESM 개선 4개, 악화 0개였으며 상세 내용은
[결과 문서](joint-tool-selection-s6-s10-results.md)에 기록한다.

Scenario 1–5 동일-cache live 재평가에서도 ESM 0.52→0.62, State F1
0.7163→0.7738, Tool F1 0.5653→0.6270, selector miss 13→8로 개선됐다.
불필요 호출도 26→22로 감소했으며, selector membership이 바뀐 33개 task에서는
ESM 개선 3개, 악화 0개였다. 다만 input token은 181,687→193,592로 약 6.6%
증가했고 multi-Fact call은 0개였다. 상세 내용은
[결과 문서](joint-tool-selection-s1-s5-results.md)에 기록한다.

## 주요 코드 경계

- 검색·query context: `ubuntu/src/palmclaw_ubuntu/fact_memory_retrieval.py`
- 기존 route: `ubuntu/src/palmclaw_ubuntu/memory_router.py`
- 기존 Fact별 hint/binding: `ubuntu/src/palmclaw_ubuntu/tool_memory_execution.py`
- 제품 Agent 연결: `ubuntu/src/palmclaw_ubuntu/agent.py`
- VehicleMemBench 연결: `ubuntu/src/palmclaw_ubuntu/vehicle_bench/runner.py`
- Fact 모델·저장 계약: `ubuntu/src/palmclaw_ubuntu/models.py`,
  `ubuntu/src/palmclaw_ubuntu/storage.py`
