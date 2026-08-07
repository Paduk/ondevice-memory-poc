# Scenario 1–10 방법론별 구조적 병목 비교

상태: `100-task 비교 분석 완료 — 단편 수정 보류, 구조적 개선 방향 선정`

## 범위

Recursive Summary와 최신 Post-normalized Recursive-assisted Ours를
Scenario 1–10의 동일한 100 task에서 비교했다.

| 방법 | Scenario 1–5 | Scenario 6–10 |
| --- | --- | --- |
| Recursive Summary | `cf3067dc-ebca-4d50-92a7-bea257cc1950` | `29518f79-654d-4da0-865a-17ca633b6e70` |
| 최신 Ours | `a9215a40-ff50-4722-9173-f33d1ad02f4c` | `15f5a789-8dd5-4d6c-9190-887290bd5bff` |

두 구간은 같은 benchmark commit, Quiz model, Memory model, embedding 설정을
사용한다. 아래 수치는 구간별 단일 run을 합친 것으로, Agent 변동을 제거한
유의성 검정은 아니다. 구조적 주장은 ESM 하나가 아니라 deterministic retrieval,
selector, hint trace와 reasoning-type 반복 패턴을 함께 근거로 한다.

## 100-task 결과

| 방법 | ESM | State F1 | Tool F1 | Argument exact | Recall@k |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Recursive Summary** | **0.60** | **0.789** | **0.658** | **0.55** | N/A |
| 최신 Ours | 0.56 | 0.726 | 0.584 | 0.48 | 0.608 |

Task별 결과는 두 방법 모두 성공 43건, Recursive만 성공 17건, Ours만 성공
13건, 둘 다 실패 27건이다. 성공 여부가 서로 교체된 task가 30개여서 두 방법은
같은 정보를 다른 방식으로 실패하고 있다.

## 방법론 차이

```text
Recursive Summary
History → 매일 전체 Summary 재작성 → 최종 Summary 전체 → Agent → Tool discovery/호출

최신 Ours
History → 기본 Fact → Summary 기반 누락 Fact 보조 생성 → 정규화·versioned DB
Query → Top-5 Fact 검색 → query route + Fact별 실행 hint → Agent → Tool 호출
```

최신 Ours에서 Recursive Summary는 저장 시 누락 후보를 떠올리는 checklist로만
사용된다. 모든 100 task에서 `recursive_summary_agent_exposure=false`였으며 Quiz
Agent는 Summary를 보지 않고 검색된 Fact와 hint만 본다. 따라서 Summary가 갖고
있던 인물·상황·복합 설정의 관계가 원자 Fact에 표현되지 않으면 실행 시점에는
다시 사용할 수 없다.

| 관점 | Recursive Summary | 최신 Ours |
| --- | --- | --- |
| 저장 단위 | 사용자별 자연어 bullet의 전체 요약 | predicate/value/identity/applicability/evidence Fact |
| 갱신 | 전체 Summary replacement | ADD/UPDATE/version/linking |
| 질의 시 Memory | 최종 Summary 전체 | query별 Top-5 Fact |
| Tool 선택 | Agent가 discovery | schema router + Fact hint + Agent discovery |
| 인자 변환 | Agent의 자연어 추론 | Fact별 deterministic mapping 후 Agent 보완 |
| 강점 | 관계·문맥을 자연어로 함께 유지, 유연한 의미 해석 | provenance·version·검증 가능성, 적은 discovery |
| 약점 | rewrite 망각·오귀속·모호한 실행 | Fact 원자화·검색 누락·후보 제한·필수 인자 결합 실패 |

## 구조적 병목

### 1. Memory 보존 방식의 서로 다른 손실

Recursive Summary는 query-time retrieval miss가 없지만 매 갱신마다 전체 Summary를
다시 쓴다. Scenario 6–10 reviewed audit에서 50건 중 16건은 정답 근거가 최종
Summary에 불충분했다. 최초 미저장 9건, 중간 저장 후 소실 2건, 오귀속·왜곡
5건이었다. 길이 제한 truncation은 0건이므로 용량보다 rewrite의 비결정성이
원인이다.

최신 Ours는 evidence와 version을 보존하지만 저장·검색 단계가 추가된다. 100건의
retrieval scorer 기준으로 full Recall 42건, partial 38건, zero 20건이었다.

| 최신 Ours Recall | Task | ESM 성공 | 성공률 |
| --- | ---: | ---: | ---: |
| Full | 42 | 30 | 71.4% |
| Partial | 38 | 19 | 50.0% |
| Zero | 20 | 7 | 35.0% |

즉 Summary는 `rewrite loss`, Ours는 `extraction + retrieval loss`를 가진다. 어느
한쪽 표현만 선택해도 Memory coverage가 완전히 해결되지 않는다.

### 2. Ours는 관계를 저장하지 않고 개별 Fact만 저장한다

Fact에는 값·대상·조건은 있지만 다음 관계를 직접 표현하는 단위가 없다.

- 같은 과거 사건에서 함께 사용한 여러 설정
- 누가 요청했고 누가 운전했는지의 관계
- 이전 값에서 얼마나 올리거나 내리는지의 연산
- switch와 setting, source와 play처럼 함께 실행해야 하는 호출 묶음
- correction 전후의 사건 맥락

이 차이는 reasoning type에서 나타난다.

| Reasoning type | Task | Recursive ESM | 최신 Ours ESM |
| --- | ---: | ---: | ---: |
| Conditional constraint | 20 | 0.500 | 0.500 |
| Coreference resolution | 19 | **0.526** | 0.368 |
| Error correction | 17 | 0.706 | 0.706 |
| Preference conflict | 28 | 0.714 | **0.750** |
| State shift | 16 | **0.500** | 0.375 |

Ours가 explicit update·versioning과 잘 맞는 preference conflict에서는 조금 낫고,
인물·시간·이전 상태의 연결이 필요한 coreference와 state shift에서는 크게 낮다.
이는 canonical value 자체보다 Fact 사이 관계와 temporal context의 부재가 더 큰
문제라는 신호다.

### 3. 질문 기반 Tool route와 Memory 기반 검색이 분리돼 있다

Ours는 Fact 검색이 memory-first여도 Tool route는 query 중심으로 계산한다. 이후
Fact hint가 일부 Tool을 보강하지만 두 결과를 하나의 공동 후보 탐색으로 풀지
않는다.

- 100건 중 selector가 reference Tool 하나 이상을 누락: 49건
- Fact hint까지 합친 initial Tool boundary에서도 reference Tool 누락: 43건
- selector 누락 49건 중 hint가 정답 Tool을 boundary에 복구: 6건
- initial Tool이 완전한 57건의 ESM: 39/57, 68.4%
- initial Tool이 누락된 43건의 ESM: 17/43, 39.5%

Agent discovery가 17건을 보완했기 때문에 selector miss가 곧 실패는 아니다.
하지만 query와 Fact capability를 별도 경로로 처리하는 현재 구조가 Agent에게
불필요한 복구 작업을 넘긴다.

### 4. Fact별 late binding은 계획 단위가 아니다

현재 hint builder는 선택된 Fact를 각각 하나의 Tool에 연결하고 schema validation
한다. 여러 Fact나 query의 runtime 정보를 모아 하나의 desired-state plan을 먼저
만들지 않는다.

- emitted hint: 189개
- mapping rejection: 311개
  - `incomplete_or_invalid_arguments`: 257
  - `no_compatible_tool`: 41
  - `ambiguous_tool_mapping`: 13
- hint가 하나도 없는 task: 19/100
- primary diagnostic가 memory argument mapping rejection: 15/100

Rejection에는 정답과 무관한 검색 Fact도 포함되므로 311건 전부가 오류는 아니다.
그러나 대부분이 필수 argument를 한 Fact에서 완성하려다 실패한 사실은 현재
mapper의 단위가 너무 작다는 것을 보여준다.

### 5. 두 방법 모두 복합 실행 계획이 약하다

Gold call이 2개 이상인 task는 11개다.

| 방법 | 복합 task ESM | 단일 task ESM |
| --- | ---: | ---: |
| Recursive Summary | 2/11, 18.2% | 58/89, 65.2% |
| 최신 Ours | 3/11, 27.3% | 53/89, 59.6% |

Summary는 호출 묶음을 자연어로 남길 수 있지만 Agent가 일부 Tool을 생략한다.
Ours는 원자 Fact와 개별 hint는 만들지만 `switch + setting`, `source + play`를 하나의
계획으로 묶지 않는다. 표현 방식과 무관하게 복합 호출을 명시하는 중간 계획이 없다.

### 6. 공통 핵심은 Memory와 Tool 사이의 semantic gap이다

Recursive Summary의 primary diagnostic는 Agent Tool omission 31건, argument
mismatch 9건이었다. 최신 Ours는 selector miss 25건, memory argument mapping
rejection 15건이었다. 이름은 다르지만 모두 다음 변환을 안정적으로 수행하지
못한 결과다.

```text
기억된 사용자 의도
  → 이번 요청에 적용할 desired state
  → 필요한 하나 이상의 capability
  → Tool과 필수 argument
  → 실행 순서와 검증
```

Recursive는 이 전체 변환을 Agent에게 맡기고, Ours는 일부를 router와 per-Fact
hint로 나눴다. 둘 다 `desired state / action plan`을 명시적인 자료구조로 갖고 있지
않다. 이것이 두 방법에 공통인 가장 큰 구조적 문제다.

## 효율 차이

| 100-task 합계 | Recursive Summary | 최신 Ours |
| --- | ---: | ---: |
| Memory generation calls | 472 | 2,270 |
| Memory generation input tokens | 1,229,148 | 7,558,891 |
| 평균 query Memory context | 약 325 tokens | 약 425 tokens |
| Agent input tokens | 392,846 | 360,547 |
| 평균 discovery calls | 1.10 | 0.34 |
| 불필요한 vehicle calls | 38 | 48 |

Ours는 discovery를 69%, 전체 Agent input을 8.2% 줄였지만 Memory 생성 input은
6.15배다. 구조화된 Top-5 Fact context도 최종 Summary 전체보다 평균 token이 더
많다. 잘못된 candidate·hint 때문에 불필요한 vehicle call도 더 많다. 현재 구조는
on-device 실행 전 단계 비용과 품질 모두 최적화되지 않았다.

## 구조적 개선 방향

단편적인 lexical 규칙보다 다음 중간 계층을 추가하는 방향이 두 방법의 장점을
결합한다.

```text
History
  → Atomic Facts + Evidence + Version
  → Event Bundle: 함께 발생한 설정·인물·조건·correction 관계

Query + 현재 차량 상태
  → Memory Bundle Retrieval
  → Desired State / Action Intent Plan
  → Public Tool schema 기반 Tool Plan
  → Agent는 남은 ambiguity만 처리
```

### A. Atomic Fact와 Event Bundle을 함께 보존

현재 Fact DB는 유지하되 같은 source event의 Fact에 `bundle_id`, participant role,
순서, correction relation을 보존한다. Summary는 authoritative store가 아니라
누락 탐지·bundle 검색용 semantic index로만 사용한다. 이렇게 하면 Recursive의
관계 보존과 Ours의 provenance·versioning을 함께 가져갈 수 있다.

### B. Query와 Fact를 함께 사용하는 capability selector

후보 Tool은 `query route ∪ retrieved Fact capability ∪ bundle capability`로 만든다.
query lexical match나 routed bonus 하나가 후보를 막지 못하게 하고, public schema로
필수 slot을 채울 가능성이 있는 후보를 공동 평가한다.

### C. Multi-Fact desired-state planner

Tool을 먼저 고르지 않고 다음 구조를 먼저 만든다.

- 대상 device/capability
- operation: restore/set/increase/decrease/enable/disable
- remembered value와 현재 query modifier
- requester, occupant role, zone/side
- applicability와 evidence
- 함께 실행할 다른 desired states

그다음 Fact, query, 현재 차량 상태에서 필수 slot을 결합하고 schema validation한다.
근거가 없는 target·zone·enum은 추측하지 않고 ambiguity로 남긴다.

### D. 복합 호출을 하나의 plan으로 평가

개별 hint 정확도뿐 아니라 plan completeness, 불필요한 state change, dependency와
실행 순서를 검증한다. 이는 두 방법 모두 20%대에 머문 복합 task를 직접 겨냥하지만
특정 Tool 이름이나 Gold 값을 규칙에 넣지 않는다.

## Overfit 방지선

1. task ID, 인물 이름, Gold argument literal을 runtime 규칙에 사용하지 않는다.
2. Tool 후보와 argument type은 public Tool schema와 Memory evidence에서만 만든다.
3. `unspecified→특정 target`, 누락 zone default, 비공개 simulator enum을 추가하지
   않는다.
4. Scenario 1–10은 원인 분석·개발 집합으로만 사용하고, 선택한 구조는 먼저 100-task
   offline stage replay에서 회귀를 확인한다.
5. 최종 E2E 주장은 아직 사용하지 않은 Scenario에서 고정 설정으로 검증한다.
6. ESM만 튜닝하지 않고 Fact coverage, bundle recall, Tool candidate coverage,
   required-slot completion, plan completeness를 각각 측정한다.

## 다음 단계

바로 router 단어 규칙을 수정하지 않는다. 먼저 기존 cache를 사용해 LLM 호출 없이
100 task에 대해 다음 표를 만든다.

1. Gold 근거가 active Fact에 존재하는가
2. 존재하면 검색됐는가
3. 검색됐으면 관련 bundle이 완전한가
4. reference capability가 candidate Tool에 있는가
5. required argument를 evidence/query/state로 채울 수 있는가
6. 단일 또는 복합 plan이 완전한가

이 표에서 가장 많은 task를 task-independent하게 회복하는 계층부터 PoC한다.
