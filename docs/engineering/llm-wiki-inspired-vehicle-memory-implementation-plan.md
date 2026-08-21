# LLM-Wiki-inspired Vehicle Memory 구현 계획

상태: `6회차 평가 완료 · 현재 형태 채택 보류`

작성일: 2026-08-10

참고: [Retrieval as Reasoning: Self-Evolving Agent-Native Retrieval via
LLM-Wiki](../assets/2605.25480v2.pdf)

## 1. 목표

LLM-Wiki의 핵심인 `search → read → link-follow → sufficiency check`를
VehicleMemBench의 두 Memory 방식에 적용한다.

1. `Summary-gated Wiki-inspired`: Recursive Summary를 기본 경로로 유지하고,
   Summary가 불충분한 질문에서만 과거 Summary version을 단계적으로 탐색한다.
2. `Post-normalized Fact-Wiki-inspired`: 기존 versioned Fact, condition,
   evidence와 Tool capability를 연결된 page로 노출하고 Agent가 필요한 관계만
   따라간다.

논문 전체를 재현하거나 별도 범용 Wiki를 새로 생성하지 않는다. 기존 Memory
artifact를 deterministic한 **가상 Wiki page**로 변환해 Memory 생성 LLM 비용을
거의 늘리지 않는 on-device adaptation을 목표로 한다.

## 2. 방법론 경계

- 새 profile:
  - `cloud_recursive_summary_gated_wiki`
  - `cloud_post_normalized_fact_wiki`
- 기존 `cloud_recursive_summary`와
  `cloud_schema_informed_recursive_assisted_fact_patch`의 코드, cache와 결과는
  변경하지 않는다.
- Gold Memory, Gold Tool, Gold argument와 Quiz 성공 여부는 page 생성, gate,
  traversal 및 Error Book에 사용하지 않는다.
- 원본 History 전체를 Quiz Agent에 노출하지 않는다.
- 논문 재현이 아니라 `LLM-Wiki-inspired Vehicle Memory adaptation`으로 보고한다.
- MVP에는 LLM 기반 Wiki compilation과 Error Book을 넣지 않는다. traversal의
  효과가 확인된 뒤에만 Error Book을 추가한다.

## 3. 공통 설계

### 3.1 가상 Wiki 계약

새 모듈 후보: `ubuntu/src/palmclaw_ubuntu/vehicle_wiki.py`

- `VehicleWikiPage`
  - `page_id`, `page_type`, `title`, `aliases`, `tags`, `description`
  - bounded `content`, `source_ids`, `links`, `version`, `updated_at`
- `VehicleWikiLink`
  - `source_page_id`, `target_page_id`, `relation`, `evidence`, `confidence`
- `VehicleWikiSearchResult`, `VehicleWikiReadResult`, `VehicleWikiTraversalTrace`
- 허용 relation:
  - `about_entity`, `has_fact`, `applies_under`, `supported_by`
  - `supersedes`, `related_capability`, `implemented_by_tool`
  - `previous_update`, `next_update`

page와 link는 기존 immutable Memory snapshot에서 deterministic하게 생성한다.
policy version과 page fingerprint를 cache/trace에 기록하고, retrieval 중 기존
Memory DB를 변경하지 않는다.

### 3.2 Agent-native traversal

Agent에 다음 두 도구만 추가한다.

- `memory_wiki_search(query)`: title, alias, tag, description 우선 검색
- `memory_wiki_read(page_ids)`: page와 따라갈 수 있는 link를 batch read

기본 예산:

- search 최대 2회
- read page 총 4개, 한 번에 최대 3개
- link hop 최대 2
- 누적 Wiki context 최대 1,200 tokens
- 연속 empty search 2회 또는 충분한 entity·value·condition·capability가 모이면 종료

Traversal 장애, 예산 초과, 빈 결과에서는 각 profile의 기존 Memory context와 Tool
선택 경로로 fallback한다. query-time page 선택과 읽기 순서는 case trace에 남긴다.

## 4. 두 profile의 구체적 구성

### 4.1 Summary-gated Wiki-inspired

```text
Query + Final Recursive Summary
→ local coverage/complexity gate
→ 충분: 기존 Summary-only Agent 경로
→ 불충분: Summary update page 검색·읽기·link-follow
→ evidence sufficiency 확인 → Tool 실행
```

- final Summary는 `overview` page로 사용한다.
- 기존 Recursive Summary의 update version 사이 sentence delta를 update page로
  만든다. 새 LLM 호출은 하지 않는다.
- update page는 날짜·entity alias·vehicle capability term과 앞뒤 update에
  deterministic하게 연결한다.
- 최신 update를 우선하며, 오래된 값에는 이후 correction 가능성을 명시한다.
- gate는 multi-entity, comparison, 조건부 요청, correction 표현, Summary의
  entity/value coverage 부족만 사용한다. 특정 task literal은 사용하지 않는다.
- gate가 닫힌 task는 기존 Recursive Summary와 동일한 prompt/context를 유지한다.

이 구성은 final Summary에서 사라진 중간 Fact를 version page에서 회수하되, 모든
질문에 traversal 비용을 부과하지 않는 것을 목표로 한다.

### 4.2 Post-normalized Fact-Wiki-inspired

```text
Query → 기존 Fact Top-k seed
→ entity/fact/condition/evidence/capability page read
→ supersedes·related capability·Tool link follow
→ sufficient evidence bundle → 기존 desired-state/Tool planning → 실행
```

- active Fact를 entity page와 fact page에 deterministic하게 투영한다.
- `identity_conditions`, `applicability`, source evidence와 version chain을 별도
  page/link로 보존한다.
- 기존 post-normalized ontology와 Tool binding 결과로 capability 및 Tool page
  link를 만든다. 새로운 Gold-derived mapping은 추가하지 않는다.
- 기존 Top-k는 첫 seed이자 fallback으로 유지한다. traversal은 seed에서 누락된
  관련 entity, condition, superseding Fact와 Tool capability를 복구하는 데만 쓴다.
- 여러 Fact가 모이면 기존 Memory Bundle/desired-state planner에 한 번에 전달한다.

이 구성은 현재 병목인 multi-Fact 회수와 Fact→Tool semantic gap을 직접 대상으로
한다. Fact 생성·validation pipeline 자체는 변경하지 않는다.

## 5. 구현 회차

### 1회차 — 공통 page/link 계약

- [완료] 가상 Wiki dataclass, deterministic ID, serializer와 fingerprint 구현
- [완료] Summary update page builder와 Fact page builder fixture 작성
- [완료] dangling link, cycle, source provenance, token budget 단위 테스트

완료 기준: 같은 snapshot에서 항상 같은 page/link/fingerprint가 생성되고 Gold 및
원본 History 전체가 page에 들어가지 않는다.

구현 결과:

- `vehicle_wiki.py`에 immutable page/link 및 search/read/traversal trace 계약을
  추가했다.
- Recursive Summary version의 sentence delta를 overview/update page로 만들고,
  Post-normalized Fact를 entity/fact/condition/evidence/capability/Tool page로
  투영하는 deterministic builder를 추가했다.
- ordered relation의 cycle, dangling/self/duplicate link, source provenance와 page
  token 상한을 공통 validator가 거절한다.
- builder 입력은 기존 Summary version 또는 Fact record와 bounded evidence뿐이며
  dataset Quiz·Gold 필드를 받지 않는다.
- 타깃 테스트 `6 passed`, 관련 Ruff 검사 통과.

### 2회차 — Search/read/traversal 기반

- [완료] local lexical search와 선택적 기존 embedding score 재사용
- [완료] bounded batch read, hop/search/page/token 예산과 종료 이유 구현
- [완료] Agent Tool registry 연결 전 Fake traversal 및 fallback 테스트

완료 기준: direct, 2-hop, empty search, budget exhaustion과 tool error 경로가 모두
결정적으로 종료된다.

구현 결과:

- `vehicle_wiki_retrieval.py`에 immutable local index와 상태 기반 traversal
  session을 추가했다. title/alias/tag/description/content에 구조별 가중치를 두며,
  기존 embedding score가 전달된 경우에만 이를 보조 신호로 재사용한다.
- `search ≤ 2`, `read page ≤ 4`, `batch ≤ 3`, `hop ≤ 2`, Wiki context
  `≤ 1,200 tokens`를 기본값으로 강제하고 각 종료 이유를 trace에 기록한다.
- link-follow는 별도 암묵적 검색이 아니라, 읽은 page가 노출한 link의 page ID를
  다음 `read`에서 선택하는 방식이다. 충분성 판단 전 실패·빈 검색·예산 초과는
  원래 profile context로 fallback한다.
- direct read, 2-hop, empty search patience, page/batch/token 예산, missing page와
  tool error를 포함한 신규 테스트 `6 passed`; 공통 Wiki 및 기존 관련 회귀까지
  `53 passed`, Ruff 검사 통과.
- 실제 Agent Tool registry와 profile wiring은 계획대로 3회차부터 수행한다.

### 3회차 — Summary-gated profile

- [완료] Summary coverage/complexity gate와 update-page traversal 연결
- [완료] 기존 Summary-only 경로 불변성, gate-open/closed trace 추가
- [완료] profile, CLI, cache signature와 Fake VehicleMemBench E2E 추가

완료 기준: gate-closed case는 기존 context hash와 동일하고, gate-open case만 추가
Wiki Tool을 사용한다.

구현 결과:

- 새 profile `cloud_recursive_summary_gated_wiki`는 기존
  `recursive_summary` snapshot/cache를 그대로 재사용한다. 가상 page fingerprint와
  gate/runtime policy만 별도 cache signature로 남기므로 기존 Summary 생성 호출은
  증가하지 않는다.
- local gate는 multi-entity, comparison, condition, correction/history 표현과
  과거 Summary에는 있지만 final Summary에는 없는 entity·setting term만 사용한다.
  Quiz Gold, Gold Tool/argument 및 원본 History는 gate 입력에 포함하지 않는다.
- gate-closed에서는 기존 Recursive Summary content가 byte 단위로 같고 Agent prompt와
  Tool registry도 기존 경로를 유지한다. gate-open에서만
  `memory_wiki_search`와 `memory_wiki_read`가 Agent에 노출된다.
- Wiki 호출은 Vehicle Tool prediction과 공식 채점에서 제외하며, search/read 수,
  선택 page, token, 종료 이유와 fallback 여부를 task memory trace에 기록한다.
- gate-open/closed snapshot, 실제 Wiki Tool 호출 후 Vehicle Tool 실행을 포함한 Fake
  E2E와 관련 회귀 총 `74 passed, 1 skipped`; Ruff 검사 통과.

### 4회차 — Post-normalized Fact-Wiki profile

- [완료] Fact/entity/condition/evidence/version/capability page projection 구현
- [완료] 기존 Top-k seed, linked expansion과 desired-state planner 연결
- [완료] 잘못된 cross-entity link와 superseded Fact 선택 방지 테스트

완료 기준: 모든 선택 Fact가 원본 record/evidence로 역추적되고 traversal 실패 시
기존 Post-normalized 결과로 안전하게 fallback한다.

구현 결과:

- 새 profile `cloud_post_normalized_fact_wiki`는 현재 Post-normalized
  Recursive-assisted Fact snapshot과 기존 Top-k retrieval을 그대로 seed/cache로
  재사용한다. 별도 Fact 생성 LLM이나 embedding 호출을 추가하지 않는다.
- versioned Fact를 entity/fact/condition/evidence/capability/Tool page로
  deterministic하게 투영한다. evidence page는 기존 record source quote와 message
  ID로, capability→Tool link는 기존 Tool ontology로만 생성한다.
- 기존 retrieval candidate의 embedding score를 Wiki search에 재사용하며, active이고
  condition-compatible한 same-entity linked Fact만 최대 4개 Fact 범위에서 기존
  desired-state planner에 전달한다. cross-entity와 superseded Fact는 planner
  expansion에서 제외한다.
- Agent에는 공통 `memory_wiki_search/read`만 추가되고, Wiki 호출은 공식 Vehicle Tool
  prediction/채점에서 제외된다. page fingerprint, seed/expanded record, traversal
  token·종료 이유·fallback을 `fact_wiki` trace에 기록한다.
- projection/runtime 구성 실패 시 Wiki Tool과 planner expansion을 제거하고 기존
  Post-normalized Top-k content/Fact hint 경로로 복귀한다.
- projection, semantic score reuse, cross-entity/superseded/condition 방지, 실패
  fallback과 Fake VehicleMemBench E2E를 포함한 관련 회귀 `99 passed, 1 skipped`;
  Ruff 검사 통과.

### 5회차 — 통합 계측과 회귀 검증

- [완료] traversal call, hop, read page, selected token,
  sufficiency/termination 이유 집계
- [완료] Idle Memory LLM/embedding과 online retrieval/Agent token·latency 분리
- [완료] privacy audit, immutable snapshot, cache resume와 기존 profile 회귀 테스트
- [완료] Ruff 및 Ubuntu 전체 테스트 수행

완료 기준: 기존 profile 결과/cache가 바뀌지 않고 새 profile의 비용이 단계별로
재현 가능하게 보고된다.

구현 결과:

- 기존 `generation`, `retrieval`, Agent 합계 필드는 유지하고 suite report에
  `idle_memory_llm`, `idle_memory_embedding`, `online_memory_retrieval`,
  `quiz_agent` stage를 추가했다. 각 stage는 호출 수, 입력/출력 token, provider 또는
  관측 latency와 설정된 단가 기준 비용을 분리해 기록한다.
- profile별 online retrieval에는 기존 base Memory context/token·latency와 Wiki read
  context/token·Tool latency를 따로 기록한다. retrieved context token은 이후 Agent
  input에 포함되는 진단량이므로 Agent billing token과 더하지 않는다는 표식과 보고서
  설명을 추가했다.
- Wiki 계측은 실제 Tool 호출·오류·latency, 성공 step, search/read 호출과 read page,
  선택 page/token, hop, fallback, sufficiency rate 및 모든 termination reason을
  task trace에서 결정적으로 집계한다.
- 두 Wiki profile에서 동일 query의 page cache signature와 Memory fingerprint가
  유지됨을 검증했다. Summary-gated run은 빈 Agent model로 checkpoint resume되어
  추가 호출 없이 같은 결과를 재사용했고, 새 artifact privacy audit와
  `raw_history_included=false`도 통과했다.
- 관련 Wiki/Memory 회귀, provider stage 단위 테스트와 Ubuntu 전체 테스트가 모두
  통과했으며 1개 환경 의존 테스트만 skip되었다. 전체 Ruff 검사도 통과했다.

### 6회차 — 평가와 채택 판정

1. [완료] network-free Fake E2E
2. [완료] Scenario 6 10-task smoke
3. [완료] 기존 Memory cache를 고정한 Scenario 6–10 50-task paired E2E
4. [중단] 각 profile의 Quiz Agent 3회 반복
5. [중단] structure-only, progressive traversal, gate ablation

전체 평균뿐 아니라 Summary-only 실패, multi-Fact, identity conflict, condition,
correction 및 compound Tool call subset을 따로 보고한다.

구현·평가 결과:

- 실제 Cloud smoke에서 OpenAI strict Tool schema의 `uniqueItems` 비호환과 shared
  condition page 충돌을 발견해 수정했다. 수정 후 전체 50-task run에서 schema 및
  projection fallback은 발생하지 않았다.
- Summary baseline/Wiki ESM은 `0.68/0.66`, Agent token은
  `203,537/274,405`였다. gate-open 13개 중 실제 Tool 사용은 9개였고 새 성공/회귀가
  `1/1`로 순개선이 없었다.
- Post-normalized baseline/Fact-Wiki ESM은 `0.64/0.70`, Tool F1은
  `0.607/0.679`였다. 실제 Tool 사용 17개에서 새 성공/회귀 `3/1`의 신호가 있었지만
  Agent token이 `196,765→374,444`로 `90.3%` 증가했다.
- 두 profile 모두 사전 효율 조건을 통과하지 못했다. 추가 반복과 controlled
  ablation은 비용 대비 채택 가능성이 없어 중단했으며 Error Book도 추가하지 않는다.
- 상세 결과와 artifact는
  [Scenario 6–10 결과](llm-wiki-inspired-vehicle-memory-s6-s10-results.md)에 기록했다.

## 6. 예상 결과와 중단 기준

다음 수치는 구현 목표 범위이며 결과를 보장하지 않는다. 기준은 현재
Scenario 6–10 단일 run과 동일 모델 설정이다.

| 항목 | Recursive Summary 기준 | Summary-gated 예상 | Post-normalized 기준 | Fact-Wiki 예상 |
| --- | ---: | ---: | ---: | ---: |
| ESM | 0.62 | **0.64–0.68** | 0.60 | **0.64–0.70** |
| State F1 | 0.772 | 0.78–0.81 | 0.745 | 0.77–0.82 |
| Tool F1 | 0.653 | 0.65–0.68 | 0.606 | 0.63–0.68 |
| Argument exact | 0.56 | 0.56–0.60 | 0.48 | 0.52–0.60 |
| Memory LLM tokens | 0.649M | **+0–5%** | 약 3.944M | **+0–3%** |
| Agent tokens | 0.195M | +10–30% | 0.193M | +20–50% |
| 평균 Quiz latency | 4.79초 | +0.5–1.5초 | 4.62초 | +1–3초 |

기대 효과는 single-hop 전체가 아니라 기존 방식이 놓친 multi-Fact, correction,
identity/condition 및 Tool-binding task에 집중된다. VehicleMemBench에 실제
compositional retrieval 수요가 적으면 전체 ESM은 오르지 않을 수 있다.

채택 조건:

- Summary-gated: 3회 평균 ESM `+0.02` 이상, gate-open subset 순개선,
  Agent token 증가 30% 이하
- Fact-Wiki: 3회 평균 ESM `+0.03` 이상 또는 Tool/Argument 지표의 안정적 개선,
  Agent token 증가 50% 이하
- 공통: 기존 성공 task의 안정적 회귀가 2건을 넘지 않고 privacy audit 통과

중단 조건:

- traversal이 바꾼 task에서 순개선이 없거나 Agent sampling 변화만 관찰됨
- 불필요한 Wiki 호출이 전체 task의 40%를 넘음
- raw History 또는 Gold-derived 정보가 page/trace에 포함됨
- virtual page인데도 Memory 생성 LLM token이 5% 이상 증가함

## 7. 후속 단계 — Error Book

두 profile 중 하나라도 채택 조건을 통과한 경우에만 Error Book을 추가한다.

- build-time validator에서 반복되는 dangling link, unsupported source,
  contradiction, stale version과 invalid Tool link를 기록한다.
- 구조 오류는 deterministic repair를 우선하고, 같은 원인이 여러 batch에서
  반복될 때만 bounded LLM repair를 수행한다.
- Quiz 결과, scorer, Gold call과 task별 정답은 Error Book 입력에서 제외한다.
- `without Error Book`과 `with Error Book`을 별도 cache/policy version으로
  비교해 정확도 이득과 추가 Idle 비용을 함께 보고한다.

Error Book은 traversal의 효과를 확인하기 전에는 구현하지 않는다. 초기 결과가
없으면 높은 compilation 비용만 추가될 가능성이 있기 때문이다.
