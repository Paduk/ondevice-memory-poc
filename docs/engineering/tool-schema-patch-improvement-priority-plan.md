# Tool-Schema Patch 개선 우선순위 계획

상태: `P6 10% 비교 평가 완료 — Hybrid 우세, Ours coverage 개선 필요`

최종 검토: 2026-07-28

## 목표

기존 Scenario 1 cache를 재사용해 병목을 먼저 분리하고, 결과에 따라
P1~P6의 실제 수행 순서를 결정한다. 한 번에 하나의 변경만 적용한 뒤
같은 실험으로 효과를 재검증한다.

## P0 — 원인 분리

Memory patch를 다시 생성하지 않고 다음 네 조건을 비교한다.

1. Gold Tool + Gold Memory: Agent·Context 실행 상한
2. Gold Tool + Patch Memory: Patch 생성·저장·검색 품질
3. Tool Selector + Gold Memory: Tool discovery 영향
4. Tool Selector + Patch Memory: 현재 E2E 품질

기존 2,438턴의 Patch Memory와 trace는 SQLite cache를 재사용한다.
P0에서는 Agent 평가 호출만 추가로 발생한다.

### P0 결과

Scenario 1의 동일한 10개 task와 기존 Patch cache를 사용했다.
Oracle에는 정답 argument가 아닌 정답 Tool schema만 제공했다.

| 조건 | ESM | State F1 | Tool F1 | Argument exact |
| --- | ---: | ---: | ---: | ---: |
| No Memory | 0.30 | 0.417 | 0.333 | 0.20 |
| Cloud Summary | 0.40 | 0.55 | 0.467 | 0.40 |
| Structured BM25 | 0.70 | 0.74 | 0.70 | 0.70 |
| Structured Embedding | 0.70 | 0.74 | 0.70 | 0.70 |
| Structured Hybrid | 0.70 | 0.74 | 0.70 | 0.70 |
| Tool Selector + Patch Memory | 0.40 | 0.55 | 0.467 | 0.40 |
| Gold Tool + Patch Memory | 0.50 | 0.65 | 0.567 | 0.50 |
| Tool Selector + Gold Memory | 0.90 | 0.98 | 0.933 | 0.80 |
| Gold Tool + Gold Memory | 0.90 | 0.98 | 0.967 | 0.90 |

기존 방법은 같은 dataset hash, Scenario 1, task ID 10개, Agent 모델을
사용한 이전 artifact에서 가져왔다. Argument exact는 저장된 Tool call을
동일한 기준으로 다시 계산했다. 실행 시점이 달라 0.1 단위 차이는 모델
변동 가능성을 함께 고려한다.

- Patch 모델 호출: 실행 전후 모두 2,456회로 추가 생성 없음
- Oracle Patch retrieval: 10개 중 8개가 empty, Recall@k 0.10
- 현재 Tool routing 영향: Oracle 적용 시 ESM +0.10
- Gold 조건의 유일한 ESM 실패는 2개 Tool 중 1개만 호출한 Agent 실패

관련 record가 DB에 있어도 alias·시간·상황 조건이 정확히 맞지 않아
`condition_mismatch`로 제거되는 사례가 다수였다. Patch record 자체가 없는
추출 누락도 함께 확인됐다.

### 병목 재분석

1. **Patch condition gate가 가장 큰 병목이다.**
   Structured와 Patch DB 모두 최소 하나의 정답 preference 값을 약 7/10
   task에서 보유했지만, Patch는 실제 추론에서 2/10 task에만 record를
   전달했다. Oracle Tool routing 후에도 8/10이 retrieval empty였다.
2. **Patch 추출 coverage도 부족하다.**
   auto-brightness, corrected ventilation, nephew의 ambient color 사례는
   active Patch record가 없었다. Condition을 고쳐도 이 task들은 남는다.
3. **Tool routing은 2차 병목이다.**
   Patch E2E 0.40에서 Gold Tool 적용 시 0.50으로 한 task만 회복했다.
4. **Agent의 multi-Tool 실행은 별도 병목이다.**
   Gold Memory와 Gold Tool을 모두 줘도 task 03에서 필요한 두 Tool 중
   하나만 호출해 상한이 0.90에 머물렀다.
5. **현재 표본에서는 retrieval scorer 차이가 보이지 않는다.**
   BM25, Embedding, Hybrid가 동일한 0.70과 같은 task 결과를 냈다.
   Structured의 남은 실패는 주로 record 누락과 Agent 실행이었다.

결론: 현재 주 병목은 `condition matching → extraction coverage →
Tool routing → Agent multi-Tool 실행` 순서다.

Artifact:
`ubuntu/evaluation/vehiclemembench/4e221955-b0a8-4cb7-9bd2-11e6bc42d02b`

비교 artifact:

- Baseline:
  `ubuntu/evaluation/vehiclemembench/5d7214f8-1488-407e-8a03-16c2314e256a`
- Summary/Structured:
  `ubuntu/evaluation/vehiclemembench/d41fdeca-2400-414b-8b10-35565e14fa82`

## 확정 우선순위

1. **P3 Identity·condition matching**: 이미 있는 record가 검색되지 않는 문제
2. **P1 Batch Patch**: 누락된 record와 호출 비용 개선
3. **P4 Validation**: rejected patch 20건의 과도한 거절 여부 조정
4. **P2 Routing·Retrieval**: E2E와 Oracle 간 0.10 차이 개선
5. **P5 Agent·Context**: multi-Tool 누락과 argument 변환 개선
6. **P6 확대 평가**

## P1·P3 구현 결과

### P1 — 실제 Batch Patch

- 기본 상한: 32 turns 또는 4,096 source tokens
- 여러 job을 하나의 Cloud LLM 호출로 처리
- 반환 순서와 무관하게 evidence message 순서로 DB 적용
- batch 외부 message를 evidence로 인용할 수 없도록 제한
- batch run과 원본 job들을 연결해 재시작 복구
- 전체 Tool ontology는 유지하되 slot schema를 압축
- prompt version을 `vehicle-tool-memory-patch-batch-v2`로 변경해
  기존 v1 cache와 분리

Scenario 1의 2,438줄은 재시도 전 기준 77 batch다. 기존 2,438회
turn-level 호출과 비교하면 최초 호출 수가 약 96.8% 감소한다.

### P3 — Alias·Condition 호환

- record 외부 구조를 `entity_id`, `identity_conditions`, `applicability`,
  `identity_family_key`로 분리 저장
- 원래 `conditions`도 호환성과 audit를 위해 유지
- 별도 alias 사전 없이 기존 record의 이름으로 alias 학습
- `Patricia`와 `Patricia Garcia`처럼 token 포함 관계가 유일할 때만
  canonical 이름으로 통합
- 동명이인처럼 여러 후보가 가능한 경우 자동 통합하지 않음
- 시간과 날씨는 충돌 방지를 위해 hard condition 유지
- `12:00 PM`과 `daytime` 같은 시간 표현을 bucket으로 정규화
- location·context 등 자유형식 조건은 hard reject하지 않고
  BM25·embedding ranking에 반영
- 기존 SQLite record는 migration 시 위 필드를 deterministic하게 backfill

기존 19개 record의 복사본으로 수행한 무비용 condition-stage 진단에서
Oracle Tool로 검색 가능한 task가 2/10에서 7/10으로 증가했다.
이는 새 batch memory의 최종 E2E 성능이 아니라 condition gate 개선만
분리한 결과다.

### 다음 검증

새 v2 cache 생성에는 최초 시도 기준 약 77회의 Memory LLM 호출이
예상된다. 재평가 전 모델 단가와 최대 재시도 비용을 확인한 뒤 실행한다.

## P1·P3 v2 Live 평가

실행일: 2026-07-28

동일한 VehicleMemBench Scenario 1의 10개 task와 P0 네 조건으로 평가했다.

| 조건 | v1 ESM | v2 ESM | v1 State F1 | v2 State F1 | v1 Tool F1 | v2 Tool F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gold Tool + Gold Memory | 0.90 | 1.00 | 0.98 | 1.00 | 0.967 | 0.933 |
| Gold Tool + Patch Memory | 0.50 | 0.70 | 0.65 | 0.70 | 0.567 | 0.667 |
| Tool Selector + Gold Memory | 0.90 | 0.80 | 0.98 | 0.93 | 0.933 | 0.900 |
| Tool Selector + Patch Memory | 0.40 | 0.30 | 0.55 | 0.417 | 0.467 | 0.383 |

### Memory 생성

| 항목 | v1 | v2 |
| --- | ---: | ---: |
| Memory LLM 호출 | 2,456 | 82 |
| 입력 token | 24,675,484 | 932,081 |
| 출력 token | 103,602 | 8,424 |
| active record | 19 | 14 |
| Patch acceptance | 0.487 | 0.700 |

v2는 최초 77 batch와 validation 재시도 5회로 총 82회 호출됐다.
호출 수는 96.7%, 입력 token은 96.2% 감소했다.

### 확인된 병목

1. **P4 validation**
   - 잘못된 `start_char/end_char` 때문에 `invalid_evidence_span` 6건 발생
   - quote 자체는 원문에 정확히 존재했지만 offset만 틀린 사례
   - Patch 하나의 실패가 batch 전체를 rollback해 32 turns가 최종 실패
2. **P2 routing**
   - Oracle Patch ESM은 0.70이지만 현재 E2E는 0.30
   - E2E selected memory가 평균 0.2개에서 1.7개로 증가했으나
     broad domain routing이 관련 없는 record까지 제공
   - retrieval Recall@k는 0에서 0.25로 올랐지만 extra Tool 실행이 증가
3. **추출 coverage**
   - seat headrest, seat ventilation, nephew ambient color 등은 record가 없음
   - P4 이후에도 남으면 Patch prompt·batch extraction을 별도 개선

### 다음 순서

1. P4: exact quote가 존재하면 offset을 deterministic하게 보정 — 완료
2. P4: invalid Patch만 격리하고 같은 batch의 valid Patch는 적용 — 완료
3. P2: broad route 제한과 Tool-description semantic routing 개선 — 완료
4. 새 v3 cache로 동일 P0 평가 재실행

v2 Artifact:
`ubuntu/evaluation/vehiclemembench/b13e2726-c061-4c2f-875f-f1a96b914362`

## P4 Validation 구현 결과

- evidence quote가 원문에 정확히 한 번 존재하면 잘못된
  `start_char/end_char`를 자동 보정
- 원래 offset과 보정 offset을 proposal validation trace에 기록
- Cloud prompt는 offset을 계산하지 않고 `null`로 반환하도록 변경
- background batch는 patch별 transaction으로 유효 patch를 즉시 보존
- rejected patch가 인용한 source job만 retry하고 나머지는 completed 처리
- run 완료와 job 상태 변경을 같은 SQLite transaction으로 처리
- direct engine의 기존 strict atomic 모드는 기본값으로 유지
- Vehicle prompt version을 `vehicle-tool-memory-patch-batch-v3`로 변경해
  기존 v2 cache와 분리

기존 v2 cache의 `invalid_evidence_span` 6건을 read-only로 재검사한 결과,
6건 모두 exact quote가 원문에 한 번만 존재해 새 보정 규칙으로 복구
가능했다. 전체 135개 테스트는 134 passed, 선택 테스트 1 skipped다.

이 결과를 기준으로 다음 단계인 P2 semantic Tool routing을 구현했다.

## P2 Semantic Routing 구현 결과

- 전체 상황 설명보다 따옴표 안의 직접 요청문을 우선해 route
- `to`, `on` 같은 stopword와 `level`, `time` 같은 약한 단독 신호 제거
- domain 단위 확장 대신 Tool별 lexical score 계산
- Tool 이름·description·argument schema embedding과 lexical score 결합
- 선택된 Tool의 topic·slot만 SQLite hard partition으로 사용
- domain당 최대 2개, 전체 최대 4개 Tool로 후보 폭 제한
- Tool schema embedding을 SQLite에 영속 cache해 재시작 후 재사용
- routing focus, lexical·semantic score, 모델·오류·latency를 trace에 기록
- routing query vector를 memory ranking에서 재사용해 중복 query embedding
  호출 제거

동일 Scenario 1의 기존 10개 query를 사용한 router-only 진단:

| 항목 | 기존 | P2 |
| --- | ---: | ---: |
| Gold Tool 포함 task | 5/10 | 7/10 |
| 평균 route Tool 수 | 13.7 | 2.5 |
| 최대 route Tool 수 | 27 | 4 |

기존 v2 DB 복사본에서 Agent 호출 없이 retrieval만 재실행한 진단에서는
Recall@k가 0.25 → 0.35, 평균 선택 memory가 1.7 → 0.9로 변했다.
이는 공식 E2E 결과가 아니라 routing·retrieval 단계만 분리한 결과다.

남은 3개 route miss는 다음 유형이다.

- 직접 요청만으로 실제 Tool을 알기 어려운 암시적 preference
- 한 질문에서 숨겨진 두 Tool을 함께 실행해야 하는 경우
- 관련 memory를 읽어야 Tool 의미가 드러나는 경우

따라서 새 v3 cache로 소규모 P0 평가를 먼저 실행하고, 남는
Agent·Context·multi-Tool 문제는 P5에서 처리한다. 전체 138개 테스트는
137 passed, 선택 테스트 1 skipped다.

## P5 Agent·Context 구현 결과

- 선택된 Memory record를 `Tool name + arguments` 후보로 deterministic하게
  변환하고 원본 Tool JSON schema로 다시 검증
- required argument가 없거나 Tool mapping이 모호하면 Agent hint로 쓰지 않고
  rejection trace에 기록
- P2 selector의 Tool 목록을 Vehicle Agent 초기 Tool set에 연결
- selector 후보가 부족할 때만 기존 `list_module_tools` discovery를 fallback으로
  사용
- Agent prompt에 여러 독립 설정의 private checklist, 전체 실행 확인,
  성공한 호출의 반복 금지 규칙 추가
- 완료된 동일 Vehicle Tool 호출은 Runtime에서 한 번 더 차단
- `selector miss`, `discovery error`, `memory argument mapping rejection`,
  `agent Tool omission`, `agent argument mismatch`를 별도 진단 코드로 분리
- source turn과 evidence는 Agent prompt에 넣지 않고 기존 memory trace에만 유지

Gold Tool·argument는 위 변환과 preloading에 사용하지 않는다. Gold 정보는
실행 이후 scorer와 실패 원인 분류에서만 사용한다.

기존 v2 Scenario 1 cache의 active record 14개를 대상으로 한 무비용 정적
진단에서는 14개 모두 정확히 하나의 schema-valid Tool hint로 변환됐다.
이는 Agent E2E 성능 결과가 아니다.

검증 결과:

- 전체 145개 테스트 중 144 passed, 공식 checkout 선택 테스트 1 skipped
- coverage 87%
- Ruff 통과

다음 단계는 P4·P2·P5가 모두 반영되는 새 v3 cache 생성과 동일 P0/E2E
재평가다. 이 단계는 Memory Cloud 호출과 Agent Cloud 호출이 발생하므로
실행 전 예상 비용을 다시 확인한다.

## P4·P2·P5 v3 통합 평가

실행일: 2026-07-28

공식 VehicleMemBench commit `5ef3c48`, Scenario 1의 2,438 history와
동일한 10개 task를 사용했다. Agent는 `gpt-5.6-terra`, Memory patch는
`gpt-5.6-luna`, embedding은 `text-embedding-3-small`이다.

| 조건 | ESM | State F1 | Tool F1 | Argument exact |
| --- | ---: | ---: | ---: | ---: |
| Gold Tool + Gold Memory | 0.90 | 0.98 | 0.90 | 0.70 |
| Gold Tool + Patch Memory | 0.50 | 0.60 | 0.50 | 0.50 |
| Tool Selector + Gold Memory | 0.90 | 0.98 | 0.933 | 0.80 |
| Tool Selector + Patch Memory | 0.40 | 0.573 | 0.40 | 0.40 |

### v2 대비 E2E와 효율

`Tool Selector + Patch Memory` 기준:

- ESM: 0.30 → 0.40
- State F1: 0.417 → 0.573
- Argument exact: 0.20 → 0.40
- Agent input token: 32,147 → 17,994
- Agent output token: 1,709 → 1,302
- 평균 discovery 호출: 0.9 → 0
- 불필요 Vehicle 호출: 9 → 5

Agent 출력의 확률적 변동과 v2/v3 Memory record 차이가 함께 있으므로
성능 상승 전체를 P5 단독 효과로 해석하지 않는다. Gold Tool + Gold
Memory도 1.00에서 0.90으로 변했다.

### v3 Memory 생성

- 최초 예상과 동일한 77 batch 호출, 재시도 0회
- 2,438 job 전체 completed, failed job 0
- 입력 869,900 token, 출력 7,697 token
- proposal 17건 중 13건 적용, 4건 거절
- active record 13건, patch acceptance 0.765
- 47개 government-ID 패턴을 Cloud 전송 전에 redaction
- artifact 잔여 sensitive span 0

전체 실행량은 Agent 89회, Memory patch 77회, embedding 16회다.
입력 token은 Agent 77,800, Memory 869,900, embedding 3,168이고
출력 token은 Agent 4,017, Memory 7,697이다. Agent input 중 cached
token은 15,229다. 가격 환경변수가 설정되지 않아 USD는 산출하지 않았다.

### 남은 병목

1. **Extraction·validation coverage**
   - headrest, ventilation enabled/speed, display auto-brightness 후보 4건이
     `unsupported_value`로 거절됐다.
   - Tool target 값과 `on/off` 같은 동의 표현을 evidence가 지지하는지
     현재 validator가 충분히 유연하게 해석하지 못한다.
2. **Condition normalization**
   - `after dusk`를 `night`로 정규화하지 못해 8:00 PM reading preference가
     `condition_mismatch`로 제외됐다.
3. **Implicit multi-Tool selector**
   - task 00·03·04에서 질문만으로 드러나지 않는 Tool을 놓쳤다.
   - 실패 code 전체 집계에서 `tool_selector_miss`는 3건이다.
4. **JSON schema와 실제 Tool 제약의 차이**
   - 공식 schema가 일부 argument를 단순 string으로만 정의해
     `reading light`, `3D`가 JSON schema를 통과했지만 simulator에서
     거절됐다.
   - enum·정규화 또는 실행 전 semantic argument validator가 필요하다.
5. **Memory coverage**
   - instrument-panel green, seat 설정, nephew ambient preference 등
     정답에 필요한 record가 여전히 없거나 다른 domain record로 대체됐다.

Artifact:
`ubuntu/evaluation/vehiclemembench/99ecd902-d9a6-43af-9465-3cd42900f316`

v3 Memory cache:
`ubuntu/evaluation/vehiclemembench-memory/d745b359794da154/scenario-01/dc5d34beaa233f79`

평가 후 성공 task를 `retrieval_empty`로 분류하던 taxonomy 조건을
수정했으며, 전체 146개 테스트 중 145 passed, 공식 checkout 선택 테스트
1 skipped와 Ruff를 통과했다.

## VehicleMemBench 10% 비교 평가

실행일: 2026-07-28

공식 VehicleMemBench 50개 Scenario 중 앞 5개를 사용했다. 각 Scenario의
10개 task, 총 50/500 task이므로 전체 평가셋의 10%다. 두 방법 모두
Agent `gpt-5.6-terra`, Memory `gpt-5.6-luna`, embedding
`text-embedding-3-small`, Agent prompt `vehicle-agent-v2`로 실행했다.

| 방법 | ESM | State F1 | Value F1 | Tool F1 | Argument exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| Structured Hybrid | **0.56** | **0.873** | **0.760** | **0.627** | **0.50** |
| Tool-schema Patch (Ours) | 0.30 | 0.460 | 0.433 | 0.287 | 0.22 |

Ours는 정확한 최종 상태 task가 15/50, Hybrid는 28/50이다. 현재 Ours는
모든 주요 정확도 지표에서 Hybrid보다 낮으며, 아직 성능 우위 가설을
지지하지 않는다.

### Scenario별 ESM

| Scenario | Hybrid | Ours |
| --- | ---: | ---: |
| 1 | 0.70 | 0.30 |
| 2 | 0.50 | 0.10 |
| 3 | 0.50 | 0.50 |
| 4 | 0.70 | 0.30 |
| 5 | 0.40 | 0.30 |

### 병목과 효율

- Retrieval Recall@k: Hybrid 0.47, Ours 0.11
- task당 선택 Memory: Hybrid 5.00개, Ours 0.34개
- Ours 실패의 31/50은 `tool_selector_miss` 21건과
  `retrieval_empty` 10건이다.
- Coreference 6개 task의 ESM은 Hybrid 0.50, Ours 0.00이다.
- Ours의 active Memory는 Scenario당 평균 7.4개, Hybrid는 평균
  14.6개여서 extraction coverage도 약 절반이다.
- Agent input은 201,902 → 76,858 token(-61.9%), 평균 latency는
  5.11 → 3.62초(-29.1%), Memory context는 239.1 → 21.0 token(-91.2%)로
  줄었다.
- 다만 Ours의 Vehicle Tool 호출도 task당 1.36 → 0.68로 줄었다.
  필요한 Tool까지 누락한 결과가 포함되므로 현재 효율 수치를 순수한
  최적화 효과로 해석하지 않는다.
- 새 Memory 생성은 Ours 410회·입력 4,218,125 token, Hybrid Memory LLM
  68회·입력 519,446 token이다. Ours는 추론 입력을 줄였지만 offline
  Memory 생성량은 훨씬 크다.
- 가격 rate 환경변수가 없어 artifact의 USD 값은 0으로 기록됐으며,
  이는 무료 실행을 의미하지 않는다.

판단: 50개 Scenario 전체로 확대하기 전에 동일 5개 Scenario에서
extraction coverage, entity/coreference identity, semantic Tool routing,
retrieval fallback을 먼저 개선한다. 특히 selector가 확신하지 못할 때
관련 Tool·Memory 후보를 넓히는 recall 우선 fallback이 필요하다.

Artifacts:

- Hybrid:
  `ubuntu/evaluation/vehiclemembench/84ec01dd-34ea-49fe-9808-ceb6cdd070b6`
- Ours:
  `ubuntu/evaluation/vehiclemembench/403cbce2-5351-4255-9723-c63baf4098e7`

## P0 판단 규칙

| 결과 | 우선 개선 |
| --- | --- |
| Gold Tool + Gold Memory도 낮음 | P5 Agent·Context |
| Gold Tool + Patch Memory가 낮음 | P1 Batch Patch, P3 Identity, P4 Validation |
| Tool Selector + Gold Memory만 낮음 | P5 Tool discovery |
| Tool Selector + Patch가 Oracle Patch보다 낮음 | P2 Routing·Retrieval |
| 개별 조건은 높고 E2E만 낮음 | 단계 연결·Context 조립 |

## 개선 후보

### P1 — Batch Patch와 비용

- 20~50 turns 또는 2~4K tokens 단위 처리
- source turn 순서가 있는 Patch 목록 생성·적용
- Tool ontology 압축·prompt cache·관련 record 제한
- Validation 재시도 피드백과 반복 실패 NO-OP 처리

### P2 — Routing·Retrieval

- Keyword를 hard filter가 아닌 점수로 사용
- Tool-description embedding과 선택적 LLM reranker
- 넓은 Tool 후보에서 BM25·embedding·condition 점수 결합

### P3 — Identity·Storage

- canonical `entity_id`와 alias
- 고정 identity와 시간·날씨·역할 등 적용 조건 분리
- 조건 호환, conflict, version 관리

### P4 — Validation

- 구조·권한·타입·transaction 검사는 엄격하게 유지
- alias, evidence 표현, 의미적 값 비교는 유연하게 처리
- 낮은 확신은 reject 대신 review 후보로 보존

### P5 — Agent·Context

- 선택 Memory와 Tool schema의 argument 변환 검증
- source turn·evidence는 trace로 유지하고 필요할 때만 검증
- Tool discovery 오류를 Memory 오류와 분리 측정

### P6 — 확대 평가·최적화

- 동일 Scenario에서 개선 전후 재평가
- 선택적 batch pre-filter와 ablation
- Scenario 1 통과 후 5 scenarios, 이후 50 scenarios

## 수행 원칙

- Gold 정보는 진단용 Oracle로만 사용하고 최종 성능에 포함하지 않는다.
- P0 결과 없이 P1~P6을 일괄 적용하지 않는다.
- Memory를 다시 생성하는 변경 전에는 예상 호출 수와 비용을 확인한다.
- 각 변경 후 동일 모델·Scenario·평가 설정으로 P0와 E2E를 재실행한다.
