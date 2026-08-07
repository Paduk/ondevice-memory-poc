# Recursive Summarization Baseline 방법론 및 구현 계획

상태: `구현 완료 · Scenario 6 smoke 및 Scenario 6–10 holdout 완료`

작성일: 2026-07-30

## 1. 목적

PalmClaw의 VehicleMemBench 평가에 공식 Recursive Summarization 계열
baseline을 추가한다. 최종 비교 대상은 다음 세 profile이다.

- `cloud_structured_hybrid`: 구조화 Memory를 BM25와 embedding으로 검색
- `cloud_recursive_summary`: 날짜별로 갱신한 최종 전체 요약을 사용
- `cloud_fact_patch`: PalmClaw의 Fact-first 방식

첫 평가는 기존 Holdout Scenario 6–10의 50 task에서 수행한다. 구현과 cache
재현성이 확인된 뒤 전체 50 scenario·500 task로 확대한다.

PalmClaw에는 이미 `cloud_summary`가 있지만 공식 VehicleMemBench의
Recursive Summarization과 prompt·batch·update 계약이 다르다. 기존 결과의
의미와 cache 호환성을 보존하기 위해 `cloud_summary`를 변경하지 않고
`cloud_recursive_summary`를 새 profile로 추가한다.

### 구현 결과

2026-07-30 기준으로 다음 항목을 구현했다.

- `memory_update(new_memory)` 전용 OpenAI provider 계약
- 날짜별 정렬과 oversized-day 순차 chunk
- update/no-op 구분, 8,192-character 제한, bounded retry
- 별도 `recursive_summary` session·cache·version·daily trace
- `cloud_recursive_summary` profile, CLI, runner 연결
- query-independent 전체 요약 주입과 immutable snapshot 검증
- Recursive Summary의 `Recall@k=N/A` 리포트 처리

Provider·Memory·Runner fixture를 포함한 전체 Ubuntu 테스트
`198 passed, 1 skipped`와 Ruff 검사를 통과했다. 실제 Cloud Memory LLM을
사용하는 Scenario 6 smoke와 Holdout 50-task 평가도 완료했다.

### 실측 평가 결과

평가일은 2026-07-30이며 Agent `gpt-5.6-terra`, Memory
`gpt-5.6-luna`, embedding `text-embedding-3-small`을 사용했다.

- [Recursive Summary Scenario 6–10 결과](../../ubuntu/evaluation/vehiclemembench/29518f79-654d-4da0-865a-17ca633b6e70/results.md)
- [기존 Structured Hybrid·Fact-first 동일 구간 결과](../../ubuntu/evaluation/vehiclemembench/ffdd8096-c990-46f6-ab9f-0644f61df56a/results.md)

| 방법 | ESM | State F1 | Tool F1 | Argument exact | Agent input |
| --- | ---: | ---: | ---: | ---: | ---: |
| Recursive Summary | **0.62** | **0.772** | **0.653** | **0.56** | 187,000 |
| Structured Hybrid | 0.56 | 0.751 | 0.607 | 0.48 | 182,311 |
| Fact-first | 0.50 | 0.661 | 0.507 | 0.42 | 170,545 |

Recursive Summary는 이 구간에서 Structured Hybrid보다 ESM `+0.06`,
Fact-first보다 `+0.12` 높았다. 다만 시나리오별 ESM은 다음과 같으며
Scenario 8의 큰 향상이 전체 차이를 주도했다.

| Scenario | Recursive Summary | Structured Hybrid | Fact-first |
| --- | ---: | ---: | ---: |
| 6 | 0.70 | 0.70 | 0.50 |
| 7 | 0.50 | 0.70 | 0.30 |
| 8 | **1.00** | 0.40 | 0.80 |
| 9 | 0.40 | 0.50 | 0.30 |
| 10 | 0.50 | 0.50 | 0.60 |

동일 task의 ESM을 직접 비교하면 Recursive Summary는 Structured Hybrid
대비 12승 29무 9패, Fact-first 대비 12승 32무 6패였다.

5개 snapshot은 총 233개 날짜 batch를 처리했고 update 65회, no-op
168회였다. 최종 요약은 평균 1,334 characters·295.4 tokens였고 truncation과
Memory 생성 실패는 모두 0회였다. 50개 task 완료율은 100%이며 residual
sensitive span도 0개로 artifact privacy audit를 통과했다.

## 2. 방법론

Recursive Summarization은 과거 대화를 개별 record로 저장하고 질의별로
검색하는 방식이 아니다. 시간순 History를 읽으면서 하나의 누적 차량 선호
요약을 반복해서 다시 작성한다.

```text
Day 1 History + Empty Memory
  → Summary 1

Day 2 History + Summary 1
  → Summary 2

Day 3 History + Summary 2
  → Summary 3

...

Final Summary
  → Scenario의 10개 task에 동일하게 전체 주입
  → Agent가 Tool을 발견·호출
  → VehicleWorld 최종 상태 평가
```

날짜 `d`의 대화를 `H_d`, 이전까지의 누적 Memory를 `M_(d-1)`이라고 하면
갱신은 다음과 같다.

```text
M_d = Summarize(M_(d-1), H_d)
```

당일 대화에 새롭거나 변경된 차량 관련 정보가 없으면 갱신하지 않고
`M_d = M_(d-1)`로 유지한다.

### 저장 대상

- 사용자별 차량 장치 설정과 선호
- 시간·날씨·장소·활동 등에 따른 조건부 선호
- 사용자 간 선호 충돌을 구분할 수 있는 사용자 identity
- 기존 선호의 명시적 수정·취소·변경
- 차량 동작에 직접 영향을 주는 자주 방문하는 위치
- 좌석·마사지 등 차량 설정에 직접 영향을 주는 신체 조건

### 저장하지 않는 대상

- 차량 동작과 무관한 일반 일상, 취미, 업무, 관계
- 일회성 명령을 영속 선호로 일반화한 정보
- 대화에 명시되지 않은 추론값
- Tool schema와 Gold Memory·Tool·argument

### Memory update 계약

Memory LLM에는 `memory_update(new_memory)` 하나만 제공한다.

- 새 정보나 변경이 있으면 `memory_update`를 호출한다.
- `new_memory`는 delta가 아니라 기존 내용까지 포함한 완전한 최신 요약이다.
- 새 정보가 없으면 Tool을 호출하지 않고 이전 Memory를 유지한다.
- 요약은 사용자별 Markdown bullet로 구성한다.
- 최종 Memory는 최대 8,192 characters로 제한한다.

### 질의 시 동작

최종 요약은 질의와 무관하게 항상 전체가 선택된다.

- BM25·embedding·Top-k 검색을 사용하지 않는다.
- 질의별 Memory reranking을 사용하지 않는다.
- 원본 History를 Agent에게 제공하지 않는다.
- 평가 중 query와 Tool 결과로 Memory를 갱신하지 않는다.
- 같은 scenario의 10개 task가 동일한 읽기 전용 최종 요약을 공유한다.

따라서 이 방법의 강점과 약점은 모두 요약 자체에 있다. 필요한 정보가 최종
요약에 남아 있으면 검색 누락이 없지만, 중간 갱신에서 삭제되거나 왜곡된
정보는 질의 시 복구할 방법이 없다.

## 3. 기존 `cloud_summary`와의 차이

현재 PalmClaw `cloud_summary`도 `previous_memory + new batch`를 반복해 최신
요약을 만들고 전체 요약을 반환한다. 재사용할 기반은 이미 존재한다.

| 항목 | 현재 `cloud_summary` | 새 `cloud_recursive_summary` |
| --- | --- | --- |
| Prompt | 범용 durable memory | 차량 선호 전용 공식 prompt |
| 입력 단위 | token budget에 따른 batch | 달력 날짜별 History |
| 날짜 경계 | 한 batch에 여러 날짜 가능 | 날짜별 순차 갱신 |
| 갱신 판단 | 빈 text 응답 | `memory_update` 호출 또는 명시적 no-op |
| 출력 형식 | 간결한 Markdown | 사용자별 차량 선호 bullet |
| 최대 크기 | provider 출력 제한 중심 | 8,192-character hard limit |
| 검색 | 최신 전체 요약 | 최신 전체 요약 |

공식 baseline의 성능을 주장하려면 단순히 기존 `cloud_summary`의 이름을
바꾸면 안 된다. 위 차이를 구현하고 prompt·cache version을 별도로 기록해야
한다.

## 4. 구현 원칙

### 4.1 기존 공통 실행 경로 재사용

새 profile도 기존 VehicleMemBench runner, VehicleWorld, Tool discovery,
scorer를 그대로 사용한다. 새로 구현할 범위는 Memory 생성과 최종 Memory
해결 단계뿐이다.

```text
공통: Dataset → Task → Agent → Tool loop → VehicleWorld → Scorer
변경: History → Recursive Summary → retrieved_memory
```

### 4.2 기존 profile과 cache 보존

- `cloud_summary`의 의미와 결과는 변경하지 않는다.
- 새 내부 strategy 이름은 `recursive_summary`로 한다.
- 새 외부 profile 이름은 `cloud_recursive_summary`로 한다.
- prompt version은 `vehicle-recursive-summary-v1`로 시작한다.
- schema version은 `recursive-summary-v1`로 시작한다.
- 기존 `summary` cache를 재사용하지 않는다.

### 4.3 실패를 조용히 숨기지 않기

공식 reference code는 일부 LLM·JSON 오류에서 이전 Memory를 그대로
사용한다. PalmClaw에서는 bounded retry 이후에도 실패하면 해당 Memory
build를 실패 처리한다. 모델 오류를 정상적인 no-op으로 오인하지 않기
위해서다.

## 5. 세부 구현 계획

### 5.1 Vehicle 전용 Recursive Summary provider

대상:

- `ubuntu/src/palmclaw_ubuntu/providers.py`
- `ubuntu/src/palmclaw_ubuntu/contracts.py`

작업:

1. 차량 선호 전용 prompt와 `memory_update` Tool schema를 추가한다.
2. 입력은 `previous_memory`, `date`, `daily_history`로 구성한다.
3. Tool choice는 `auto`로 두고 다음 결과를 구분한다.
   - `updated`: 유효한 `memory_update(new_memory)` 호출
   - `noop`: Tool 호출 없음
   - `failed`: API·schema·크기 검증 실패
4. update 결과의 앞뒤 공백과 형식을 정규화한다.
5. 8,192 characters를 초과하면 마지막 완전한 bullet 경계에서
   deterministic하게 자르고 truncation metadata를 기록한다.
6. 모델 ID, prompt version, schema version, usage, latency, response ID,
   update/no-op 상태를 저장한다.

기존 `MemoryResponse`의 `content`는 다음처럼 사용한다.

- update: 완전한 최신 요약
- no-op: 빈 문자열과 `metadata.update_status=noop`
- failure: 예외

### 5.2 날짜별 History batch

대상:

- `ubuntu/src/palmclaw_ubuntu/vehicle_bench/memory.py`

작업:

1. `VehicleHistoryEntry.date`를 기준으로 날짜별 group을 만든다.
2. 날짜는 오름차순으로 처리한다.
3. 같은 날짜 안의 turn 순서는 원본 timestamp 순서를 유지한다.
4. 한 날짜가 모델 입력 한도를 초과할 때만 그 날짜 내부를 순차 chunk로
   나눈다. 이 경우 같은 날짜의 chunk끼리도 recursive update를 적용하고
   manifest에 deviation을 기록한다.
5. 일반 `--memory-batch-tokens`와 혼동하지 않도록 날짜별 group 정보와
   실제 token 수를 cache manifest에 저장한다.

### 5.3 Summary version 저장과 cache

기존 `MemoryEngine`과 `SQLiteRepository`의 summary version 저장 경로를
재사용한다.

작업:

1. summary 계열 판정을 `summary`와 `recursive_summary`가 공유하도록
   일반화한다.
2. update가 있으면 이전 version을 supersede하고 완전한 새 version을
   저장한다.
3. no-op이면 consolidation watermark만 진행시키고 새 version은 만들지
   않는다.
4. 날짜별 snapshot과 update/no-op log를 trace에서 검사할 수 있게 한다.
5. cache key에 다음을 포함한다.
   - dataset와 History SHA-256
   - 날짜 grouping policy
   - Memory model ID
   - prompt와 schema version
   - maximum output tokens
   - 8,192-character limit
   - reasoning effort와 PII policy
6. cache가 `ready`가 되기 전에는 task 평가를 시작하지 않는다.

### 5.4 Snapshot resolution

대상:

- `ubuntu/src/palmclaw_ubuntu/vehicle_bench/memory.py`

`VehicleMemorySnapshot.resolve()`에 `cloud_recursive_summary` 분기를
추가한다.

반환 계약:

```text
content: 최종 전체 요약
memory_strategy: recursive_summary
retrieval_mode: full_recursive_summary
retrieval_selected_count: 요약이 있으면 1, 없으면 0
query_dependent: false
```

trace에는 final summary hash, version 수, update/no-op 수, 최종
character/token 수, truncation 여부를 포함한다. resolve 전후 Memory
fingerprint가 같아야 한다.

### 5.5 Profile·CLI·runner 연결

대상:

- `ubuntu/src/palmclaw_ubuntu/vehicle_bench/memory.py`
- `ubuntu/src/palmclaw_ubuntu/vehicle_bench/runner.py`
- `ubuntu/src/palmclaw_ubuntu/vehicle_bench/__init__.py`
- `ubuntu/src/palmclaw_ubuntu/cli.py`

작업:

1. `VEHICLE_MEMORY_PROFILES`와 `VEHICLE_AGENT_PROFILES`에
   `cloud_recursive_summary`를 등록한다.
2. `required_memory_strategies()`가 `recursive_summary`를 요청하게 한다.
3. builder가 별도 session과 cache profile을 생성하게 한다.
4. 최종 summary를 기존 `retrieved_memory` 위치에 주입한다.
5. Tool selector, execution hint, embedding retrieval은 적용하지 않는다.
6. CLI help와 run manifest에 새 profile을 표시한다.

## 6. 공정 비교 조건

세 profile에서 다음 조건은 같아야 한다.

- Scenario와 task 순서
- Agent `gpt-5.6-terra`
- Memory LLM `gpt-5.6-luna`
- 최대 Tool round와 VehicleWorld 초기 상태
- Tool discovery 방식
- scorer와 metric 계산
- scenario별 Memory 1회 생성 및 10개 task 공유
- task query와 Tool 결과의 Memory 재저장 금지
- Gold Memory·Tool·argument 비공개

방법 고유 차이로 허용하는 것은 Memory representation과 retrieval뿐이다.

- Hybrid: 구조화 record + BM25/embedding Top-k
- Recursive: 날짜별 누적 전체 요약
- Ours: versioned Fact + BM25/embedding + late binding

Recursive Summary에는 Top-k가 없으므로 `Recall@k`를 `0`으로 기록하면 안
된다. 결과표에는 `N/A`로 두고, 대신 최종 summary의 크기와 post-hoc
필요정보 포함 여부를 진단한다.

## 7. 테스트 계획

### Unit test

1. 날짜별 grouping과 시간순 정렬
2. 첫 차량 선호의 `memory_update`
3. 차량 정보가 없는 날짜의 no-op
4. 기존 선호 수정 시 이전 값 제거와 최신 값 유지
5. 서로 다른 사용자의 충돌 선호 분리
6. 조건부 선호 보존
7. 8,192-character 제한과 bullet-boundary truncation
8. malformed Tool argument retry와 최종 실패

### Storage·cache test

1. update 날짜에만 새 summary version 생성
2. no-op 이후에도 consolidation watermark 진행
3. 중단 후 같은 cache에서 resume
4. prompt/model/grouping 설정 변경 시 cache 불일치 탐지
5. final summary와 version trace hash 일치

### Runner integration test

1. `cloud_recursive_summary` 단독 1 scenario fixture 실행
2. 세 profile 동시 실행 시 task 수와 순서 일치
3. 10개 task가 동일한 summary hash 사용
4. query 실행 전후 Memory fingerprint 불변
5. Agent context에 raw History와 Gold Memory가 없음
6. embedding call과 Tool-aware hint가 생성되지 않음

## 8. 평가 단계

### 단계 A: Fixture와 1-scenario smoke

- deterministic fixture Memory model로 update/no-op 경로를 검증한다.
- 실제 Memory LLM으로 Scenario 6의 1 task를 실행한다.
- cache, trace, summary formatting, Tool loop를 수동 확인한다.

### 단계 B: Holdout Scenario 6–10

```bash
cd ubuntu

export VEHICLEMEMBENCH_ROOT=/home/hj153lee/VehicleMemBench
export PALMCLAW_MEMORY_MAX_OUTPUT_TOKENS=2048
export PALMCLAW_MODEL_TIMEOUT_SECONDS=120

palmclaw eval vehicle \
  --benchmark-root "${VEHICLEMEMBENCH_ROOT}" \
  --mode live \
  --profiles cloud_structured_hybrid,cloud_recursive_summary,cloud_fact_patch \
  --scenario 6 \
  --scenario-limit 5 \
  --task-limit 10 \
  --model gpt-5.6-terra \
  --memory-model gpt-5.6-luna \
  --embedding-model text-embedding-3-small \
  --memory-cache-dir ./evaluation/vehiclemembench-memory
```

보고 지표:

- ESM
- State F1
- Tool F1
- Argument exact
- Agent input/output tokens와 latency
- Memory build input/output tokens, latency, update/no-op 수
- 최종 summary characters/tokens와 truncation 수
- reasoning type별 ESM

### 단계 C: 진단

ESM 실패 task를 다음 단계로 분류한다.

1. 필요한 정보가 당일 History에서 추출되지 않음
2. 중간에는 있었지만 이후 recursive rewrite에서 삭제됨
3. 최종 summary에 있으나 Agent가 사용하지 못함
4. 올바른 정보를 잘못된 Tool·argument로 변환함
5. 복합 Tool 호출을 누락하거나 과잉 실행함

Ours의 reviewed Fact annotation은 post-hoc 진단에만 사용하며 Agent 입력에는
사용하지 않는다.

### 단계 D: 전체 500-task

Holdout에서 cache 재현, 완료율, trace가 모두 정상이고 profile 구현을 더
수정할 필요가 없을 때 prompt와 설정을 동결한다. 이후 전체 50
scenario·500 task를 실행한다.

## 9. 완료 기준

구현 완료는 성능 우위가 아니라 다음 조건으로 판단한다.

- 공식 방법의 날짜별 recursive update와 no-op 계약 구현
- 50-task 평가 완료율 100%
- 모든 scenario에서 읽기 전용 snapshot 검증 통과
- Gold 정보와 raw History의 Agent 유출 0건
- 세 profile이 동일 Agent·task·scorer에서 실행됨
- cache 삭제 없이 동일 설정 resume 가능
- 결과 artifact에 prompt/model/cache/summary trace 기록
- `Recall@k=N/A`와 full-summary 비용이 정확히 보고됨

## 10. 예상 한계

- 초기 요약 오류가 이후 모든 날짜에 전파될 수 있다.
- 완전한 요약을 반복 작성하면서 오래된 Fact가 사라질 수 있다.
- query-aware retrieval이 없어 관련 없는 선호도 Agent context에 포함된다.
- 사용자별·조건별 표현이 압축되며 정확한 enum과 수치가 손실될 수 있다.
- 요약 크기 제한에 도달하면 오래된 정보가 잘릴 수 있다.
- VehicleMemBench 원 논문의 수치는 다른 backbone 결과이므로 PalmClaw
  결과와 직접 동일하다고 기대하면 안 된다.

이 한계는 Structured Hybrid와 Fact-first가 개별 record 검색을 사용하는
이유를 보여주는 비교 지점이기도 하다.

## 11. 참고 구현

- VehicleMemBench 공식 Recursive Summarization:
  `../../../VehicleMemBench/evaluation/model_evaluation.py`
  - `split_history_by_day`
  - `summarize_day_with_previous_memory`
  - `build_memory_recursive_summary`
  - `process_task_with_memory`
- PalmClaw 기존 summary engine:
  `../../ubuntu/src/palmclaw_ubuntu/memory.py`
- PalmClaw VehicleMemBench snapshot/cache:
  `../../ubuntu/src/palmclaw_ubuntu/vehicle_bench/memory.py`
- PalmClaw VehicleMemBench runner:
  `../../ubuntu/src/palmclaw_ubuntu/vehicle_bench/runner.py`

공식 VehicleMemBench checkout은 commit
`5ef3c48a4dbb446e6bb84a91dcc3632e9b1d203b`를 기준으로 한다.
