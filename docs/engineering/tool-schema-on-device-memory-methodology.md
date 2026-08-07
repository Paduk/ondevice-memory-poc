# Fact-First Incremental Memory Patching with Late Tool Binding

상태: `Cloud LLM PoC R5 50-task 평가 완료`

최종 검토: 2026-07-29

## 1. 가설

Mobile Agentic Framework에서 대화의 지속적인 사실·선호·제약·상태를 작은
memory record로 폭넓게 추출하고, 기존 record를 제한된 patch로 갱신하면
전체 대화나 거대한 memory document를 다시 제공하지 않고도 정확한 Tool
Calling이 가능하다.

Memory 생성 단계에서는 특정 Tool의 전체 schema와 실행 argument를 강제하지
않는다. 대신 추론 시 현재 요청과 관련된 record를 먼저 검색하고, 그때 Tool
schema에 연결해 실행 가능한 argument로 변환한다.

검증할 효과는 다음과 같다.

- ADD·UPDATE·MERGE·DELETE 기반의 일관된 장기 Memory 유지
- Memory 누락을 줄이면서 Tool 선택과 argument 정확도 유지 또는 향상
- 추론 시 top-k Memory와 선택 Tool만 제공해 context와 latency 제한
- 영속적인 사용자 선호와 요청마다 달라지는 실행 argument의 분리

Memory 생성은 대화 종료 후 유휴 시간에 비동기로 실행한다. 현재 PoC는
OpenAI Cloud LLM을 사용하고, 최종 단계에서는 같은 protocol의 On-device
SLM으로 교체한다.

## 2. 핵심 구조

```text
대화 Batch
  → 고재현율 Fact 후보 추출
  → Evidence·권한·구조 Hard Gate
  → 값 정규화와 deterministic Validation
  → 불확실 후보만 Batch Semantic Validation
  → 기존 Memory와 identity·의미 비교
  → ADD·UPDATE·MERGE·DELETE
  → SQLite versioned record 저장

사용자 요청
  → 인물·역할·시간·상황 해석
  → Memory-first top-k 검색
  → Tool schema 후보와 결합·rerank
  → runtime argument binding
  → Agent Tool Calling
```

Tool schema는 Memory 후보를 제한하는 hard boundary가 아니라 다음 용도로
사용한다.

- 저장된 predicate에 선택적인 capability hint 제공
- 추론 시 관련 Tool 후보 생성
- 최종 Tool name과 argument schema 검증
- Tool 실행 trace와 평가 연결

## 3. Fact Memory record

하나의 통합 문서를 재작성하지 않고 작은 record를 SQLite에 저장한다.

```json
{
  "record_key": "vehicle_scenario_1|justin|seat_ventilation_speed|hot",
  "user_id": "vehicle_scenario_1",
  "entity_id": "justin",
  "predicate": "seat_ventilation_speed",
  "value": 2,
  "identity_conditions": {"person": "justin"},
  "applicability": {"weather": "hot"},
  "capability_hints": ["seat", "ventilation"],
  "memory_type": "preference",
  "status": "active",
  "confidence": 0.96,
  "version": 2,
  "evidence_message_ids": [142],
  "supersedes_id": "previous-record-id"
}
```

외부 구조는 엄격하게 유지하지만 predicate와 value 표현은 유연하게
정규화한다.

- `user_id`: Memory partition 소유자
- `entity_id`: 선호·사실의 주체
- `predicate`: 안정적인 사실 종류
- `value`: 영속적으로 기억할 값
- `identity_conditions`: 같은 record 계열을 찾는 핵심 대상 조건
- `applicability`: 시간·날씨·장소·상황 등 적용 조건
- `capability_hints`: 선택적인 Tool domain 후보이며 정답 Tool 보장은 아님
- `record_key`: canonical identity를 이용한 version 계열 식별자

`seat=driver`, `zone=passenger`처럼 현재 요청에서 결정되는 값은 기본적으로
Memory에 고정하지 않는다. 예를 들어 `Patricia의 headrest=44`를 저장하고,
Patricia가 현재 운전자인지는 추론 시 binding한다.

## 4. Patch와 Validation

Memory 모델은 DB를 직접 수정하지 않고 다음 patch만 제안한다.

- `ADD`: 대응하는 active record가 없을 때 추가
- `UPDATE`: 동일 record의 값·적용 조건이 변경됐을 때 새 version 생성
- `MERGE`: 의미가 같은 중복 record 통합
- `DELETE`: 명시적으로 철회된 record를 tombstone 처리

Patch linker는 canonical entity, predicate, identity condition과 의미적 값
유사도를 사용해 기존 record를 찾는다. exact 문자열 일치만으로 identity를
결정하지 않는다.

Validation은 세 층으로 분리한다.

1. **항상 엄격함**
   - JSON 구조, user·scope 권한, target 존재, transaction
   - exact evidence quote 위치, PII·secret 정책, hard-delete 금지
2. **deterministic 정규화**
   - `three ↔ 3`, `turn on ↔ true` 등 값 정규화
   - UPDATE·철회 표현의 동의어
   - entity alias와 runtime-bound argument
3. **선택적 Semantic Validation**
   - deterministic 결과가 `review`인 후보만 동일 batch에서 한 번 검증
   - 기존 record와 evidence를 비교해 `ACCEPT·REVIEW·REJECT` 판정
   - 구조·권한·근거·PII hard rejection은 LLM에 보내거나 번복하지 않음

Semantic Validation은 Fact 추출과 동일한 최대 32 turns 또는 4,096 source
token 주기로 대화 종료·idle worker에서 수행한다. 후보마다 호출하지 않고
불확실 후보 전체를 한 번에 검증한다. 검증 호출이 실패하거나 confidence가
낮으면 후보는 `review`로 보존하며 foreground Agent 요청을 차단하지 않는다.
다음 Agent 요청에는 commit이 끝난 Memory snapshot만 사용하므로 semantic
검증 호출이 추론 latency에 포함되지 않는다.

## 5. Retrieval과 Late Tool Binding

질문만으로 Tool을 먼저 확정해 Memory 후보를 잘라내지 않는다.

1. 요청에서 사용자·인물·driver/passenger 역할·시간·날씨·상황을 해석한다.
2. 해당 entity와 조건에 맞는 Memory를 BM25·embedding으로 넓게 검색한다.
3. query→Tool schema 결과와 query→Memory 결과의 합집합을 만든다.
4. 조건 점수와 semantic score로 top-k Memory와 Tool을 함께 rerank한다.
5. route가 모호하면 top-k 후보를 Cloud Agent가 최종 rerank한다. 별도
   classifier는 추가 호출의 이득이 확인될 때만 선택적으로 사용한다.
6. 선택된 Memory의 영속 값과 현재 요청의 역할을 결합해 Tool argument를
   생성한다.
7. JSON schema와 simulator semantic rule을 모두 통과한 후보만 Agent에게
   제공한다.

Agent에게는 시스템 지침, 현재 질문, top-k Memory, 선택 Tool schema와 실행
결과만 제공한다. 전체 history와 비선택 Memory는 기본 context에 포함하지
않는다.

## 6. PoC와 최종 목표

| 단계 | Memory 모델 | 검증 범위 |
| --- | --- | --- |
| 현재 PoC | OpenAI Cloud LLM | Fact 추출, Patch, retrieval, Tool 성능 |
| 최종 단계 | On-device SLM | 동일 기능, 기기 자원, Cloud 노출 감소 |

두 단계는 동일한 `FactMemoryModel`과 `PatchMemoryModel` 입출력 계약을
사용한다. Cloud PoC에서는 redaction된 history가 Cloud로 전송되므로
On-device privacy 효과를 주장하지 않는다.

## 7. VehicleMemBench 검증

VehicleMemBench history를 시간순 replay하고 평가 query 전에 Memory snapshot을
고정한다. 평가 query와 Tool 결과는 Memory 생성 DB에 다시 저장하지 않는다.

주요 지표:

- Exact State Match, State/Value F1
- Tool F1, argument exact match
- Fact candidate recall·acceptance와 operation별 정확도
- Retrieval Recall@k, top-k 수, context token과 latency
- entity·condition·Tool routing 실패율
- Memory 생성과 Agent 추론 각각의 Cloud token·전송량

Gold Tool과 Gold Memory는 원인 분리용 Oracle로만 사용한다. 최종 비교에는
정답 정보를 노출하지 않는다.

## 8. 경계

- VehicleMemBench 정답을 record schema나 routing rule에 사용하지 않는다.
- Memory를 많이 저장하는 것 자체를 성과로 주장하지 않는다.
- 구조·권한·근거 검증을 없애 정확도를 높이지 않는다.
- Background Memory 실패가 foreground Tool Runtime을 차단하지 않는다.
- Cloud PoC 결과로 On-device 전력·발열·privacy 효과를 주장하지 않는다.
