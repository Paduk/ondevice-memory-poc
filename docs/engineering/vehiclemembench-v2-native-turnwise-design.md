# VehicleMemBench V2 Native Turn-wise 설계

## 목적

`native_turnwise`는 새 메모리 방법론이 아니라 **데이터 생성 순서**다. V1처럼 event의
대화 전체를 먼저 만들지 않고, 대화 한 turn을 생성한 직후 그 시점의 memory label을
확정한다.

```text
Stage 2 persona/event 고정
  -> turn t 생성
  -> M(t-1) + turn t로 NO_OP/UPDATE 확정
  -> Patch를 결정론적으로 적용해 M(t) 저장
  -> 필요하면 현재 시점 Turn Quiz 생성
  -> turn t+1
```

최종적으로 동일 scenario에서 다음 두 출력을 함께 만든다.

- V1-compatible: 전체 history + delayed final Quiz 10개
- V2-native: 모든 turn의 memory label/snapshot + 중간 Turn Quiz

`Turn-wise Patch/Summary` 평가 방법론과 혼동하지 않는다. Native는 benchmark 생성
경로이고, Patch/Summary는 생성된 turn을 처리하는 평가 대상이다.

## 재사용 범위

| 기존 자산 | Native에서의 사용 |
| --- | --- |
| `v1_generation.py` | persona, event chain, interleaved timeline을 그대로 사용 |
| `v1_stage3.py` | 논문 기반 dialogue 규칙, turn 수(배경 40/차량 26), materialize/serializer, final Quiz와 simulator 검증 재사용 |
| `v2_hybrid.py` | structured update, memory entry, Patch 생성·적용, evidence/hash-chain 검증 재사용 |
| `v2_turn_quiz.py`·`v2_turn_quiz_expansion.py` | Hybrid/post-hoc와 같은 immediate·delayed·composite Turn Quiz 30개, gold Tool call 파생, answerability/simulator 검증 재사용 |
| consensus Gold 코드 | 최종 production label 정제에만 사용; 3-way 경로 비교에는 넣지 않음 |

Hybrid와 Native의 차이는 dialogue 생성 cadence뿐이다. 공정한 pilot에서는 memory
표현·UPDATE 기준·Quiz 생성 규칙을 동일하게 둔다.

## Turn 처리

### 1. 한 turn 생성

기존 `V1_DIALOGUE_INSTRUCTIONS`를 유지하되 출력만 `turns[]` 일괄 생성에서 정확히 한
개의 `{speaker_id, text}`로 바꾼다. 입력은 다음으로 제한한다.

- 현재 persona와 structured event
- 같은 chain의 과거 event와 최근 global event
- 현재 event에서 이미 생성된 **과거 turn만**
- 현재 turn index, 남은 turn 수와 speaker별 누적 turn 수

미래 dialogue, final Quiz, target state, memory label은 주지 않는다. 승인 memory도
대화 생성기에 주지 않는다. 사람이 말할 내용이 메모리 출력에 의해 역으로 오염되는
self-confirming loop를 막기 위해서다.

긴 prefix의 반복 입력은 event 내부 최근 turn window와 결정론적 progress state로
제한한다. 단, artifact에는 전체 causal prefix hash를 남긴다.

### 2. 현재 turn의 label 확정

배경 event는 모두 결정론적 `NO_OP`이다. 차량 event는 아직 반영되지 않은 structured
preference update와 **현재까지의 causal prefix**를 대조한다.

- 현재 turn에서 update가 충분히 드러나지 않음: `NO_OP`
- subject/value/setting/condition이 prefix에서 충분히 확인됨: `UPDATE`
- 이미 반영된 사실의 반복: `NO_OP/DUPLICATE_ALREADY_STORED`

LLM은 evidence boundary와 exact quote만 정렬한다. 실제 memory 문장과
`add/replace/delete`는 structured event에서 결정론적으로 만들고 atomic apply한다.
기존 Hybrid의 future-prefix, exact-quote, memory hash-chain 검증을 그대로 통과해야 한다.

학습용 입력은 기존 V2와 동일한 `M(t-1) + current turn` view로 별도 export한다. causal
prefix가 없으면 UPDATE 의미를 복원할 수 없는 label은 `context_dependent=true`로
표시해, 단일-turn 학습에서 제외하거나 bounded recent-context view로만 사용한다.

### 3. Quiz 생성

Native 전용 Quiz 규칙을 새로 만들지 않는다. Hybrid/post-hoc에서 확정한 공통 경로로
시나리오당 Turn Quiz 30개와 V1-compatible final Quiz 10개를 만든다.

- immediate: 모든 주요 `UPDATE` 직후
- delayed: 같은 memory fact를 해당 vehicle chain 종료 시점에 재질문
- composite: 두 개의 호환되는 memory fact를 한 요청에서 함께 적용
- final: 기존 V1 Stage 3의 delayed final Quiz 10개를 그대로 재사용

모든 `NO_OP` turn에 Quiz를 만들지는 않는다. 모든 Turn Quiz에는 다음 제약을 적용한다.

- 입력: `M(t)`와 `history[0:t]`
- 정답: 현재까지 드러난 structured update에서만 결정론적으로 파생
- 금지: 미래 event, final V1 Quiz, 최종 memory

## Artifact와 resume

event별이 아니라 turn별 checkpoint를 atomic하게 저장한다.

```text
native-turnwise/<scenario>/
  manifest.json
  events/<timeline>-<event>/
    progress.json
    turns/000.json ...
  turn-quizzes/*.json
  native.json
  v1-public/history.json
  v1-public/qa.json
```

각 turn checkpoint에는 생성 turn, model/prompt/usage, before/after memory hash,
NO_OP/UPDATE와 reason code, exact evidence, Patch, source update index를 저장한다. 재개 시
마지막 승인 hash부터 이어가며 이미 저장된 turn은 재생성하지 않는다.

## 구현 최소 범위

1. `v2_native_turnwise.py`: one-turn schema/model, 순차 orchestrator, checkpoint/audit
2. 기존 Hybrid의 memory operation 생성기를 공용 helper로 노출
3. `run_vehiclemembench_v2_native_turnwise_smoke.py`: 1 event/1 scenario resume runner
4. 단위 테스트: 미래 입력 금지, exact turn 수, NO_OP/UPDATE, atomic Patch, hash-chain,
   resume 동일성, V1 serializer 호환
5. 1개 canary 통과 후 고정 Stage 2 artifact 5개로 post-hoc/Native/Hybrid paired pilot

초기 구현에서는 기존 Hybrid schema를 가능한 한 재사용하고 대규모 schema refactor는
하지 않는다.

### 현재 구현 상태

- **1회차 완료:** `v2_native_turnwise.py`의 strict one-turn schema, Terra provider,
  bounded causal input, provider 결과 검증과 V1 event-dialogue 조립 adapter
- **테스트 완료:** causal window, 금지 입력 부재, participant/agent 검증, prefix hash,
  기존 V1 dialogue schema 호환
- **2회차 완료:** turn별 NO_OP/UPDATE 확정, Hybrid 공용 Patch helper를 통한 atomic
  memory 적용, committed source index와 dialogue/memory/checkpoint hash-chain, atomic
  JSON checkpoint 저장
- **3회차 완료:** unresolved update만 보는 Terra current-turn alignment provider,
  background/반영 완료 turn의 deterministic NO_OP, event 단위 resume/retry와 완결성
  검사, 1-event Cloud canary runner
- **4회차 완료:** Native artifact의 공통 Quiz checkpoint contract 연결, 80-event
  scenario orchestrator, event/turn resume, event 간 memory·checkpoint hash-chain,
  global turn 연속성, 누적 token/latency audit와 `progress.json`/`native.json` 저장
- **5회차 완료:** 공통 Turn Quiz 실행기의 Native source·embedded dialogue 지원,
  immediate/delayed/composite 총 30개와 기존 Stage 3 final Quiz 10개를 순차적으로
  checkpoint/resume하는 `run_vehiclemembench_v2_native_quizzes.py` 연결
- **다음 회차:** 1개 scenario Cloud canary로 2,640 turns + 30+10 Quiz 품질·비용 확인

## 평가와 선택 기준

세 경로는 같은 persona/event/final Tool target과 같은 모델 설정을 사용한다.

- V1: history 형식·길이, event fidelity, final Quiz 10개 executable validity와 ESM
- V2: UPDATE/NO_OP 정확성, evidence sufficiency, memory replay, immediate Quiz ESM/Arg
- 품질: 자연스러움, 반복, preference를 억지로 말하는 정도의 blind review
- 효율: dialogue/label/Quiz별 calls, input/cached/output tokens와 latency

5개 pilot은 통계적 우위를 주장하는 실험이 아니라 명백한 실패 경로를 제거하는
scale gate다. 현재 Stage 2 canary는 scenario당 80 events(배경 40 + 차량 40), 2,640
dialogue turns다. Native는 이를 순차 생성하므로 Hybrid의 event당 1회 생성보다
호출·시간이 크게 늘어난다. 따라서 1 scenario canary 없이 곧바로 5개를 실행하지 않는다.

## 성공 조건

- 현재 Stage 2 contract의 80 events, 2,640 turns와 final Quiz 10개를 만족
- 모든 label이 해당 시점 이전 정보만 사용
- 모든 UPDATE가 exact evidence와 deterministic replay를 가짐
- final Quiz와 Turn Quiz 모두 simulator 통과
- Native의 품질 이득이 추가 호출량을 정당화하거나, 그렇지 않으면 Hybrid를 선택할
  명확한 근거를 제공
