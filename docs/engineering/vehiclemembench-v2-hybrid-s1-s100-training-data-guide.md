# VehicleMemBench V2 Hybrid S1–S100 데이터·학습 가이드

## 1. 데이터 범위와 경로

- 총 100개 시나리오: 기존 Hybrid `S1–S20` + state-evolution Hybrid `S21–S100`
- 기준 루트: `/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-hybrid`
- canonical run:
  - S1: `hybrid-full-terra-s1-r2`
  - S2·S4·S5: `hybrid-pilot-terra-sNN-r2`
  - S3·S6–S100: `hybrid-pilot-terra-sNN-r1`

시나리오별 핵심 파일은 다음과 같다.

| 파일 | 용도 |
|---|---|
| `stage2-v2-anchored.json` | persona, event chain, preference update 원천 |
| `dialogues/<event_id>.json` | event별 원문 대화 |
| `hybrid.json` | Turn-wise memory와 학습 label의 canonical source |
| `turn-quiz-30-v1/turn-quizzes.json` | Turn Quiz 30개와 gold Tool/Arguments/state |
| `final-v1/stage3.json` | Final Quiz 10개와 전체 dialogue |

품질 평가 결과는 `.../vehiclemembench-v2-hybrid-state-evolution-{s21-s25,s26-s50,s51-s100}-evaluation`에 있다. S21–S100 합산 기준 Answerability Full Support는 `96.0%`, Agent ESM은 `0.947`이다.

세 방법의 최신 학습 데이터 루트는 다음과 같다.

```text
/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/
  hybrid-s1-s100-summary-patch-delta-v2/
```

이전 `hybrid-s1-s100-summary-delta-v1`은 Patch JSONL이 없는 과거 산출물이므로 신규
3-way 학습에는 사용하지 않는다.

### 사용자별 grouped 파생본 (S1–S100 + T1–T20)

기존 flat append-log는 보존하고, 사용자별 메모리 블록을 갖는 학습 파생본을 추가했다.

```text
/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/
  grouped-s1-s100-plus-temporal-t1-t20-v2/
```

- V1과 같은 `### <speaker_name>` 아래에 해당 사용자의 fact만 시간순으로 둔다.
- `speaker_id`는 메모리 본문에서 제거하고 row metadata에만 보존한다.
- `shared-vehicle`은 별도 공용 블록으로 유지한다.
- Summary/Patch/Delta-v2와 Quiz memory hash를 추가 LLM 호출 없이 함께 변환한다.
- Delta-v2는 inline `add/replace/delete` 배열과 5 UPDATE 단위 deterministic compaction을 사용한다.
- 335,504 Turn, 1,724 UPDATE, Quiz 4,800건을 전수 replay/hash 검증했다.
- 원본 fact multiset, sample ID, split, Quiz 정답은 변하지 않는다.

변환기는 `memory_training/prepare_grouped_memory_data.py`, 학습 method는
`memory_training/methods/delta_v2.py`에 있다. 이후 사용자별 메모리 실험은 이 파생본을
우선 사용하고, flat 원본은 비교·복구용 canonical source로 유지한다.

실제 noop5 학습에는 기존 V1 10개 시나리오를 training-only로 추가한 다음 통합본을
사용한다. Validation `S81–S85 + T11`과 Test `S86–S100 + T12–T20`은 grouped V2 원본을
그대로 유지한다.

```text
/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/
  grouped-s1-s100-plus-temporal-t1-t20-plus-v1-10-v2/
```

## 2. 학습·평가에 필요한 속성

`hybrid.json/event_checkpoints[].turn_labels[]`에서 시간순으로 아래 값을 추출한다.

- 입력: 직전 `after_memory`, 현재 `turn_id`의 speaker/text
- 판단 label: `decision` (`NO_OP|UPDATE`), `reason_code`, `reason`
- Delta label: `operations[]` (`add|replace|delete`, `target`, `content`)
- Summary label: 현재 label의 `after_memory`
- 추적 정보: `scenario_id`, `event_id`, turn index, evidence, source update index, before/after hash

Quiz 평가는 `query`, `reasoning_type`, `gold_calls`, `target_state`, `memory_snapshot_sha256`를 사용한다. Split은 반드시 **scenario 단위**로 만들고 세 학습 view에서 동일하게 고정한다.

내보낸 Quiz는 공통적으로 다음 연결 정보를 갖는다.

```text
memory_ref = scenario_index + global_turn_index + turn_id
             + memory_snapshot_sha256
             + summary/patch/delta sample_id
```

따라서 하나의 Quiz가 세 방법의 동일한 시점 메모리를 참조하며, 메모리 전문을 Quiz
파일에 중복 저장하지 않는다. 정답은 정규화된 `gold_calls[{name, arguments}]`, simulator
`target_state`와 hash를 모두 포함한다.

## 3. Summary / Patch / Appended Delta 변환 방식

하나의 canonical turn sample에서 세 SFT view를 추가 LLM 호출 없이 만든다. 세 파일은
동일한 `scenario_index`, `global_turn_index`, `turn_id`, split과 provenance를 공유한다.

### Summary view

```text
input  = previous_memory + current_turn
target = decision + reason + next_memory(after_memory)
```

- `NO_OP`: `next_memory == previous_memory`
- `UPDATE`: 기존 유효 사실을 포함한 전체 memory를 출력

### Patch Baseline view

```text
input  = previous_memory + current_turn
target = decision + reason + operations[]
```

- `NO_OP`: 빈 operation을 출력하고 memory를 변경하지 않는다.
- `UPDATE`: `add/replace/delete` Patch를 출력한다.
- deterministic executor가 Patch를 전체 memory에 즉시 적용한다.
- 적용된 전체 memory가 바로 다음 Turn의 `previous_memory`가 된다.

### Appended Delta + 5-UPDATE compaction view

```text
runtime_memory = base_summary + pending_deltas
turn target    = decision + reason + operations[]
```

- `NO_OP`: 빈 operation
- `UPDATE`: 필요한 `add/replace/delete`만 출력
- `NO_OP`은 compaction counter에 포함하지 않고 `UPDATE`만 센다.
- `UPDATE` 5회마다 `pending_deltas`를 적용한 현재 원본 `after_memory`를 새 `base_summary` label로 삼고 Delta를 비운다.
- 시나리오 종료 시 남은 1–4개 Delta도 최종 Summary로 flush한다.
- deterministic executor가 각 operation 및 compaction을 적용한 결과는 반드시 원본 `after_memory` 및 hash와 같아야 한다.

따라서 학습 데이터는 매 Turn의 **Delta 판단 sample**과 5회 주기의 **Summary compaction sample**로 분리한다. 현재 표본은 시나리오당 UPDATE가 약 13–16회이므로 Summary 전체 출력은 약 13–16회에서 3–4회로 감소한다.

### Quiz view

- `turn_quiz.jsonl`: 각 중간 cutoff 직후의 메모리로 푸는 Quiz 30개/시나리오
- `final_quiz.jsonl`: 모든 Turn 처리 후 최종 메모리로 푸는 Quiz 10개/시나리오
- Turn Quiz는 원래 기억이 생긴 source checkpoint와 실제 평가 시점 cutoff checkpoint를
  provenance에서 별도로 보존한다.
- Final Quiz는 각 시나리오의 마지막 Turn과 `final_memory_sha256`을 참조한다.
- Quiz/정답은 memory updater SFT 입력에 섞지 않고 Agent 평가 또는 별도 Agent 학습에만
  사용한다.

## 4. 구현·내보내기 결과

- 구현: `ubuntu/src/palmclaw_ubuntu/vehicle_bench/v2_training_export.py`
- 실행기: `ubuntu/evaluation/experiment-scripts/export_vehiclemembench_v2_hybrid_training_views.py`
- 산출물: `/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/hybrid-s1-s100-summary-patch-delta-v2`
- 고정 split: Train S1–S80 / Validation S81–S90 / Test S91–S100

| 파일 | 건수 | 용도 |
|---|---:|---|
| `summary.jsonl` | 276,938 | 매 Turn 전체 Summary 학습 |
| `patch.jsonl` | 276,938 | 전체 memory 입력 기반 Patch Baseline 학습 |
| `delta.jsonl` | 276,938 | base Summary + 누적 Patch 입력 기반 Delta 학습 |
| `compaction.jsonl` | 291 | 5 UPDATE 또는 종료 시 누적 Patch 병합 학습 |
| `summary_batch.jsonl` | 7,815 | 날짜별 전체 Turn + 이전 memory → 날짜 마지막 Summary |
| `patch_batch.jsonl` | 7,815 | 날짜별 전체 Turn + 이전 memory → 시간순 Patch 목록 |
| `batch_manifest.json` | 1 | Batch 원천·건수·SHA-256·시나리오별 검증 결과 |
| `manifest.json` | 1 | 입력 run, split, 건수, SHA-256 기록 |
| `turn_quiz.jsonl` | 3,000 | Turn cutoff Quiz, Tool/Arguments/target state 정답 |
| `final_quiz.jsonl` | 1,000 | 시나리오 최종 Quiz, Tool/Arguments/target state 정답 |
| `quiz_manifest.json` | 1 | Quiz 원천, 건수, split, training manifest 연결과 SHA-256 |

전체 276,938 Turn을 replay하여 `NO_OP 275,630`, `UPDATE 1,308`, operation 1,398건을 확인했다. hash-chain, deterministic patch replay, 5-UPDATE compaction과 final flush 검증 및 관련 테스트 16개를 통과했다.

`summary.jsonl`, `patch.jsonl`, `delta.jsonl`을 행 단위로 전부 교차검사한 결과도 오류
0건이다. Patch 적용 결과와 다음 Summary/Patch 입력, Delta의 `base_summary +
pending_deltas`, 215회 5-UPDATE compaction과 76회 final flush가 모두 일치한다.

Turn Quiz 3,000개와 Final Quiz 1,000개의 target-state hash 및 Tool 정답을 검증했고,
4,000개 `memory_ref`가 Summary/Patch/Delta의 동일 Turn 및 memory hash와 모두 연결됨을
확인했다.

날짜 단위 Batch view도 추가 LLM 호출 없이 같은 canonical Turn label에서 결정론적으로
파생했다. 전체 276,938 Turn은 7,815개 날짜 Batch로 변환됐고, `UPDATE Batch 1,306`,
`NO_OP Batch 6,509`다. 한 날짜에 UPDATE가 두 번 발생한 2개 Batch는 operation을
시간순으로 모두 보존한다. 모든 Batch Patch를 직전 memory에 replay한 결과가
`summary_batch.target.next_memory` 및 각 시나리오의 최종 memory hash와 일치함을
100개 시나리오 전수 확인했다.

```text
Summary Batch:
  input  = previous_memory + 같은 날짜의 전체 turns[]
  target = decision + update/no-op count + 날짜 마지막 next_memory

Patch Batch:
  input  = previous_memory + 같은 날짜의 전체 turns[]
  target = decision + update/no-op count + 시간순 operations[]
```

Batch view는 날짜 마지막 memory를 손실 없이 보존하지만, Batch 내부의 중간 memory를
평가하려는 용도가 아니다. 원본 turn-wise JSONL은 그대로 유지하며 두 view를 별도
학습·효율 비교에 사용한다.

멀티태스크에서는 Batch 파일 안에 Quiz를 중복 저장하지 않는다. 기존
`quiz_sft.jsonl`의 Gold Tool Call과 공식 `vehicle_tools.json`을 Summary/Patch Batch
Memory SFT 사이에 동일하게 삽입한다. 즉 학습 target은 memory output과 Gold Tool
Call이며, ESM·Tool F1·Argument Exact는 직접 label로 예측하지 않고 validation/test의
simulator 실행으로 산출한다. `batch_manifest.json`이 이 공유 파일과 S15–S80 Train,
S81–S85 Validation, S86–S100 Test 정책을 명시한다.

원본 분포는 NO_OP가 매우 많으므로 실제 SFT에서는 이 JSONL을 그대로 균등 소비하지 말고, **scenario split은 유지한 채 Train의 NO_OP만 downsampling/가중치 조정**해야 한다. Validation/Test는 원래 분포를 보존한다.

## 5. Temporal / Current 확장 계획

### 현재 데이터의 범위와 한계

기존 Patch label에 V1 Combined의 temporal metadata를 결정론적으로 부착한 파생본은
다음 경로에 준비되어 있다.

```text
/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/
  hybrid-s1-s100-temporal-patch-v1/
```

- 구현: `ubuntu/src/palmclaw_ubuntu/vehicle_bench/v2_temporal_training_export.py`
- 실행기: `ubuntu/evaluation/experiment-scripts/export_vehiclemembench_v2_temporal_patch_dataset.py`
- 학습 method: `memory_training/methods/temporal_patch.py`
- 전체 276,938 Turn의 before/after hash-chain과 Patch replay를 모두 통과했다.

| Temporal action | operation 수 |
|---|---:|
| `conditional_upsert` | 1,291 |
| `durable_upsert` | 16 |
| `non_temporal` | 90 |
| `temporary_override` | 1 |
| `current_upsert` | 0 |
| `end_temporary` | 0 |

이 파생본은 기존 canonical memory를 바꾸지 않는 **손실 없는 annotation view**다.
따라서 학습 형식 검증에는 사용할 수 있지만, 현재 S1–S100만으로는 temporary/current
처리의 성능 이득을 판단할 수 없다. 또한 시나리오별 최대 ADD는 13개, 최대 memory는
527 tokens라 V1 Combined의 soft-30 compaction 조건(누적 ADD 64개 또는 1,000 tokens)을
만족하는 시나리오가 없다.

### 추가 데이터 생성 원칙

기존 S1–S100을 임의 변형하지 않고 별도 `V2-Temporal` 시나리오 묶음을 추가한다.
대화보다 먼저 structured gold state와 상태 전이를 확정하고, 대화는 그 전이를
자연어로 표현하도록 생성한다.

기존 S1–S100은 Temporal을 전제로 생성하지 않았으므로 그대로 legacy/general
학습 데이터로 고정한다. 기존 trajectory에 temporary/current 의미를 사후 추정해
덮어쓰지 않는다. 신규 Temporal label은 T01–T20에서만 생성한다.

Memory updater의 입력 계약도 기존 방법과 동일하게 고정한다.

```text
previous approved memory + current turn -> NO_OP 또는 UPDATE
```

- 최근 2–3개 turn이나 event prefix를 별도 입력으로 추가하지 않는다.
- 이전 turn의 미확정 제안은 approved memory에 저장하지 않는다.
- UPDATE는 `previous memory + current turn`만으로 결정 가능한 첫 turn에 배치한다.
- 임시 설정 시작은 subject, value와 temporal scope가 한 발화에 함께 재확인된
  self-contained turn에서만 실행한다. 그 전의 제안·부분 확인은 NO_OP이다.
- 임시 설정 종료는 previous memory에 durable baseline과 active override가 있으므로,
  current turn의 명시적 종료 cue만으로 override를 deterministic delete한다.
- 필요한 self-contained turn이 생성되지 않으면 label을 억지로 이동하지 않고 해당
  dialogue를 재생성한다.

각 시나리오는 가능한 한 다음 흐름을 포함한다.

```text
durable baseline
  -> observed current setting
  -> explicit temporary override
  -> unrelated / NO_OP turns
  -> explicit temporary termination
  -> durable baseline restoration
  -> conditional preference or durable correction
```

- `current_upsert`는 현재 관측값만 기록하고 durable preference를 덮어쓰지 않는다.
- `temporary_override`는 종료 조건·기간을 current-turn evidence에 명시하고 baseline을
  별도 항목으로 보존한다.
- `end_temporary`는 명시적 종료 evidence가 있을 때만 실행하고 baseline을 복원한다.
- 단순 시간 경과만으로 temporary 상태의 종료를 추론하지 않는다.
- current/temporary 표현이 등장하지만 memory를 바꾸면 안 되는 hard-negative NO_OP도
  포함한다.
- 최종 Quiz뿐 아니라 각 상태 전환 직후 Turn Quiz를 배치해 중간 상태를 직접 평가한다.

### 생성·검증 단계

1. **Gold state machine:** `identity_key`, `temporal_action`, value, condition, 종료 cue와
   복원할 baseline을 먼저 생성한다.
2. **Gold anchor turn:** 각 preference update마다 `previous memory + current turn`만으로
   완전히 해석 가능한 한 개의 anchor 발화를 별도 structured call로 생성한다. anchor는
   event/update key, 허용 speaker, setting hint, value, condition과 temporal cue를 전수
   검증하고 먼저 동결한다.
3. **Dialogue materialization:** LLM은 동결된 anchor의 speaker/text를 정확히 한 번
   포함하면서 주변의 자연스러운 대화와 NO_OP turn을 생성한다. anchor가 변형·누락되면
   해당 event dialogue만 재생성한다.
4. **Deterministic alignment:** 별도 alignment LLM이나 사후 cue 검색을 사용하지 않는다.
   dialogue 안의 exact anchor 위치가 곧 UPDATE turn이며 나머지는 NO_OP이다.
5. **Temporal Patch export:** Turn별 `NO_OP|UPDATE`와 temporal operation을 만들고,
   exact evidence, before/after hash, deterministic replay, baseline 복원을 전수 검증한다.
6. **Pilot 후 확장:** 우선 10개 안팎의 pilot으로 action 분포와 Quiz 난도를 점검한 뒤,
   scenario 단위 Train/Validation/Test로 확장한다. 같은 persona·event chain과 전이
   조합은 split 사이에 공유하지 않는다.

전체 생성 전에는 focus별 대표인 `T01`, `T05`, `T09`, `T13`을 순서대로 통과시킨다.
각 pilot은 self-contained UPDATE, exact current-turn evidence, deterministic Patch
replay, temporary baseline 보존·복원 및 Turn/Final Quiz 완주를 모두 만족해야 한다.
T01의 anchor-first 재생성은 검증된 Stage 2 `r2`를 재사용하되 기존 실패 산출물을
덮어쓰지 않는 별도 `hybrid-temporal-anchor-*` 경로에서 시작한다.

초기 dialogue-first T01은 hard gate와 자연 대화의 우연한 동시 충족에 의존해 총 8회의
파이프라인 중단이 발생했다. 따라서 `r1/r2`는 진단용으로 보존하고, anchor-first
파이프라인의 최초 결과는 별도 revision에서 생성한다. 한 anchor 실패는 전체 scenario를
무효화하지 않고 해당 anchor 또는 해당 event dialogue만 제한 재시도한다.

평가는 기존 ESM/Tool F1/Arg Exact/State F1에 `temporal_action F1`, current와 durable의
혼동률, temporary 종료 후 baseline 복원 정확도, 불필요한 UPDATE 비율을 추가한다.
Soft-30 compaction은 이 실험에 억지로 섞지 않고, memory가 실제 임계값을 넘는 별도의
long/noisy stress subset에서 효과를 측정한다.

## 6. 다음 학습 순서

1. Train의 NO_OP sampling 비율과 loss weight를 고정한다.
2. 동일 모델·학습 budget으로 Summary, Patch Baseline과 Appended Delta를 각각 학습한다.
3. Validation S81–S90의 Quiz로 checkpoint를 선택하고 Test S91–S100은 최종 1회만 평가한다.
4. `turn_quiz.jsonl`과 `final_quiz.jsonl`에서 ESM/Tool F1/Arg Exact/State F1, output token, latency를 비교한다.
