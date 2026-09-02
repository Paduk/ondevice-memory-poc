# On-device Memory 방법론·학습 데이터 운영 가이드

## 1. 목적과 공통 원칙

방법론과 데이터 변형을 별개 축으로 관리한다. 같은 Patch라도 `S+T`와 `S+T+V1-10`은
서로 다른 Run이며, Validation/Test split과 seed가 같을 때만 직접 비교한다.

- 현재 기준 모델: `Qwen3.5-4B`; 이후 9B, Granite 3B/8B에 같은 설정을 확장한다.
- 공통 학습: LoRA rank 16/alpha 32, BF16, max length 2048, 4 epochs, seed 45.
- Memory sampling: UPDATE 전부, UPDATE당 NO_OP 5개, 인접 NO_OP 30%, trajectory 20%.
- Quiz multitask: Train Quiz를 전체 학습 동안 정확히 2회 노출한다.
- assistant target에만 loss를 적용하며 입력 memory, turn, Tool schema에는 loss를 주지 않는다.
- 추론 시 reasoning은 기본적으로 생성하지 않는다. `Summary+Reason`만 별도 ablation이다.

## 2. 방법론별 학습 형식

| 방법론 | 입력 | 학습 출력 | 상태 적용 | 주 용도/상태 |
|---|---|---|---|---|
| Summary | 이전 전체 memory + 현재 turn | `NO_OP` 또는 전체 `next_memory` | UPDATE면 전체 교체 | 기본 비교군, 현재 S+T 학습 완료 |
| Patch | 이전 전체 memory + 현재 turn | `NO_OP` 또는 `add/replace/delete` | exact-block deterministic 적용 | 주력 효율 방법, S+T 및 V1 보강 비교 |
| Temporal Patch | Patch 입력 | Patch + `identity_key`, `temporal_action`, `temporal_cue` | temporal 필드를 검증한 뒤 Patch 적용 | 임시값/종료/조건부 상태 학습, Terra-audited label 사용 |
| Appended Delta | base summary + pending deltas + 현재 turn | Patch operation | UPDATE 5회마다 deterministic compaction | token/latency ablation; 현재 주력 Run 아님 |
| Summary+Reason | Summary 입력 | 한 줄 reason/code + Summary 결정 | Summary와 동일 | reasoning decode 비용·판단력 ablation; 주 성능표와 분리 |

`SummaryBatch`/`PatchBatch`는 날짜 단위 Cloud 비교용 adapter이며 현재 turn-wise on-device
주 학습 matrix에는 포함하지 않는다.

## 3. 학습 데이터 변형

공통 데이터 prefix:
`/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/`

| 데이터 변형 | 경로 끝 이름 | Memory Train | Quiz Train | 권장 방법론 |
|---|---|---|---|---|
| Base S1–S100 | `hybrid-s1-s100-summary-patch-delta-v2` | S15–S80 | S15–S80, 2 passes | Summary, Patch, Delta, Summary+Reason |
| S+Temporal | `hybrid-s1-s100-plus-temporal-t1-t20-patch-t1t10-v2` | S15–S80 + T1–T10 | 동일 범위 | Summary, Patch |
| Temporal-audited | `hybrid-s1-s100-plus-temporal-t1-t20-temporal-patch-t1t10-terra-audited-v2` | S15–S80 + T1–T10 | S+Temporal의 Quiz 공유 | Temporal Patch 전용 |
| S+T+V1-10 | `hybrid-s1-s100-plus-temporal-t1-t20-plus-v1-10-patch-v1` | S15–S80 + T1–T10 + V1 10개 | S+T + V1 Quiz 40개 | Patch 전용 distribution 보강 |

시나리오 ID는 V2 `S1–S100=1–100`, Temporal `T1–T20=101–120`, 학습 전용 원본 V1
`S1–S50=201–250`으로 인코딩한다. V1-10은 원본
`S2/S5/S6/S14/S17/S23/S31/S32/S33/S36`만 사용한다.

V1-10 추가 trace는 27,270턴이다. 불량 operation이 없는 시나리오만 채택했으며, soft-30
compaction 14턴은 replay/state에는 유지하되 `train_eligible=false`로 loss와 trajectory에서
제외한다. 최종적으로 UPDATE 545건과 Quiz 40건을 학습에 추가한다.

## 4. Validation 관리

모든 현재 S+T 계열 Run은 다음 Validation을 공유한다.

- Validation 전체 범위: `S81–S85 + T11(111)`
- teacher-forced/one-step: 위 범위에서 최대 512 rows를 고정 seed로 표본 평가
- full closed-loop memory + Quiz: `S83, S84, T11`
- sparse closed-loop memory: `S82, S85`의 NO_OP을 10%만 유지
- Gold-memory Quiz 진단 50건: full closed-loop와 겹치지 않는 `S81, S82, S85`에서 추출
- 주요 지표: Update F1, false UPDATE rate, State F1/final State F1, ESM, Tool F1, Arg Exact

각 epoch 종료 시 adapter checkpoint를 저장하고 Validation을 실행한다. 중간 checkpoint는
500 optimizer steps마다 저장한다. best checkpoint는 closed-loop Quiz ESM을 우선하고,
동률이면 final State F1으로 선택한다. `false_update_rate > 0.20`인 checkpoint는
best-checkpoint 후보에서 제외한다. Validation은 Test 결과를 보고 checkpoint를 다시 고르는
용도로 사용하지 않는다.

참고로 memory catalog의 과거 split 표기는 S81–S90을 validation으로 둘 수 있지만, 현재
논문 실험 정책은 **S81–S85만 Validation, S86–S100은 Test**로 고정한다.

## 5. Test 관리

### V2/Temporal 최종 Test

- 기본 V2: `S86–S100`
- Temporal holdout: `T12–T20(112–120)`
- 통합 Test: `S86–S100 + T12–T20`
- memory는 매 턴 모델 예측을 다음 턴에 넣는 closed-loop predicted-only로 평가한다.
- 해당 시나리오의 Turn/Final Quiz를 예측 memory snapshot으로 풀어 ESM/Tool/Arg를 계산한다.
- Gold memory 평가는 기본 최종 Test에 섞지 않고 필요할 때만 진단용으로 별도 실행한다.

### 원본 VehicleMemBench V1 일반화 Test

- 범위: 원본 `S1–S50`, 시나리오당 Final Quiz 10개
- Gold-memory Quiz와 predicted final-memory Quiz를 분리해 memory 병목과 Agent 병목을 본다.
- V1-10 보강 Patch는 학습에 사용한 10개를 `seen`, 나머지 40개를 `unseen`으로 반드시
  분리 보고한다. 전체 S1–S50 평균만 기존 모델과 비교하면 데이터 leakage로 오해될 수 있다.

## 6. Run 및 산출물 관리

- workspace: `/mnt/data/hj153lee/PalmClaw/on-device-memory-training/`
- Run 이름: `<model>-<method>-multitask-noop5-<dataset>-rN`
- Run별 저장: `config.json`, `metrics.jsonl`, `checkpoints/epoch-XX`, step checkpoints,
  `validation-*.json`, Test output, `best-checkpoint.json`
- Dashboard: SSH tunnel 사용 시 `http://127.0.0.1:15060/`; Train/Validation/Test 상태와 GPU,
  loss, 주요 metric을 Run별 및 선택 Run 비교로 확인한다.
- 재개는 epoch/step checkpoint의 adapter, optimizer, scheduler, progress를 함께 사용한다.
- 데이터 root와 catalog/manifest는 Run 시작 후 수정하지 않는다. label 또는 split이 바뀌면
  새 데이터 디렉터리와 새 Run ID를 만든다.

## 7. 현재 권장 실험 matrix

| 비교 목적 | Run A | Run B | 고정 Validation/Test |
|---|---|---|---|
| 표현 방식 | Summary S+T | Patch S+T | S81–85+T11 / S86–100+T12–20 |
| temporal 효과 | Patch S+T | Temporal Patch audited | 동일 |
| V1 distribution 보강 효과 | Patch S+T | Patch S+T+V1-10 | 동일; 추가로 V1 seen/unseen 분리 |
| reasoning 비용 | Summary | Summary+Reason | 동일, 별도 ablation 표 |
| delta 효율 | Patch 또는 Summary Base | Appended Delta Base | S81–85 / S86–100 |

V1-10 Patch 학습 실행기는
`memory_training/scripts/run_qwen35_4b_patch_v1_10_mix.sh <GPU>`이며, 다른 방법론과 공정하게
비교할 때는 epoch 수, seed, sampling, Quiz passes 및 checkpoint 선택 규칙을 바꾸지 않는다.
