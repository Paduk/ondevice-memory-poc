# GPU 작업 큐 모니터링: 2026-09-05 ~ 2026-09-06

## 운영 범위

- 모니터링 시작: 2026-09-05 14:55 UTC
- 모니터링 종료: 2026-09-06 11:00 UTC
- 주기: 1시간에 한 번
- 목표: 아래 GPU별 작업을 끝까지 완료하고, 예상치 못한 종료나 후속 작업 미기동을
  발견하면 기존 checkpoint와 완료 artifact를 보존하면서 재개한다.
- GPU 5의 `steve97` Ray 작업은 관찰만 하며 변경하지 않는다.

매 점검은 이 문서의 **점검 이력**에 기록한다. 문제가 없으면 `정상(문제 없음)`을
명시하고, 문제가 있으면 증상, 원인, 조치, 복구 결과를 간략히 기록한다.

## GPU별 목표와 완료 조건

| GPU | 작업 큐 | 현재 기준점 | 최종 완료 조건 |
|---:|---|---|---|
| 0 | Qwen3.5 2B Delta-v3 compact `k=2` 학습 → fixed Validation → fixed Test → Stress Test | 학습 epoch 3, step 2,865/3,287 | fixed Test `complete=true` 24/24 후 stress update 20/40/60/80의 cache·composite 완료 |
| 1 | Qwen3.5 2B Delta-v3 compact `k=5` 학습 → fixed Validation → fixed Test → Stress Test | 학습 epoch 3, step 2,715/3,290 | fixed Test `complete=true` 24/24 후 stress update 20/40/60/80의 cache·composite 완료 |
| 2 | Qwen3.5 0.8B Summary Stress Test | update 20/40/60 완료, update 80 cache 실행 중 | update 80 cache와 composite 완료 |
| 3 | Granite 4 1B Summary Stress Test | update 20/40/60 완료, update 80 cache 실행 중 | update 80 cache와 composite 완료 |
| 4 | Granite 4 350M Summary Stress Test | update 20/40 완료, update 60 cache 실행 중 | Summary update 20/40/60/80의 cache·composite 완료 |
| 5 | 외부 사용자 작업 | `steve97` Ray worker, GPU memory 16.3 GiB | 모니터링 대상 작업과의 간섭 여부만 관찰 |
| 6 | Granite 4 350M Patch → Delta-v3 compact `k=2,5,10` Stress Test | Patch 완료, `k=2` update 60 cache 실행 중 | Patch와 k2/k5/k10 각각 update 20/40/60/80의 cache·composite 완료 |
| 7 | Qwen3.5 2B Delta-v3 compact `k=10` 학습 → fixed Validation → fixed Test → Stress Test | 학습 epoch 3, step 2,625/3,297 | fixed Test `complete=true` 24/24 후 stress update 20/40/60/80의 cache·composite 완료 |

## 예약 및 실행 정보

### Qwen3.5 2B compact

- 학습 파이프라인 tmux: `qwen2b-k2-gpu0`, `qwen2b-k5-gpu1`, `qwen2b-k10-gpu7`
- Stress 예약 tmux: `qwen2b-compact-stress-20260905`
- Stress 출력:
  `/mnt/data/hj153lee/PalmClaw/on-device-memory-training/benchmarks/qwen35-2b-stress-test-once-20260905-v1`
- GPU별 fixed Test 완료를 독립적으로 기다린 뒤 같은 GPU에서 k2/k5/k10 stress를 병렬 실행한다.
- Summary와 Patch는 이 예약에 포함하지 않는다.

### Summary 및 Granite 4 350M

- Qwen3.5 0.8B·Granite 4 1B Summary 실행 스크립트:
  `memory_training/scripts/run_qwen_granite_summary_stress_test_once.sh`
- Granite 4 350M tmux: `granite350m-stress-20260905`
- Granite 4 350M 실행 스크립트:
  `memory_training/scripts/run_granite4_350m_stress_test_once.sh`
- Granite 4 350M은 GPU 4에서 Summary, GPU 6에서 Patch와 compact k2/k5/k10을
  순차 실행한다.

## 점검 및 복구 기준

다음 조건을 이상으로 취급한다.

- 완료되지 않은 작업의 프로세스 또는 tmux 세션이 사라짐
- 학습 `status.json` 또는 `metrics.jsonl`이 한 시간 이상 갱신되지 않음
- 실행 로그에 traceback, CUDA OOM, non-zero exit, `ERROR`가 기록됨
- 후속 Validation, Test, Stress 예약이 선행 작업 완료 후 시작되지 않음
- Test summary가 `complete=true` 또는 예상 scenario 수를 충족하지 못함
- GPU memory는 점유하지만 프로세스·로그·artifact가 함께 진행되지 않음

복구 시 다음 원칙을 적용한다.

- 기존 checkpoint와 정상 완료 artifact를 삭제하거나 덮어쓰지 않는다.
- idempotent skip/reuse 로직을 이용해 미완료 단계부터 재개한다.
- 재실행 전 대상 GPU의 충돌 여부와 정확한 실패 단계를 확인한다.
- 이 문서 범위의 `hj153lee` 작업만 변경하며 GPU 5의 외부 작업은 건드리지 않는다.
- 모든 장애와 조치를 아래 이력에 기록한다.

## 점검 이력

| 시각 (UTC) | 상태 | 진행 요약 | 문제 및 조치 |
|---|---|---|---|
| 2026-09-05 14:55 | 정상 | GPU 0/1/7 2B 학습 진행 중; GPU 2/3 Summary update 80 cache; GPU 4 350M Summary update 60 cache; GPU 6 350M k2 update 60 cache; 2B stress 예약 3개 모두 대기 중 | 현재 실행 중인 프로세스의 GPU·CPU 활동과 로그 갱신 확인. 문제 없음. 기준점 이전에 Granite 350M Summary update 60 cache와 k2 update 40 composite가 재시작된 흔적이 있으나, 완료 artifact 재사용 방식으로 이미 재개되어 현재 정상 진행 중. |
| 2026-09-05 15:55 | 정상(문제 없음) | GPU 0/1/7 2B 학습 step 3,110/2,940/2,820으로 진행; GPU 2 Qwen Summary update 80 composite 진입; GPU 3 Granite 1B Summary update 80 cache; GPU 4 Granite 350M Summary update 60 cache; GPU 6 Granite 350M k2 전체 완료 후 k5 update 40 cache 진입; 2B stress 예약 정상 대기 | 오류 로그와 stale 상태 없음. GPU 2에 `hgkim` 프로세스가 3.1 GiB를 함께 사용하기 시작했으나 본 작업도 GPU·CPU 활동 중이고 오류가 없어 관찰만 유지. 외부 프로세스는 변경하지 않음. |
| 2026-09-05 16:55 | 정상(문제 없음) | GPU 0 k2 학습·epoch 3/4 fixed Validation 완료 후 fixed Test 5/24 진행; GPU 1/7 k5/k10 학습 step 3,175/3,030; GPU 2 Qwen 0.8B Summary stress 전 단계 완료; GPU 3 Granite 1B Summary update 80 cache; GPU 4 Granite 350M Summary update 80 cache 진입; GPU 6 Granite 350M k5 완료 후 k10 update 20 composite 진입; 2B stress 예약 정상 대기 | 최근 오류 없음. k2의 학습 `state=COMPLETED`만으로 stress가 조기 실행되지 않고 selection과 24/24 Test를 기다리는 예약 조건이 정상 작동함. Qwen 0.8B Summary는 cache summary와 composite `complete=true`를 update 20/40/60/80 모두 확인함. |
| 2026-09-05 17:55 | 정상(문제 없음) | GPU 0 k2 fixed Test 24/24 완료 후 Stress Test가 정상 기동되어 update 40 composite 완료, update 60 cache 진행; GPU 1 k5 학습·Validation 완료 후 fixed Test 21/24; GPU 7 k10 학습 3,250/3,297; GPU 2 Qwen 0.8B Summary 완료 상태 유지; GPU 3 Granite 1B Summary update 80 cache; GPU 4 Granite 350M Summary update 80 cache; GPU 6 Granite 350M k5 완료 후 k10 update 80 composite 진행 | 최근 오류와 stale 상태 없음. GPU 6은 최초 순간 사용률이 0%였으나 프로세스 CPU 73% 및 재측정 GPU 47%로 정상 활동을 확인해 조치하지 않음. |
| 2026-09-05 18:55 | 정상(문제 없음) | GPU 0/1/7의 2B k2/k5/k10 학습·fixed Validation·fixed Test 24/24가 모두 완료되고 각 Stress Test가 정상 자동 기동됨: k2 update 80 cache, k5 update 60 cache, k10 update 20 composite 진행; GPU 2 Qwen 0.8B Summary 완료; GPU 3 Granite 1B Summary update 80 composite; GPU 4 Granite 350M Summary update 80 cache; GPU 6 Granite 350M Patch·compact k2/k5/k10 전 단계 완료 | 최근 오류와 stale 상태 없음. 세 2B 예약 체인이 선행 Test 완료 후 각 GPU를 정상 인계받은 것을 확인했으며 별도 조치 없음. |
| 2026-09-05 19:55 | 정상(문제 없음) | GPU 0 Qwen3.5 2B k2 전체 큐 완료; GPU 1 k5 Stress update 80 composite 진행; GPU 7 k10 Stress update 60 composite 진행; GPU 2 Qwen 0.8B Summary 완료 상태 유지; GPU 3 Granite 1B Summary update 80 composite 완료로 전체 큐 완료; GPU 4 Granite 350M Summary update 80 composite 진행; GPU 6 Granite 350M 큐 완료 상태 유지 | 최근 오류와 stale 상태 없음. GPU 0·2·3·6은 목표 작업을 완료해 유휴 상태이며 남은 GPU 1·4·7 프로세스는 정상 활동 중. |
| 2026-09-05 20:55 | 정상(문제 없음) | GPU 0/1/7 Qwen3.5 2B k2/k5/k10의 학습·Validation·Test·Stress 전체 완료; GPU 2 Qwen 0.8B Summary 완료; GPU 3 Granite 1B Summary 완료; GPU 4/6 Granite 350M Summary·Patch·compact k2/k5/k10 완료 | 모든 지정 작업의 update 20/40/60/80 cache summary와 composite `complete=true` 5/5를 확인함. 최근 오류 없음. GPU 0에 `skjinlee`의 새 비대상 작업이 올라왔으나 지정 작업은 이미 완료됐으므로 관찰만 유지. |
| 2026-09-05 21:55 | 정상(문제 없음) | 모든 지정 작업 완료 상태 유지; GPU 0/1/2/3/4/6/7 유휴, GPU 5 외부 Ray 작업만 16.3 GiB 점유 | 2B fixed Test 3건과 전체 Stress cache·composite artifact 재검증 통과. 최근 한 시간 오류 없음. |
| 2026-09-05 22:55 | 정상(문제 없음) | 모든 지정 작업 완료 상태 유지; GPU 0/1/2/3/4/6/7 유휴, GPU 5 외부 Ray 작업만 16.3 GiB 점유 | 주요 최종 artifact와 2B stress scheduler 완료 상태 확인. 최근 한 시간 오류 없음. |
| 2026-09-05 23:55 | 정상(문제 없음) | 모든 지정 작업 완료 상태 유지; GPU 0/1/2/3/4/6/7 유휴, GPU 5 외부 Ray 작업만 16.3 GiB 점유 | 완료 마커와 주요 update 80 composite artifact 확인 통과. 최근 한 시간 오류 없음. |
| 2026-09-06 00:23 | 정상(최종 점검) | 모든 지정 작업 완료 상태 유지; GPU 0/1/2/3/4/6/7 유휴, GPU 5 외부 Ray 작업만 16.3 GiB 점유 | 2B fixed Test 3건(각 24/24)과 전체 Stress cache·composite artifact 재검증 통과. 완료 이후 오류 없음. 사용자 요청에 따라 11:00 UTC 이전에 모니터링을 종료함. |

## 종료 요약

- 종료 시각: 2026-09-06 00:23 UTC (사용자 요청에 따른 조기 종료)
- 최종 결과: 계획된 모든 학습·Validation·Test·Stress Test 완료
- 모니터링 중 예상치 못한 종료 또는 신규 장애: 없음
- 장애 복구 조치: 없음. 최초 기준점 이전의 Granite 350M 재시작 흔적은 완료 artifact 재사용 방식으로 이미 정상 재개되어 이후 끝까지 완료됨.
