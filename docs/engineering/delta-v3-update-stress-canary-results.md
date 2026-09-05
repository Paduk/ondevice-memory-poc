# Delta-v3 Update Stress KV-cache Canary 결과

## 실행 범위

- 날짜: 2026-09-04
- 데이터: Validation `S81-S85`, 시나리오당 총 UPDATE `20/40/60/80`
- 모델: Qwen3.5 0.8B seed45 `epoch-03`
- 방법: Patch, Delta-v3 `k=2/5/10`
- 설정: HF, cache ON, controlled replay, background prefill OFF, warm-up 3, 반복 1회
- 장치: A100-SXM4-80GB GPU 0/1/2/7; 방법별 GPU를 고정해 병렬 실행
- Cache canary: 16/16 job 완료, OOM/context overflow/truncation/실행 실패 없음
- Composite 평가: predicted closed loop 16/16 job 및 Gold Quiz 3,200건 완료

이 결과는 실행 가능성과 효과 방향을 확인하는 **1회 canary**이며 최종 통계가 아니다.
서로 다른 GPU에서 동시에 실행했으므로 HF wall-clock보다 deterministic한 evaluated
prefill token을 주 지표로 본다.

## 핵심 결과

### UPDATE 직후 요청

UPDATE가 캐시 재사용에 미친 비용은 UPDATE를 생성한 턴이 아니라 그 **다음 요청**에서
측정한다.

| UPDATE/시나리오 | Patch | Delta k=2 | Delta k=5 | Delta k=10 |
|---:|---:|---:|---:|---:|
| 20 | 559.8 | 367.0 (-34.4%) | **364.3 (-34.9%)** | 525.3 (-6.2%) |
| 40 | 956.5 | 564.2 (-41.0%) | **441.9 (-53.8%)** | 582.8 (-39.1%) |
| 60 | 1,354.7 | 763.2 (-43.7%) | **522.2 (-61.5%)** | 618.8 (-54.3%) |
| 80 | 1,754.5 | 964.0 (-45.1%) | **602.9 (-65.6%)** | 659.9 (-62.4%) |

단위는 평균 evaluated prefill tokens이며 괄호는 Patch 대비 절감률이다. 30 prefill
tok/s 기기에서 80 UPDATE 평균을 단순 환산하면 Patch `58.5s`, k2 `32.1s`, k5
`20.1s`, k10 `22.0s`이다.

### UPDATE 직후 p95

| UPDATE/시나리오 | Patch | Delta k=2 | Delta k=5 | Delta k=10 |
|---:|---:|---:|---:|---:|
| 20 | 846.6 | 805.3 | **754.3** | 956.3 |
| 40 | 1,579.5 | 1,520.1 | 1,357.7 | **1,107.1** |
| 60 | 2,345.2 | 2,229.6 | 1,959.7 | **1,511.1** |
| 80 | 3,105.1 | 2,948.8 | 2,559.8 | **1,911.4** |

k5가 평균 token 비용은 가장 작고, k10은 compaction 빈도가 낮아 40 UPDATE부터 p95가
가장 작다. 즉 `k`는 평균과 tail 사이의 실제 trade-off를 만든다.

### 전체 턴 평균

| UPDATE/시나리오 | Patch | Delta k=2 | Delta k=5 | Delta k=10 |
|---:|---:|---:|---:|---:|
| 20 | 204.8 | 149.9 | **149.3** | 185.5 |
| 40 | 415.3 | 261.9 | **217.0** | 268.8 |
| 60 | 684.5 | 401.3 | **289.2** | 334.1 |
| 80 | 988.9 | 557.6 | **363.7** | 394.3 |

80 UPDATE에서 총 evaluated prefill tokens는 Patch `736,754`, k2 `415,407`, k5
`270,934`, k10 `293,754`였다. k5는 Patch 대비 총 prefill 계산을 63.2% 줄였다.

### NO_OP 턴 평균

| UPDATE/시나리오 | Patch | Delta k=2 | Delta k=5 | Delta k=10 |
|---:|---:|---:|---:|---:|
| 20 | 214.7 | **155.5** | 156.4 | 192.5 |
| 40 | 498.4 | 311.8 | **245.7** | 311.3 |
| 60 | 931.3 | 525.5 | **379.1** | 432.1 |
| 80 | 1,559.6 | 854.8 | **558.5** | 584.0 |

NO_OP 자체는 state를 바꾸지 않지만, 직전 UPDATE로 손상된 prefix의 비용이 다음 NO_OP에
반영되므로 stress 규모가 커질수록 방법 간 차이가 커진다.

## 정식 Stress Composite

같은 checkpoint와 데이터로 predicted closed loop를 별도 실행했다. 다섯 시나리오의
state와 Quiz snapshot은 독립적으로 유지한 채 accuracy 평가만 `scenario_batch_size=5`로
가속했다. 각 규모에는 원본 Gold Quiz 200개가 포함된다.

```text
Composite = 0.60 × Quiz ESM
          + 0.25 × Final-state F1
          + 0.15 × Update F1
```

| UPDATE/시나리오 | Patch | Delta k=2 | Delta k=5 | Delta k=10 |
|---:|---:|---:|---:|---:|
| 20 | **64.37** | 59.47 | 54.81 | 58.28 |
| 40 | **65.37** | 63.46 | 50.28 | 58.32 |
| 60 | 63.65 | **64.09** | 51.38 | 57.90 |
| 80 | **64.16** | 62.29 | 48.18 | 57.77 |

### Composite와 평균 prefill 절감의 결합

각 Delta cell은 `Composite (Patch 대비 %p) / UPDATE 직후 prefill 절감률`이다.

| UPDATE/시나리오 | Delta k=2 | Delta k=5 | Delta k=10 |
|---:|---:|---:|---:|
| 20 | 59.47 (-4.90) / 34.4% | 54.81 (-9.56) / 34.9% | 58.28 (-6.09) / 6.2% |
| 40 | **63.46 (-1.91) / 41.0%** | 50.28 (-15.09) / 53.8% | 58.32 (-7.05) / 39.1% |
| 60 | **64.09 (+0.44) / 43.7%** | 51.38 (-12.26) / 61.5% | 57.90 (-5.75) / 54.3% |
| 80 | **62.29 (-1.88) / 45.1%** | 48.18 (-15.98) / 65.6% | 57.77 (-6.39) / 62.4% |

k2는 40–80 UPDATE에서 Patch와 Composite가 `-1.91~+0.44%p` 범위이면서 평균
prefill을 `41.0~45.1%` 줄이는 보수적인 trade-off다. k10은 Composite를
`5.75~7.05%p` 양보하지만 평균 prefill을 최대 `62.4%`, p95를 최대 `38.4%` 줄이는
공격적인 latency 후보이다. k5는 평균 token은 가장 적지만 Composite 손실이 크다.

### Composite 구성요소

| UPDATE | 방법 | Quiz ESM | Final-state F1 | Update F1 | Composite |
|---:|---|---:|---:|---:|---:|
| 20 | Patch | 65.00 | 56.09 | 75.65 | **64.37** |
| 20 | k2 | 61.00 | 44.70 | 77.95 | 59.47 |
| 20 | k5 | 51.50 | 53.54 | 70.18 | 54.81 |
| 20 | k10 | 55.50 | 52.33 | 79.31 | 58.28 |
| 40 | Patch | 61.50 | 61.86 | 86.68 | **65.37** |
| 40 | k2 | 58.50 | 60.80 | 87.76 | 63.46 |
| 40 | k5 | 44.00 | 52.67 | 71.38 | 50.28 |
| 40 | k10 | 49.00 | 63.81 | 86.43 | 58.32 |
| 60 | Patch | 59.00 | 61.13 | 86.43 | 63.65 |
| 60 | k2 | 56.50 | 66.21 | 90.91 | **64.09** |
| 60 | k5 | 43.00 | 57.06 | 75.46 | 51.38 |
| 60 | k10 | 45.00 | 69.67 | 89.89 | 57.90 |
| 80 | Patch | 58.50 | 64.29 | 86.60 | **64.16** |
| 80 | k2 | 58.00 | 59.32 | 84.37 | 62.29 |
| 80 | k5 | 39.50 | 55.38 | 70.91 | 48.18 |
| 80 | k10 | 43.00 | **71.84** | **93.40** | 57.77 |

## Controlled replay 진단

이 실행은 controlled replay이므로 아래 수치는 모델 출력 진단용이다. 다음 턴 state는
항상 Gold로 복구되며, Quiz ESM이나 predicted closed-loop accuracy를 대신하지 않는다.

| 80 UPDATE | Decision accuracy | UPDATE recall | Synthetic ADD decision recall | Apply errors |
|---|---:|---:|---:|---:|
| Patch | 93.2% | 90.5% | 95.2% | 7/745 |
| Delta k=2 | **95.2%** | **94.0%** | **100.0%** | 13/745 |
| Delta k=5 | 83.9% | 71.3% | 80.1% | 14/745 |
| Delta k=10 | 93.8% | 89.3% | 98.5% | **6/745** |

k5 checkpoint는 controlled replay에서도 UPDATE recall이 낮았고, 이는 위 predicted
closed-loop Composite 저하와 같은 방향이다. 다만 최종 성능 판단은 이 진단표가 아니라
정식 Composite를 사용한다.

## HF 시간과 메모리

80 UPDATE의 UPDATE 직후 평균 HF prefill은 Patch `538ms`, k2 `523ms`, k5 `422ms`,
k10 `402ms`였다. A100에서는 작은 fragmented prefill과 cache 관리 overhead 때문에 token
절감률만큼 시간이 줄지 않는다.

80 UPDATE 전체 턴의 평균 KV cache는 Patch `324.7MiB`, k2 `331.3MiB`, k5
`333.2MiB`, k10 `346.7MiB`; p95 peak CUDA allocation은 각각 `2.34/2.30/2.17/2.21GiB`였다.
k10의 긴 pending prefix가 KV 메모리를 조금 더 사용하지만 절대 증가는 작았다.

## 판단 및 다음 단계

- Canary는 성공했으며 stress가 커질수록 Delta-v3의 Patch 대비 평균 prefill 이점이
  커지는 가설을 지지한다.
- 성능 보존을 우선하면 k2, 평균·tail latency를 더 공격적으로 줄이면 k10이 유력하다.
  k5는 최대 평균 절감은 보이지만 현재 checkpoint의 Composite 손실이 너무 크다.
- 다음에는 latency 조건을 3회 이상 반복해 GPU wall-clock 분산을 확인한 뒤, 규칙을
  바꾸지 않고 S86-S90 Test에서 cache ON/OFF와 predicted closed loop/Quiz를 실행한다.
- background prefill은 별도 ablation으로 foreground latency와 total compute를 나눠
  보고한다.

원시 결과:
`/mnt/data/hj153lee/PalmClaw/on-device-memory-training/benchmarks/kv-cache-qwen35-0.8b-stress-validation-once-20260904-v1`

Composite 원시 결과:
`/mnt/data/hj153lee/PalmClaw/on-device-memory-training/benchmarks/qwen35-0.8b-stress-validation-composite-once-20260904-v1`

## Held-out Stress Test 결과

Validation에서 고정한 규칙을 바꾸지 않고 Test `S86-S90`에 적용했다. 각 UPDATE 규모는
Gold UPDATE 다음 요청 `100/200/300/400`건을 포함하며, cache latency와 predicted
closed-loop Composite 모두 `16/16` job이 완료됐다.

### UPDATE 직후 evaluated prefill tokens

각 Delta 괄호는 Patch 대비 평균 token 절감률이다.

| UPDATE/시나리오 | Patch | Delta k=2 | Delta k=5 | Delta k=10 |
|---:|---:|---:|---:|---:|
| 20 | 546.3 | 359.0 (-34.3%) | **352.8 (-35.4%)** | 513.8 (-5.9%) |
| 40 | 940.8 | 555.5 (-41.0%) | **436.9 (-53.6%)** | 575.7 (-38.8%) |
| 60 | 1,338.9 | 755.6 (-43.6%) | **514.8 (-61.5%)** | 610.7 (-54.4%) |
| 80 | 1,737.6 | 954.7 (-45.1%) | **596.0 (-65.7%)** | 654.2 (-62.3%) |

UPDATE 직후 p95는 80 UPDATE에서 Patch `3,102.1`, k2 `2,928.1`(-5.6%), k5
`2,500.1`(-19.4%), k10 `1,884.1`(-39.3%)이다. 30 prefill tok/s 기기의 평균 단순
환산값은 각각 `57.9s`, `31.8s`, `19.9s`, `21.8s`다.

### Predicted closed-loop Composite

각 Delta 괄호는 같은 UPDATE 규모의 Patch 대비 차이(%p)다.

| UPDATE/시나리오 | Patch | Delta k=2 | Delta k=5 | Delta k=10 |
|---:|---:|---:|---:|---:|
| 20 | 57.29 | **61.11 (+3.81)** | 53.17 (-4.12) | 57.00 (-0.30) |
| 40 | **64.71** | 62.36 (-2.35) | 54.89 (-9.82) | 59.36 (-5.36) |
| 60 | **66.54** | 58.25 (-8.29) | 49.81 (-16.72) | 61.96 (-4.57) |
| 80 | 63.20 | **66.94 (+3.74)** | 48.52 (-14.68) | 58.13 (-5.07) |

Test에서도 latency 방향은 Validation과 재현됐다. `k=2`는 평균 prefill을
`34.3-45.1%` 줄이면서 20·80 UPDATE cell에서는 Patch보다 높은 Composite를 보였다.
`k=10`은 40 UPDATE 이상에서 p95 절감이 가장 크고, `k=5`는 평균 token 절감이 가장
크지만 Composite 손실도 가장 크다. 단, 이 Stress Test Composite는 5개 시나리오와
합성 UPDATE를 사용하므로 원래 24개 시나리오의 표준 Test Composite와 직접 비교하지
않는다.

Test 원시 결과:
`/mnt/data/hj153lee/PalmClaw/on-device-memory-training/benchmarks/qwen35-0.8b-stress-test-once-20260904-v1`
