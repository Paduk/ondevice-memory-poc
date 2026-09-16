# Summary·Delta-v3·Patch 유형별 Test 정확도

- Test: S86–S100, S112–S120, 모델·방법별 동일 960 quiz
- 지표: ESM (`exact_state_match`), 단위 `%`
- Delta-v3: compact `k=5`
- 유형별 표본 수: conditional constraint 164, coreference resolution 195,
  error correction 145, preference conflict 235, state shift 221
- Quiz scope 표본 수: TURN 720, FINAL 240

## Granite 350M

| Reasoning type | N | Summary | Delta-v3 | Patch |
|---|---:|---:|---:|---:|
| Conditional constraint | 164 | **67.07** | 60.37 | 66.46 |
| Coreference resolution | 195 | 52.31 | **69.23** | 54.36 |
| Error correction | 145 | 49.66 | 51.72 | **55.17** |
| Preference conflict | 235 | 54.04 | 55.74 | **61.70** |
| State shift | 221 | 55.66 | 62.44 | **64.71** |
| **Overall** | **960** | **55.62** | **60.21** | **60.73** |

| Quiz scope | N | Summary | Delta-v3 | Patch |
|---|---:|---:|---:|---:|
| TURN | 720 | 55.97 | 60.28 | **60.69** |
| FINAL | 240 | 54.58 | 60.00 | **60.83** |

## Qwen 0.8B

| Reasoning type | N | Summary | Delta-v3 | Patch |
|---|---:|---:|---:|---:|
| Conditional constraint | 164 | 73.78 | 73.17 | **82.32** |
| Coreference resolution | 195 | **72.31** | 69.74 | 71.28 |
| Error correction | 145 | 55.86 | **71.03** | 68.97 |
| Preference conflict | 235 | 63.40 | 67.23 | **74.89** |
| State shift | 221 | 69.68 | 68.78 | **73.30** |
| **Overall** | **960** | **67.29** | **69.69** | **74.17** |

| Quiz scope | N | Summary | Delta-v3 | Patch |
|---|---:|---:|---:|---:|
| TURN | 720 | 66.94 | 68.89 | **73.19** |
| FINAL | 240 | 68.33 | 72.08 | **77.08** |

## Granite 1B

| Reasoning type | N | Summary | Delta-v3 | Patch |
|---|---:|---:|---:|---:|
| Conditional constraint | 164 | 87.20 | 78.05 | **87.80** |
| Coreference resolution | 195 | 72.82 | 73.33 | **74.36** |
| Error correction | 145 | 76.55 | 69.66 | **79.31** |
| Preference conflict | 235 | 70.21 | 66.38 | **72.77** |
| State shift | 221 | 77.38 | 68.33 | **78.28** |
| **Overall** | **960** | **76.25** | **70.73** | **77.92** |

| Quiz scope | N | Summary | Delta-v3 | Patch |
|---|---:|---:|---:|---:|
| TURN | 720 | 75.69 | 70.42 | **76.53** |
| FINAL | 240 | 77.92 | 71.67 | **82.08** |

## Llama 3.2 1B

| Reasoning type | N | Summary | Delta-v3 | Patch |
|---|---:|---:|---:|---:|
| Conditional constraint | 164 | **79.88** | 69.51 | **79.88** |
| Coreference resolution | 195 | 58.97 | **72.82** | 69.23 |
| Error correction | 145 | 50.34 | 55.17 | **71.72** |
| Preference conflict | 235 | 58.30 | 65.53 | **69.79** |
| State shift | 221 | 57.92 | 70.59 | **71.04** |
| **Overall** | **960** | **60.83** | **67.29** | **71.98** |

| Quiz scope | N | Summary | Delta-v3 | Patch |
|---|---:|---:|---:|---:|
| TURN | 720 | 61.11 | 67.08 | **71.81** |
| FINAL | 240 | 60.00 | 67.92 | **72.50** |

## Qwen 2B

| Reasoning type | N | Summary | Delta-v3 | Patch |
|---|---:|---:|---:|---:|
| Conditional constraint | 164 | 80.49 | 76.83 | **82.32** |
| Coreference resolution | 195 | 69.74 | 70.26 | **74.36** |
| Error correction | 145 | 67.59 | 69.66 | **75.17** |
| Preference conflict | 235 | 59.57 | 69.79 | **71.49** |
| State shift | 221 | 73.76 | 72.40 | **77.38** |
| **Overall** | **960** | **69.69** | **71.67** | **75.83** |

| Quiz scope | N | Summary | Delta-v3 | Patch |
|---|---:|---:|---:|---:|
| TURN | 720 | 68.89 | 69.31 | **75.83** |
| FINAL | 240 | 72.08 | **78.75** | 75.83 |

## Llama 3.2 3B

| Reasoning type | N | Summary | Delta-v3 | Patch |
|---|---:|---:|---:|---:|
| Conditional constraint | 164 | 78.66 | 76.22 | **84.15** |
| Coreference resolution | 195 | **75.90** | 74.87 | 74.36 |
| Error correction | 145 | 57.93 | 62.07 | **67.59** |
| Preference conflict | 235 | 62.13 | **75.32** | 74.47 |
| State shift | 221 | **73.76** | 72.40 | 71.04 |
| **Overall** | **960** | **69.79** | **72.71** | **74.27** |

| Quiz scope | N | Summary | Delta-v3 | Patch |
|---|---:|---:|---:|---:|
| TURN | 720 | 70.00 | 71.81 | **73.47** |
| FINAL | 240 | 69.17 | 75.42 | **76.67** |

## 1차 관찰

- Patch는 모델별 reasoning-type 30개 비교 중 21개에서 단독 1위, 1개에서 Summary와
  공동 1위다. TURN/FINAL 12개 비교에서는 11개에서 1위다.
- Patch의 우위는 error correction과 preference conflict에서 가장 일관적이다. 이는
  기존 fact의 교정·충돌 해결이라는 Patch의 설계 목적과 부합한다.
- Delta-v3는 Granite 350M·Llama 1B의 coreference resolution, Qwen 0.8B의 error
  correction, Llama 3B의 preference conflict에서 Patch보다 높다.
- 유일한 scope-level 역전은 Qwen 2B FINAL이며, Delta-v3가 `78.75`, Patch가 `75.83`이다.
- Granite 1B에서는 Summary가 Delta-v3보다 모든 reasoning type에서 높다. Delta-v3의
  효율 이점과 정확도 trade-off가 모델에 따라 달라진다는 신호다.

## Patch vs Delta-v3: 큰 차이 중심 분석

아래 값은 `Patch ESM − Delta-v3 ESM`이다. `±2%p` 이내는 실질적 동률로 보고,
`2–4%p`는 약한 방향성, `4%p` 이상을 우선 해석한다.

| Model | Overall | TURN | FINAL | 명확한 유형별 차이 (`≥4%p`) |
|---|---:|---:|---:|---|
| Granite 350M | +0.52 | +0.42 | +0.83 | Delta: Coref `+14.87`; Patch: Cond `+6.10`, Conflict `+5.96` |
| Qwen 0.8B | **+4.48** | **+4.31** | **+5.00** | Patch: Cond `+9.15`, Conflict `+7.66`, Shift `+4.52` |
| Granite 1B | **+7.19** | **+6.11** | **+10.42** | Patch: Cond `+9.76`, Error `+9.66`, Conflict `+6.38`, Shift `+9.95` |
| Llama 3.2 1B | **+4.69** | **+4.72** | **+4.58** | Patch: Cond `+10.37`, Error `+16.55`, Conflict `+4.26` |
| Qwen 2B | **+4.17** | **+6.53** | −2.92 | Patch: Cond `+5.49`, Coref `+4.10`, Error `+5.52`, Shift `+4.98` |
| Llama 3.2 3B | +1.56 | +1.67 | +1.25 | Patch: Cond `+7.93`, Error `+5.52` |

- 가장 강한 일반 패턴은 **conditional constraint**다. Patch가 6/6 모델에서
  `+5.49–10.37%p` 높다.
- **Error correction**은 Patch가 5/6 모델에서 높고, 그중 Granite 1B·Llama 1B는
  `+9.66`, `+16.55%p`로 크다. Qwen 0.8B의 Delta 우위는 `2.07%p`로 경계 수준이다.
- **Coreference resolution**은 방향이 모델마다 바뀐다. 특히 Granite 350M에서는
  Delta-v3가 `14.87%p` 높지만, Qwen 2B에서는 Patch가 `4.10%p` 높다. 공통 결론으로
  일반화하기 어렵다.
- Overall 기준 Qwen 0.8B·Granite 1B·Llama 1B·Qwen 2B는 Patch의 `+4.17–7.19%p`
  우위가 명확하다. Granite 350M·Llama 3B는 `+0.52`, `+1.56%p`이므로 정확도는
  사실상 동률이며 latency가 낮은 Delta-v3가 합리적인 선택이다.
- Qwen 2B는 TURN에서 Patch가 `+6.53%p`, FINAL에서는 Delta-v3가 `+2.92%p`다.
  즉 실시간 turn-wise 정확도를 우선하면 Patch, 최종 snapshot 질의를 더 중시하면
  Delta-v3도 경쟁력이 있다.

동일 문항 정오를 이용한 exact McNemar 검정에서 Overall Patch 우위는 Qwen 0.8B,
Granite 1B, Llama 1B, Qwen 2B에서 `p<0.01`이며, Granite 350M과 Llama 3B는 유의하지
않다. 유형별 검정은 다중 비교가 많으므로 개별 `p`값보다 큰 효과크기와 모델 간 방향
일관성을 우선 해석한다.

## 재현 artifact

- 집계 JSON:
  `/mnt/data/hj153lee/PalmClaw/on-device-memory-training/evaluations/three-method-type-accuracy.json`
- 집계 스크립트:
  `memory_training/scripts/aggregate_three_method_type_accuracy.py`
