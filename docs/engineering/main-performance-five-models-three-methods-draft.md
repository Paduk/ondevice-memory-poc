# Main Test Composite: 6 Models × 3 Methods (Draft)

전체 Test 24개 시나리오(`S86–S100`, `S112–S120`)의 Composite 결과 초안이다.
Delta-v3는 latency–accuracy trade-off의 대표 설정인 compact `k=5`를 사용한다. 성능 값은
`%`이며 높을수록 좋다. Latency는 모델이 실제로 적용한 UPDATE와 그 직후 턴을 하나의
UPDATE cycle로 묶어 추정한 값이며 낮을수록 좋다.

```text
Composite = 0.60 × Quiz ESM
          + 0.25 × Final-state F1
          + 0.15 × Update F1
```

| Model | Method | Composite ↑ | Quiz ESM ↑ | Final-state F1 ↑ | Update F1 ↑ | Projected UPDATE-cycle latency (s) ↓ | Relative latency (Summary=100) ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| Granite 350M | Summary | 55.28 | 55.62 | 42.45 | 75.26 | 6.54 | 100.0 |
|  | Patch | **59.62** | **60.73** | 43.92 | **81.38** | 2.96 | 45.3 |
|  | Delta-v3 (k=5) | 59.01 | 60.21 | **44.92** | 77.68 | **2.67** | **40.9** |
| Qwen 0.8B | Summary | 64.38 | 67.29 | 46.51 | 82.49 | 15.28 | 100.0 |
|  | Patch | **70.39** | **74.17** | **52.93** | **84.40** | 8.38 | 54.9 |
|  | Delta-v3 (k=5) | 66.68 | 69.69 | 50.78 | 81.15 | **7.62** | **49.9** |
| Granite 1B | Summary | 69.90 | 76.25 | 46.69 | 83.16 | 23.36 | 100.0 |
|  | Patch | **71.74** | **77.92** | **50.23** | 82.86 | 12.47 | 53.4 |
|  | Delta-v3 (k=5) | 67.38 | 70.73 | 49.22 | **84.22** | **11.08** | **47.4** |
| Llama 3.2 1B | Summary | 60.30 | 60.83 | 47.74 | 79.11 | 18.00 | 100.0 |
|  | Patch | **67.06** | **71.98** | 46.49 | 81.64 | 9.31 | 51.7 |
|  | Delta-v3 (k=5) | 65.91 | 67.29 | **51.75** | **83.96** | **8.55** | **47.5** |
| Llama 3.2 3B | Summary | 66.55 | 69.79 | 49.59 | 81.84 | 47.63 | 100.0 |
|  | Patch | **71.06** | **74.27** | **54.97** | **85.05** | 28.95 | 60.8 |
|  | Delta-v3 (k=5) | 69.22 | 72.71 | 53.31 | 81.78 | **24.82** | **52.1** |
| Qwen 2B | Summary | 65.61 | 69.69 | 47.51 | 79.47 | 36.38 | 100.0 |
|  | Patch | **71.13** | **75.83** | 52.28 | 83.75 | 20.46 | 56.3 |
|  | Delta-v3 (k=5) | 69.10 | 71.67 | **54.00** | **84.01** | **17.32** | **47.6** |

굵은 값은 각 모델·지표에서 현재 확인된 최고값이다.

## Draft notes

- Qwen 0.8B와 Qwen 2B의 Patch는 기존 메인 테이블 규칙에 따라 train seed 46 결과를
  사용했다. 나머지 값은 train seed 45다.
- Llama 3.2 1B의 Summary/Patch/Delta-v3 `k=5`는 각각 선택된 epoch 4/3/4의
  `complete=true`, 24/24 Test 결과다.
- Llama 3.2 3B의 Summary/Patch/Delta-v3 `k=5`는 각각 선택된 epoch 3/4/4의
  `complete=true`, 24/24 Test 결과다.
- 최종 표에서는 seed 단일값, seed 평균, 또는 seed별 행 중 한 가지 보고 규칙으로
  통일해야 한다.
- UPDATE-cycle latency는 Cache-ON 로그에서 `applied_update=true`인 턴의 decode token과
  같은 시나리오의 바로 다음 턴에서 실제 재계산한 prefill token을 사용했다.

  ```text
  UPDATE-cycle latency
    = UPDATE decode tokens / decode tokens-per-second
    + next-turn evaluated prefill tokens / prefill tokens-per-second
  ```

- 처리량은 Galaxy Z Fold7 Q4_K_M 측정치인 Granite 350M `137.7/51.8`, Granite 1B
  `25.7/16.4`, Llama 1B `35.7/19.8`, Llama 3B `11.1/7.3` prefill/decode tok/s를
  사용했다. 정확히 같은 모델이 없는 Qwen 0.8B와 Qwen 2B에는 각각 Qwen3 0.6B
  `68.0/27.6`, Qwen3 1.7B `24.7/14.5`를 proxy로 사용했다. 따라서 실측 latency가
  아니라 projected latency다.
- 실제 UPDATE-cycle 표본 수 `(Summary/Patch/Delta-v3)`는 Granite 350M `33/48/38`,
  Qwen 0.8B `39/64/32`, Granite 1B `59/59/41`, Llama 1B `34/55/35`, Llama 3B
  `27/45/33`, Qwen 2B `28/53/42`다. 방법별 UPDATE 선택이 달라 조건부 평균의 대상
  턴도 서로 다르다.
- 성능은 전체 Test 24개 시나리오 결과지만 latency는 현재 완료된 natural base
  `S86–S90` Cache-ON 로그의 평균이다. 논문 최종본에서는 전체 Test latency를 추가로
  측정하거나, 이 범위 차이를 표 caption에 명시해야 한다.

## Llama 3.2 3B Delta-v3 k ablation

| Method | Selected epoch | Test Composite ↑ | Quiz ESM ↑ | Final-state F1 ↑ | Update F1 ↑ | Projected UPDATE-cycle latency (s) ↓ | Relative latency (Summary=100) ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Summary | 3 | 66.55 | 69.79 | 49.59 | 81.84 | 47.63 | 100.0 |
| Patch | 4 | **71.06** | 74.27 | **54.97** | **85.05** | 28.95 | 60.8 |
| Delta-v3 (k=2) | 3 | 70.60 | **74.90** | 51.80 | 84.72 | 27.80 | 58.4 |
| Delta-v3 (k=5) | 4 | 69.22 | 72.71 | 53.31 | 81.78 | 24.82 | 52.1 |
| Delta-v3 (k=10) | 4 | 67.19 | 70.83 | 50.16 | 80.99 | **24.36** | **51.1** |

3B에서는 `k=2`가 Patch에 근접한 Composite를 유지하고, `k=5/10`은 더 낮은
UPDATE-cycle latency를 보인다. 즉 `k` 증가에 따라 대체로 latency는 감소하지만 Test
Composite도 함께 낮아지는 명확한 trade-off가 나타난다.
