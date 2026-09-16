# Main Performance: 4 Models × 3 Methods

전체 Test 24개 시나리오(`S86–S100`, `S112–S120`) 결과다. Qwen 0.8B와 Qwen 2B의
Patch는 seed46, 나머지는 seed45를 사용한다. Delta-v3는 latency–accuracy 절충의 대표
설정으로 compact `k=5`를 사용한다. 모든 값은 `%`이며 높을수록 좋다.

```text
Composite = 0.60 × Quiz ESM
          + 0.25 × Final-state F1
          + 0.15 × Update F1
```

| Model | Method | Composite ↑ | Quiz ESM ↑ | Final-state F1 ↑ | Update F1 ↑ |
|---|---|---:|---:|---:|---:|
| Granite 350M | Patch | **59.62** | **60.73** | 43.92 | **81.38** |
|  | Summary | 55.28 | 55.62 | 42.45 | 75.26 |
|  | Delta-v3 (k=5) | 59.01 | 60.21 | **44.92** | 77.68 |
| Qwen 0.8B | Patch (seed46) | **70.39** | **74.17** | **52.93** | **84.40** |
|  | Summary | 64.38 | 67.29 | 46.51 | 82.49 |
|  | Delta-v3 (k=5) | 66.68 | 69.69 | 50.78 | 81.15 |
| Granite 1B | Patch | **71.74** | **77.92** | **50.23** | 82.86 |
|  | Summary | 69.90 | 76.25 | 46.69 | 83.16 |
|  | Delta-v3 (k=5) | 67.38 | 70.73 | 49.22 | **84.22** |
| Qwen 2B | Patch (seed46) | **71.13** | **75.83** | 52.28 | 83.75 |
|  | Summary | 65.61 | 69.69 | 47.51 | 79.47 |
|  | Delta-v3 (k=5) | 69.10 | 71.67 | **54.00** | **84.01** |

굵은 값은 각 모델의 세 방법 중 열별 최고값이다. 이 표는 main accuracy 결과만 다루며,
Stress latency와 Cache ON/OFF 결과는 별도 표와 figure로 보고한다.

## 보고 시 주의

- `F1 score`처럼 모호하게 쓰지 않고 `Final-state F1`, `Update F1`을 구분한다.
- Delta-v3의 `k=5` 선택은 표 제목이나 캡션에 반드시 명시한다.
- Composite만 굵게 강조하고 세부 metric의 최고값은 밑줄로 표시하는 방식도 가능하지만,
  현재 표는 모든 열의 최고값을 굵게 표시했다.
- 모델 크기뿐 아니라 Granite/Qwen 모델 family도 함께 바뀌므로 순수한 scaling curve로
  해석하지 않는다.
