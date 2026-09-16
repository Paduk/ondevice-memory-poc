# GPT-5.6 Luna Reasoning-Effort Comparison

## 설정

- 모델: `gpt-5.6-luna`
- 방법론: Patch closed-loop
- Memory writer와 Quiz reader 모두 동일 reasoning effort 사용
- 비교: 기존 `low` 대 신규 `none`
- V2 Test: 24 scenarios, 960 quizzes
- HST: 20 scenarios, 320 quizzes
- HCT: 20 scenarios, 574 quizzes
- Composite: `0.60 × Quiz ESM + 0.25 × Final-memory F1 + 0.15 × Update F1`
- Update F1은 전체 turn을 합친 micro 집계

Luna의 자유 형식 memory가 deterministic line scorer와 일치하지 않아 두 조건 모두
Final-memory F1이 0으로 측정된다. 따라서 Composite는 참고값이고 Quiz ESM을 주요
성능 지표로 사용한다.

## 주요 결과

| Dataset | Composite low → none (Δ) | Quiz ESM low → none (Δ) | Update F1 low → none (Δ) | Cost low → none |
|---|---:|---:|---:|---:|
| V2 Test | 49.29 → 48.07 (−1.22) | 73.33 → 69.38 (−3.96) | 35.28 → 42.95 (+7.67) | $0.952 → $0.592 (−37.8%) |
| HST | 62.63 → 60.78 (−1.85) | 87.81 → 86.88 (−0.94) | 66.30 → 57.69 (−8.61) | $0.129 → $0.099 (−23.8%) |
| HCT | 41.06 → 38.60 (−2.46) | 54.53 → 50.00 (−4.53) | 55.59 → 57.31 (+1.72) | $0.350 → $0.296 (−15.4%) |

세 dataset 총 API 비용은 `$1.431 → $0.986`, 약 31.1% 감소했다.

## Quiz 세부 결과

| Dataset | Effort | Turn ESM | Final ESM | Tool F1 | Argument Exact | Invalid memory outputs |
|---|---|---:|---:|---:|---:|---:|
| V2 Test | low | 72.64 | 75.42 | 74.81 | 71.25 | 18 |
|  | none | 70.83 | 65.00 | 71.30 | 66.25 | 31 |
| HST | low | 87.50 | 88.13 | 87.81 | 87.81 | 0 |
|  | none | 85.63 | 88.13 | 86.88 | 86.88 | 3 |
| HCT | low | 53.26 | 58.33 | 60.45 | 51.74 | 3 |
|  | none | 49.77 | 50.69 | 56.12 | 46.17 | 11 |

## 해석

1. `none`은 세 dataset 모두에서 성능이 하락했지만 비용은 총 31.1% 절감했다.
2. HST의 ESM 하락은 0.94%p로 작아, 쉬운 명시적 문제에서는 `none`도 비용 효율적인
   선택이 될 수 있다.
3. V2 Test와 HCT에서는 Final ESM이 각각 10.42%p, 7.64%p 하락했다. 누적 memory를
   종합해 최종 행동을 고르는 문제에서 reasoning 감소의 영향이 더 크게 나타났다.
4. V2 Test와 HCT는 Update F1이 오히려 상승했는데도 Quiz ESM이 하락했다. 따라서
   성능 저하는 UPDATE/NO_OP 판정만의 문제가 아니라 memory 내용 구성과 reader의
   최종 tool 선택에도 걸쳐 있다.
5. Invalid memory output이 `18→31`, `0→3`, `3→11`로 모두 증가했다. `none`은 출력
   형식과 memory contract 준수 안정성도 낮췄다.

논문에서는 `low`를 주 Cloud reference로 유지하고, `none`은 비용-성능 ablation으로
제시하는 편이 적절하다.

## 결과 위치

- 신규 `none`: `/mnt/data/hj153lee/PalmClaw/on-device-memory-training/evaluations/luna-reasoning-none-v2-hst-hct-v1`
- V2 Test aggregate: `v2_test/patch/aggregate.json`
- HST aggregate: `hst/patch/aggregate.json`
- HCT aggregate: `hct/patch/aggregate.json`
