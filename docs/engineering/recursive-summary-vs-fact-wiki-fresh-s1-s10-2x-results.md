# Recursive Summary vs Fact-Wiki Fresh-cache Scenario 1–10 결과

상태: `완료 · 두 방법 각 2회 독립 E2E 평가`

평가일: 2026-08-10

## 조건

- VehicleMemBench Scenario 1–10, 방법·회차별 100 task
- Agent `gpt-5.6-terra`, Memory `gpt-5.6-luna`
- Embedding `text-embedding-3-small`, 256 dimensions
- 매 방법·회차마다 빈 cache에서 Memory를 새로 생성
- 두 회차 사이 같은 Memory fingerprint: 두 방법 모두 `0/10`
- Gold Memory 및 Gold Tool/argument 미사용
- 총 유효 평가량: `2 methods × 2 runs × 100 tasks = 400 task-runs`
- 표의 `±`는 두 run의 sample standard deviation이다. `n=2`이므로 추세 확인용이며
  통계적 유의성을 주장하지 않는다.

유효 artifact:

- [Recursive run 1](../../ubuntu/evaluation/vehiclemembench/fresh-recursive-fact-wiki-s1-s10-2x-20260810/results/repetition-1/recursive-summary/2a41ef62-48cf-4075-90ae-cd136c5c8828/results.md)
- [Fact-Wiki run 1](../../ubuntu/evaluation/vehiclemembench/fresh-recursive-fact-wiki-s1-s10-2x-20260810/results/repetition-1/fact-wiki/3f8065b6-611d-406d-9284-f6ff58e540c4/results.md)
- [Recursive run 2](../../ubuntu/evaluation/vehiclemembench/fresh-recursive-fact-wiki-s1-s10-2x-20260810/repetition-2-retry-1/results/recursive-summary/2846d8cb-a0d7-40fc-9fe2-0d5efc133a90/results.md)
- [Fact-Wiki run 2](../../ubuntu/evaluation/vehiclemembench/fresh-recursive-fact-wiki-s1-s10-2x-20260810/repetition-2-retry-1/results/fact-wiki/b97dc01a-212b-4b03-bcf1-ef4f2069d142/results.md)

중복 resume 과정에서 중단된 Recursive run
`be8dd878-9701-494b-a05d-9f71852df597`은 집계에서 제외하고, 두 번째 회차를 새
cache로 처음부터 다시 실행했다. 네 유효 run은 모두 100/100 task 완료, failed task
0건, artifact privacy audit 통과다.

## 전체 정확도

| 방법 | Run | ESM | State F1 | Tool F1 | Arg exact | Recall@k |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Recursive Summary | 1 | 0.670 | 0.870 | 0.719 | 0.620 | N/A |
| Recursive Summary | 2 | 0.640 | 0.848 | 0.702 | 0.580 | N/A |
| **Recursive Summary** | **평균** | **0.655 ± 0.021** | **0.859 ± 0.016** | **0.710 ± 0.012** | **0.600 ± 0.028** | N/A |
| Fact-Wiki | 1 | 0.620 | 0.756 | 0.630 | 0.510 | 0.575 |
| Fact-Wiki | 2 | 0.600 | 0.754 | 0.590 | 0.470 | 0.572 |
| **Fact-Wiki** | **평균** | **0.610 ± 0.014** | **0.755 ± 0.001** | **0.610 ± 0.028** | **0.490 ± 0.028** | **0.573 ± 0.002** |

Fact-Wiki의 평균 차이는 Recursive Summary 대비 ESM `-0.045`, State F1
`-0.104`, Tool F1 `-0.101`, Arg exact `-0.110`이다. 두 run 모두 ESM 방향은
Recursive Summary 우위였다.

## 시나리오별 ESM

| Scenario | Recursive R1 | Recursive R2 | Recursive 평균 | Fact-Wiki R1 | Fact-Wiki R2 | Fact-Wiki 평균 | Fact − Recursive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.70 | 0.80 | **0.75** | 0.80 | 0.60 | 0.70 | -0.05 |
| 2 | 0.80 | 0.40 | **0.60** | 0.60 | 0.40 | 0.50 | -0.10 |
| 3 | 0.70 | 0.60 | 0.65 | 0.70 | 0.60 | 0.65 | 0.00 |
| 4 | 0.70 | 0.70 | 0.70 | 0.70 | 0.70 | 0.70 | 0.00 |
| 5 | 0.40 | 0.20 | 0.30 | 0.40 | 0.40 | **0.40** | +0.10 |
| 6 | 0.70 | 0.80 | **0.75** | 0.50 | 0.60 | 0.55 | -0.20 |
| 7 | 0.90 | 0.90 | **0.90** | 0.50 | 0.60 | 0.55 | -0.35 |
| 8 | 0.80 | 1.00 | **0.90** | 0.80 | 0.90 | 0.85 | -0.05 |
| 9 | 0.60 | 0.50 | **0.55** | 0.40 | 0.40 | 0.40 | -0.15 |
| 10 | 0.40 | 0.50 | 0.45 | 0.80 | 0.80 | **0.80** | +0.35 |

Scenario 1–5 평균은 Recursive `0.60`, Fact-Wiki `0.59`로 거의 같았다.
Scenario 6–10 평균은 Recursive `0.71`, Fact-Wiki `0.63`으로 차이가 커졌다.
Fact-Wiki는 Scenario 5와 10에서 앞섰고, 3과 4에서 동률, 나머지 6개
Scenario에서 뒤졌다.

| Scenario | Recursive State F1 | Fact State F1 | Recursive Tool F1 | Fact Tool F1 | Recursive Arg | Fact Arg |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.865 | 0.773 | 0.820 | 0.667 | 0.70 | 0.60 |
| 2 | 0.825 | 0.708 | 0.733 | 0.567 | 0.60 | 0.50 |
| 3 | 0.837 | 0.806 | 0.758 | 0.723 | 0.60 | 0.55 |
| 4 | 0.933 | 0.867 | 0.700 | 0.667 | 0.70 | 0.60 |
| 5 | 0.849 | 0.715 | 0.458 | 0.433 | 0.30 | 0.30 |
| 6 | 0.936 | 0.676 | 0.757 | 0.540 | 0.65 | 0.40 |
| 7 | 0.953 | 0.700 | 0.937 | 0.533 | 0.85 | 0.50 |
| 8 | 0.950 | 0.850 | 0.798 | 0.750 | 0.75 | 0.65 |
| 9 | 0.776 | 0.556 | 0.633 | 0.500 | 0.55 | 0.35 |
| 10 | 0.664 | 0.900 | 0.508 | 0.717 | 0.30 | 0.45 |

## Fresh-cache 변동과 task 교차

- 같은 방법의 두 run 사이 ESM이 바뀐 task는 Recursive `21/100`, Fact-Wiki
  `20/100`이었다. Memory와 Agent를 모두 다시 실행하면 약 20%의 task churn이 있다.
- 두 run을 합친 200 paired task-run에서 Recursive만 성공 `35`, Fact-Wiki만 성공
  `26`, 둘 다 성공 `96`, 둘 다 실패 `43`이었다.
- 따라서 Fact-Wiki가 Recursive의 실패를 복구하는 사례는 존재하지만, 현재 gate는
  반대 방향의 회귀를 더 많이 만든다.
- 이전 Scenario 6–10 단일 Fact-Wiki ESM `0.70`은 이번 fresh-cache 반복에서
  재현되지 않았다. 이번 S6–10은 Fact-Wiki `0.60/0.66`(평균 `0.63`), Recursive
  `0.68/0.74`(평균 `0.71`)였다.

## 토큰·비용·지연

아래 값은 100-task run 하나의 두 회차 평균이다. 지연은 wall-clock이 아니라 provider
호출 누적 latency다. 비용은 기존 비교와 같은 Agent input/output `$2/$8`, Memory
input/output `$1/$6`, embedding input `$0.02` per 1M token의 uncached 추정치다.

| 방법 | Memory LLM tokens | Embedding tokens | Quiz Agent tokens | 전체 tokens | 추정 비용 | Idle Memory latency | Online retrieval | Quiz latency | 누적 latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Recursive Summary** | **1.282M** | 0 | **0.434M** | **1.716M** | **$2.50** | **16.8분** | 0분 | **10.9분** | **27.8분** |
| Fact-Wiki | 8.002M | 0.036M | 0.739M | 8.777M | $10.62 | 86.1분 | 1.2분 | 12.2분 | 99.5분 |

Fact-Wiki는 Recursive Summary 대비 Memory LLM token `6.24×`, Quiz Agent token
`1.70×`, 전체 token `5.11×`, 추정 비용 `4.24×`, 누적 latency `3.58×`였다.
embedding 자체는 평균 0.036M으로 병목이 아니며, 주된 비용은 Post-normalized Fact를
만드는 Memory LLM 호출이다.

## 판정

1. 현재 Fact-Wiki는 Scenario 1–10 전체 기준으로 Recursive Summary를 정확도와
   비용 어느 쪽에서도 지배하지 못하므로 baseline 대체 후보로 채택하지 않는다.
2. Scenario 10의 반복 가능한 `+0.35`와 Fact-only 성공 26건은 linked traversal이
   특정 질의에는 유효하다는 신호다.
3. 후속 연구는 전체 질의에 Fact-Wiki를 노출하기보다 Scenario 10형 질의를 사전에
   식별하는 deterministic gate 또는 Recursive 기본 경로 위의 선택적 traversal로
   제한하는 편이 타당하다.
4. 다음 성능 개선 판단에서는 단일 run 최고값보다 fresh-cache 반복 평균과 workflow
   비용을 함께 사용한다.
