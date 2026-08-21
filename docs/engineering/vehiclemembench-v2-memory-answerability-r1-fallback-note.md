# VehicleMemBench V2 메모리 Answerability 및 R1 대체 현황

상태: `R2 S1–S50 감사 완료 · R1 fallback 감사 완료 · Luna R3 재생성 진행 중`

## 1. 감사 기준

- 대상: Turn-wise Summary R2와 Combined Patch R2의 시나리오별 최종 메모리
- 범위: S1–S50, 총 500 Quiz/방법
- 판정 모델: `gpt-5.6-sol`, 시나리오당 10개 Quiz를 한 호출로 감사
- 판정:
  - `SUPPORTED`: 최종 메모리와 Query만으로 gold tool call을 도출할 수 있음
  - `AMBIGUOUS`: 관련 정보는 있으나 조건·시간·사용자·충돌 해소가 불충분
  - `MISSING`: 필수 값이나 규칙이 없어 gold answer가 메모리에서 entail되지 않음

최종 메모리 추출과 집계는 결정론적이지만 의미 판정은 SOL 1회 결과이므로
완전한 gold label이 아니라 보수적 audit label로 취급한다.

## 2. R2 전체 결과

| 방법 | Supported | Ambiguous | Missing |
| --- | ---: | ---: | ---: |
| Summary R2 | 381 (76.2%) | 23 (4.6%) | 96 (19.2%) |
| Combined Patch R2 | 371 (74.2%) | 27 (5.4%) | 102 (20.4%) |

시나리오별 `Missing >= 30%`, 즉 10개 중 3개 이상인 경우를 R1 fallback
검사 대상으로 삼았다.

- Summary: S4, S5, S10, S11, S12, S13, S21, S24, S28, S32, S36,
  S37, S38, S39, S40
- Patch: S5, S9, S11, S12, S16, S21, S22, S24, S25, S32, S34, S35,
  S36, S38, S39, S45, S48

## 3. R1 fallback 결과

R1 최종 메모리를 동일한 방식으로 다시 감사해 `Missing < 30%`, 즉 0–2개인
경우만 R1을 채택한다. 원본 R1/R2 artifact는 덮어쓰지 않고 향후 데이터 조립 시
선택할 source repetition만 결정한다.

### R1 채택

- Summary: S4, S10, S13, S24, S28, S37, S39
- Patch: S9, S22, S25, S38, S39

R1 대체를 반영한 예상 전체 결과는 다음과 같다.

| 방법 | Supported | Ambiguous | Missing |
| --- | ---: | ---: | ---: |
| Summary hybrid | 392 (78.4%) | 23 (4.6%) | 85 (17.0%) |
| Patch hybrid | 372 (74.4%) | 34 (6.8%) | 94 (18.8%) |

### SOL 재생성 필요

- Summary: S5, S11, S12, S21, S32, S36, S38, S40
- Patch: S5, S11, S12, S16, S21, S24, S32, S34, S35, S36, S45, S48
- 양쪽 공통 우선순위: S5, S11, S12, S21, S32, S36

## 4. SOL 메모리 재생성 비용 추산

가격 가정:

- SOL input: $5.00 / 1M tokens
- SOL cached input: $0.50 / 1M tokens
- SOL output: $30.00 / 1M tokens

각 재생성 대상의 실제 R2 `model_calls(role=memory)` 사용량을 SOL에서도
유사하게 사용한다고 가정했다. 기존 trace의 cached token은 사실상 0이므로 아래는
cached-input 할인을 반영하지 않은 보수적 추정이다.

| 재생성 범위 | Calls | Input tokens | Output tokens | Total tokens | SOL 예상 비용 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Summary 8개 | 20,827 | 18.231M | 1.107M | 19.338M | **$124.36** |
| Patch 12개 | 33,038 | 35.103M | 0.709M | 35.811M | **$196.77** |
| 두 방법 전체 | 53,865 | 53.334M | 1.815M | 55.149M | **$321.13** |
| 공통 6개만 양쪽 우선 실행 | 31,148 | 28.305M | 0.960M | 29.265M | **$170.34** |

SOL은 Luna와 NO_OP reasoning 및 출력 길이가 다를 수 있고 재시도도 발생할 수 있다.
따라서 전체 실행 예산은 약 **$320**, 안전 예산은 **$350–400**으로 잡는다.
이 비용에는 재생성 후 answerability 재감사와 Quiz 재평가 비용은 포함하지 않았다.

## 5. 다음 실행 순서

비용을 고려해 SOL 대신 Luna로 모든 미해결 대상을 fresh R3 재생성한다.

- 시작 시각: 2026-08-18 11:20 UTC
- Summary: S5, S11, S12, S21, S32, S36, S38, S40
- Patch: S5, S11, S12, S16, S21, S24, S32, S34, S35, S36, S45, S48
- 메모리 모델: `gpt-5.6-luna`
- 실행 방식: 방법별 tmux controller, 최대 10개 시나리오 병렬
- Quiz: cache 완성을 위해 1개만 실행하며, 전체 Quiz는 R3 audit 통과 후 frozen
  replay한다.

R2 실측 토큰을 Luna 가격(input $0.20/M, output $1.20/M)에 적용한 예상
재생성 비용은 Summary 약 $4.97, Patch 약 $7.87, 합계 약 **$12.85**다.

완료 후 새 최종 메모리의 Missing 비율을 측정하고 `Missing < 30%`를 통과한
R3 source만 V2 silver training source로 채택한다.

## 6. Artifact

- R2 Summary audit:
  `/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2/summary-r2-final-memory-answerability-sol-s1-s50-20260818`
- R2 Patch audit:
  `/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2/combined-r2-final-memory-answerability-sol-s1-s50-20260818`
- Selected R1 Summary audit:
  `/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2/summary-r1-final-memory-answerability-sol-r2-ge30-selected-20260818`
- Selected R1 Patch audit:
  `/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2/combined-r1-final-memory-answerability-sol-r2-ge30-selected-20260818`
