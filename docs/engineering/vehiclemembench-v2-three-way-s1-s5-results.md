# VehicleMemBench V2 3-way S1–S5 결과

## 비교 조건

- `post_hoc`, `hybrid`, `native_turnwise`가 동일한 frozen Stage 2의 S1–S5를 사용했다.
- 방법별 40 Quiz × 5 scenarios × 2 fresh repeats, 총 400 Agent tasks를 평가했다.
- Answerability는 방법별 200 Quiz를 전수 검사했다.
- UPDATE 감사의 주 비교는 방법별 동일한 86 core events를 사용한다. 표본 오류로
  자동 전수 확대된 NO_OP 층은 진단용 결과로 분리해 주 비교 분모에서 제외했다.
- 비용은 GPT-5.6 Luna/Terra/Sol의 2026-08 사용자 제공 단가로 추정했다.

## 전체 결과

| Method | ESM | Tool F1 | Arg Exact | Answer support | Audit PASS | Expected recall | Candidate precision | NO_OP accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Post-hoc | 0.892 | 0.906 | 0.845 | 0.940 | 0.941 | 1.000 | 0.940 | 0.944 |
| **Hybrid** | **0.917** | **0.936** | **0.858** | **0.965** | **0.953** | **1.000** | **0.941** | **0.972** |
| Native turn-wise | 0.902 | 0.925 | 0.853 | 0.955 | 0.686 | 0.860 | 0.490 | 0.944 |

Post-hoc S3의 `veh-09-e2` 한 건이 Human Review 대기 중이므로 Audit 수치는
잠정값이다. 이 한 건의 결론은 전체적인 방법 순위에는 영향을 주지 않는다.

## 효율

### 데이터셋 생성

Dialogue, turn-wise memory alignment, Turn Quiz 30개와 Final Quiz 10개 생성 비용을
합산했다. 누적 latency는 병렬 wall-clock 시간이 아니라 provider 호출 latency의 합이다.

| Method | Input | Output | 누적 latency | 예상 비용 |
|---|---:|---:|---:|---:|
| Post-hoc | 2.325M | 0.687M | 125.3분 | $12.89 |
| **Hybrid** | **2.326M** | **0.684M** | **123.2분** | **$12.77** |
| Native turn-wise | 46.083M | 1.298M | 529.7분 | $107.74 |

Native turn-wise는 매 turn마다 증가하는 causal prefix로 dialogue를 생성하므로 Hybrid
대비 입력이 약 19.8배, 비용이 약 8.4배였다.

### 평가

| Method | Quiz input/output | Quiz 누적 latency | Quiz+Answerability+Core Audit 비용 |
|---|---:|---:|---:|
| Post-hoc | 1.594M / 0.043M | 27.7분 | $3.83 |
| Hybrid | 1.641M / 0.044M | 28.7분 | $3.74 |
| Native turn-wise | 1.641M / 0.044M | 27.8분 | $5.40 |

Native의 평가 비용이 높은 이유는 Luna/Terra 불일치와 Sol adjudication이 더 많이
발생했기 때문이다. 이는 방법 자체의 runtime 비용이 아니라 품질 감사 비용이다.

## 해석

- **Hybrid가 현재 scale gate의 최선이다.** ESM·Tool F1·Arg Exact·Answerability와
  UPDATE 감사 모두 가장 높고, 생성 비용은 Post-hoc과 사실상 같다.
- Native turn-wise의 Quiz 성능은 나쁘지 않지만, temporal label 품질이 약하다.
  core candidate 51건 중 `VALID 25`, `PREMATURE 25`, `SPURIOUS 1`이었고 expected
  update 50건 중 7건이 `MISSING`이었다.
- Native의 audit-error turn Quiz 52회 중 50회는 memory상 정답이 지원되었고 Agent도
  맞혔다. 즉 낮은 Audit PASS는 주로 **최종 memory 내용 부족보다 UPDATE 시점 label의
  부정확성**을 드러낸다.
- 따라서 학습 데이터 생성 방식은 Hybrid를 기본으로 선택하고, Native turn-wise는
  자연스러운 실시간 대화 생성 연구용 ablation으로 두는 것이 타당하다.

재현 가능한 raw summary는
`vehiclemembench-v2-three-way-evaluation/three-way-summary.json`과 `.md`에 저장된다.
