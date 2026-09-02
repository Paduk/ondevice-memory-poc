# V1-style 20개 시나리오 증강 계획

## 목표

기존 V1 학습용 10개를 반복 변형하지 않고, **나머지 V1 시나리오 중 서로 다른 20개**를
참조하여 V1의 표현·UPDATE·Quiz 분포에 유리한 증강 시나리오 20개를 만든다. 기존 V2
능력은 유지하고 2B 모델의 V1 invalid 출력과 Tool/Argument 오류를 줄이는 것이 목적이다.

## 원본 선택

- 기존 학습 10개 `S2, S5, S6, S14, S17, S23, S31, S32, S33, S36`은 제외한다.
- 남은 40개에서 20개를 한 번씩 선택한다.
- 2B Patch의 invalid·UPDATE miss·Quiz 오류가 큰 사례를 우선하되 Tool, reasoning type,
  선호 변경 및 조건부 상태가 한쪽에 몰리지 않도록 균형화한다.
- 선택되지 않은 20개는 **clean held-out V1**으로 고정한다.

## 생성 방식 (style-v2)

각 원본의 **모든 turn을 구조 template**로 사용한다. turn 수, 세션 경계, timestamp,
speaker 순서, UPDATE/NO_OP 위치, 선호 변화, Tool/Argument 조합, multi-call 수와 reasoning
구조는 동일하게 유지한다. 사용자 이름과 모든 대화·Quiz 문장은 Luna가 source text를
semantic blueprint로만 사용하여 처음부터 다시 쓰되,
차량 설정값과 조건 의미는 teacher trajectory와 동일하게 보존한다. Patch와 memory는
teacher의 승인된 memory version을 새 사용자 이름으로 결정론적으로 변환한다.

이 방식은 clean-heldout 생성이 아니라 **training-only structural clone**이다. 참조한 원본
시나리오는 adaptation 진단 및 최종 평가에서 반드시 제외한다. 원문 복사는 막되 4/8-gram
중복보다는 state/trajectory의 완전한 보존을 우선한다.

## 품질 검사와 학습 변환

- Patch schema 파싱 및 turn-wise 결정론적 적용 100%
- 사용자별 memory 배치와 최종 state 일치 검사
- Quiz tool 실행 가능성 및 정답-memory 정합성 검사
- UPDATE 전부, UPDATE 인접 NO_OP 우선, 일반 NO_OP 일부를 기존 noop5
  멀티태스크 포맷으로 변환
- 새 V1-style 20개 시나리오의 Final Quiz는 10개씩 총 200개를 모두 학습
- 기존 데이터와 동일한 inline JSON 및 grouped memory 형식을 사용

## 평가

동일 2B Patch 설정으로 `기존 학습`과 `+V1-style 20`을 비교한다.

1. clean held-out V1 20개: ESM, Tool F1, Arg Exact, invalid rate
2. V2 S86–S100 및 T12–T20: 기존 성능 유지 여부
3. 증강 참조 20개 성능은 adaptation 진단값으로만 보고하고 최종 일반화 점수에서는 제외

채택 기준은 clean V1 ESM·invalid rate 개선과 V2 ESM 하락 `1%p 이내`를 동시에 만족하는
것이다.

## 실행 현황 (2026-08-31)

- 선정 원본 20개: `S3, S4, S7, S8, S9, S11, S12, S15, S16, S18, S21, S25,
  S26, S27, S28, S37, S38, S44, S46, S49`
- clean held-out 20개: `S1, S10, S13, S19, S20, S22, S24, S29, S30, S34, S35,
  S39, S40, S41, S42, S43, S45, S47, S48, S50`
- 첫 S44 파일럿은 V2식 신규 생성이라 UPDATE가 후반 81–94%에 몰려 폐기했다.
- `S44-style-v2` 파일럿 생성 및 audit 완료: 원본의 2,650 turns, 70 sessions,
  UPDATE 24개 위치, reasoning 순서와 12개 Tool calls를 모두 exact constraint로 보존했다.
- 본문 생성 모델은 `GPT-5.6 Luna`로 통일하고, Patch 적용·Tool schema·구조 audit만
  결정론적으로 수행한다.
- Luna 사용량은 input 189,647, cached 20,838, output 62,476 tokens이며 추정 비용은
  `$0.109`이다. 원문과 완전히 동일한 발화는 짧은 관용 표현 2/2,650건(0.075%)이다.
- 파일럿 경로:
  `/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v1-style-augmentation-v1/pilot-s44-style-v2`
- 1차 확장 `S3, S4, S7, S8` 생성 및 audit 완료. 네 시나리오 모두 원본의
  turn/session/UPDATE 위치, reasoning type, Tool/Arguments를 보존했고 기존 인명 잔존은
  없다. 총 10,930 turns, Luna 1.129M tokens, 추정 비용 `$0.503`이다.

산출물은
`/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v1-style-augmentation-v1/`에
있다. S11 복구분을 포함해 **V1-style 20개가 준비된 상태**로 고정한다.

## V1 S1–S50 평가 집계 규칙

향후 모델은 V1 `S1–S50` 전체를 한 번에 평가하되 결과는 반드시 다음 집단으로 같이
집계한다. 시나리오별 원시 결과를 보존하여 재실행 없이 집단별 점수를 다시 계산할 수
있어야 한다.

| 보고 집단 | 시나리오 | 해석 |
|---|---|---|
| 전체 50 | `S1–S50` | 기존 V1 전체 성능 |
| V1-style 참조 20 | `S3, S4, S7, S8, S9, S11, S12, S15, S16, S18, S21, S25, S26, S27, S28, S37, S38, S44, S46, S49` | 구조를 참조한 adaptation 진단; 일반화 점수로 사용하지 않음 |
| 비참조 30 | `S1, S2, S5, S6, S10, S13, S14, S17, S19, S20, S22, S23, S24, S29, S30, S31, S32, S33, S34, S35, S36, S39, S40, S41, S42, S43, S45, S47, S48, S50` | 참조 20개를 제외한 성능. 단, 아래 기존 학습 10개를 포함함 |
| 기존 V1 학습 10 | `S2, S5, S6, S14, S17, S23, S31, S32, S33, S36` | 이미 학습에 사용된 seen 분포 진단 |
| strict clean-heldout 20 | `S1, S10, S13, S19, S20, S22, S24, S29, S30, S34, S35, S39, S40, S41, S42, S43, S45, S47, S48, S50` | 최종 일반화 판단의 주 지표 |

모든 집단에서 `ESM`, `Tool F1`, `Arg Exact`, `invalid rate`를 보고한다. V1-style 보강의
효과는 `참조 20`의 상승만으로 결론 내리지 않고, **strict clean-heldout 20 개선**과 기존
V2 성능 유지가 동시에 확인될 때 채택한다. 기계 판독용 고정 split은 산출물 루트의
`evaluation-splits.json`을 사용한다.
