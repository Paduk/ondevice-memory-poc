# VehicleMemBench V4 방법론별 예시

이 문서는 V4 실행 결과를 바탕으로 네 가지 memory profile의 차이를 쉽게
설명한다. 예시는 공식 benchmark 문장을 짧게 번역·요약했으며, 수치와 Tool
호출은 저장된 평가 trace를 따른다.

- 실행 ID: `a144fcfc-e6f7-4bdc-93bd-046d929c22c0`
- 범위: scenario 1–5, 50개 task, profile별 동일 task 실행
- 모델: Agent `gpt-5.6-terra`, Memory `gpt-5.6-luna`

## 1. No Memory

과거 대화 없이 현재 요청만 보고 행동하는 기준선이다.

**예시 — `vehicle-01-05`**

> 더운 날 Justin이 “환기를 켜되 지난번에 찾은 편안한 단계로 맞춰줘”라고
> 요청한다.

- 정답: 운전석 **시트 환기 속도 2**
- 실행 결과: 차량 **에어컨 전원 켜기**
- State F1: `0.0`, Tool F1: `0.0`

“지난번의 편안한 단계”가 현재 문장에 없으므로, Agent는 시트 환기 대신 일반
공조 환기로 잘못 해석했다. 이 profile은 memory가 실제로 필요한 task의
난이도를 보여준다.

## 2. Gold Memory

benchmark가 제공하는 정답 memory를 Agent에게 전달한다. 실제 memory 시스템의
성능이라기보다, 필요한 정보가 정확히 제공됐을 때의 상한선에 가깝다.

**예시 — `vehicle-01-05`**

- 제공된 핵심 정보: Justin의 더운 날 시트 환기 선호는 `2`
- 실행 결과: `seat_set_ventilation_speed(driver, 2)`
- State F1: `1.0`, Tool F1: `1.0`

같은 요청에서 No Memory는 틀렸지만 Gold Memory는 정확한 Tool과 인자를
선택했다. 다만 Gold Memory도 Agent의 Tool 선택 오류 가능성까지 제거하는
완전한 oracle은 아니다.

## 3. Cloud Summary Memory

긴 대화 이력을 Cloud LLM이 사람에게 읽기 쉬운 요약으로 압축하고, 그 요약을
Agent context에 넣는다.

**성공 예시 — `vehicle-04-03`**

> Matthew가 “내가 좋아하는 만큼 창문을 열고, 거기서 5% 더 열어줘”라고
> 요청한다.

- 요약에서 필요한 선호: 운전석 창문 개방도 `10`
- 정답 및 실행 결과: 운전석 창문 개방도 `15`
- State F1: `1.0`, Tool F1: `1.0`

요약이 기존 선호값을 보존한 경우 Agent는 `10 + 5`를 계산해 정확히
실행할 수 있었다.

**한계 예시 — `vehicle-01-05`**

최종 제공 요약에는 Justin의 시트 환기 선호가 유지되지 않았다. 원래 대화에는
필요한 정보가 있었지만 Agent가 받은 요약에서는 사용할 수 없어, No Memory와
동일하게 공조 전원을 켰다. 긴 대화를 하나의 갱신형 요약으로 압축할 때 중요한
세부 정보가 사라질 수 있음을 보여준다.

## 4. Cloud Structured Memory + Hybrid Retrieval

대화에서 `사용자–속성–값` 형태의 memory를 추출하고, 현재 요청과 관련된
항목만 BM25와 embedding 점수를 결합해 top-k로 검색한다.

**성공 예시 — `vehicle-01-05`**

- 검색 1순위: `Justin Martinez.seat_ventilation_level: 2`
- 실행 결과: `seat_set_ventilation_speed(driver, 2)`
- State F1: `1.0`, Tool F1: `1.0`

전체 요약에서는 빠졌던 정보를 구조화된 독립 항목으로 보존하고, “Justin”,
“ventilation”, “comfortable level”과 관련된 항목을 직접 검색해 복구했다.

**한계 예시 — `vehicle-05-06`**

> John이 Rachel과 둘이 이동하며 “오늘은 우리뿐이니 내비게이션 볼륨을
> 맞춰줘”라고 요청한다.

- 정답: 볼륨 `38`
- 검색 1순위: `John Robinson.navigation_volume: 80`
- 실행 결과: 볼륨 `80`
- State F1: `0.5`, Tool F1: `0.0`

구조화 과정에서 “누구와 타고 있는지”와 같은 조건을 충분히 보존하지 못하고
최신 일반값 `80`을 우선 검색했다. Hybrid가 Summary보다 평균적으로
우수하더라도, 조건 표현과 version 충돌 처리가 중요하다는 반례다.

## 5. State F1과 Tool F1이 다른 예시

**`vehicle-03-07` — 화학 공장 냄새가 다시 난 상황**

정답은 공조 순환을 `inside`로 설정하는 Tool 1회다.

| Profile | 실행 | State F1 | Tool F1 |
| --- | --- | ---: | ---: |
| No Memory | 아무 동작 없음 | 0.0 | 0.0 |
| Gold Memory | `inside` 1회 | 1.0 | 1.0 |
| Cloud Summary | 잘못된 값 1회 후 `inside` 실행 | 1.0 | 0.667 |
| Structured Hybrid | `inside` 1회 | 1.0 | 1.0 |

Cloud Summary는 최종 차량 상태를 정답으로 만들었으므로 State F1은 `1.0`이다.
하지만 불필요한 선행 호출이 있어 Tool F1은 `0.667`이다. 즉 State F1은
**최종 결과**, Tool F1은 그 결과에 도달한 **행동의 정확성과 효율성**을 본다.

## 6. V4 전체 결과 해석

| Profile | ESM | State F1 | Tool F1 | 핵심 해석 |
| --- | ---: | ---: | ---: | --- |
| No Memory | 0.20 | 0.360 | 0.210 | 과거 선호·조건을 추론할 근거가 부족함 |
| Gold Memory | 0.90 | 0.975 | 0.913 | 정확한 memory가 주어졌을 때의 상한선 |
| Cloud Summary | 0.30 | 0.462 | 0.333 | 단순 선호에는 유효하지만 세부 정보가 소실될 수 있음 |
| Structured Hybrid | 0.58 | 0.786 | 0.606 | 관련 fact 검색은 개선됐지만 조건·충돌 처리 오류가 남음 |

V4에서 Structured Hybrid는 No Memory와 Cloud Summary보다 뚜렷하게
개선됐지만 Gold Memory와는 차이가 있었다. V5에서는 전체 500개 task와
retrieval, gate, redaction ablation을 통해 이 차이의 원인을 분리한다.
