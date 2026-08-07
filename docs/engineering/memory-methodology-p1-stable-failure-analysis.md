# P1 안정 실패의 selector·binding 분석

상태: `초기 분석 완료 — Scenario 1–5 교차 확인 후 구현 보류`

## 목적과 범위

P0에서 Recursive-assisted와 Post-normalized가 모두 3회 연속 실패한 7개 task를
이용해 Tool selector와 argument binding의 반복 가능한 원인을 찾았다. 분석에는
Gold call을 실패 위치 확인에만 사용했다. 수정 후보에는 task ID, Gold 값, 특정
인물 이름을 넣지 않는다.

두 방법은 방법별 고정 cache를 사용하므로 같은 방법의 세 Agent run에서 Fact,
retrieval, selector 결과는 동일하다. 아래 분석은 각 방법의 대표 run
`60795558-33e4-47b5-80e9-0d677ed27a08`,
`15f5a789-8dd5-4d6c-9190-887290bd5bff`의 trace를 기준으로 했다.

## 결론

7건을 모두 binding 오류로 보면 안 된다.

- 4건은 필요한 핵심 Fact 값이 active memory에 없다. 후단만 고쳐서는 복구할 수
  없으며 P2 extraction 대상이다.
- 2건은 값 일부는 있지만 원문 또는 공개 Tool schema만으로 target·필수 인자를
  결정할 수 없다. Gold에 맞춘 default를 넣으면 overfit이다.
- 1건은 정답 Fact와 Tool이 모두 존재하지만 시간 조건 비교가 이를 잘못 제거한다.
  이는 P1에서 바로 고칠 수 있는 명확한 범용 결함이다.
- Gold Tool은 7건 중 6건에서 selector 결과에 없었다. 이 현상은 안정 실패에만
  국한되지 않는다. 두 대표 run 모두 전체 50건 중 27건에서 reference Tool 하나
  이상이 빠졌다.

Selector 누락이 곧 실패를 뜻하지는 않는다. Agent discovery로 성공한 경우도
Recursive-assisted 11건, Post-normalized 14건이었다. 그러나 실패 task에서는
각각 16/21건, 13/20건이 selector 누락을 포함해, Agent가 보완해야 하는 부담이
지나치게 크다.

## 7개 안정 실패 분류

| Task | 관찰 | 1차 분류 | 범용 처리 |
| --- | --- | --- | --- |
| `vehicle-06-05` | `480p`·재생 대상·switch를 복원할 핵심 Fact가 없고 selector는 quality Tool만 선택 | P2 Fact 누락 + 복합 호출 누락 | P1만으로 정답 복구를 주장하지 않음 |
| `vehicle-07-05` | `English` Fact는 있으나 target이 `unspecified`; center display와 instrument panel의 공개 language schema도 동일 | target 결정 불가 | 두 sibling Tool을 유지하고 ambiguity로 처리; center display 강제 금지 |
| `vehicle-07-06` | `18°C` Fact는 선택됐지만 selector가 circulation/mode로 향하고 `zone=all` 근거는 없음 | P1 후보 제한 + 필수 인자 불충분 | Fact capability로 temperature Tool을 보강하되 zone은 추측하지 않음 |
| `vehicle-07-08` | Jacob의 overhead screen `level=1`, `time=night` Fact가 존재하지만 8 PM을 `evening`으로 해석해 `condition_mismatch`로 제거 | 명확한 P1 retrieval 오류 | `dark outside` 같은 명시적 환경 신호를 시간대 조건에 반영 |
| `vehicle-09-00` | ambient `green` Fact가 없고, 관용구 `right headspace`의 `right`가 turn signal Tool을 지배 | P2 Fact 누락 + P1 lexical 오탐 | 방향 단어만으로 Tool을 확정하지 않음 |
| `vehicle-09-09` | 이전 radio station과 switch 상태를 복원할 핵심 Fact가 없고 switch Tool도 selector에서 누락 | P2 Fact 누락 + 복합 호출 누락 | P1만으로 값은 복구할 수 없음 |
| `vehicle-10-05` | air-direction 핵심 값이 없고, `1:00 PM`의 `1`이 level/speed Tool과 lexical match | P2 Fact 누락 + P1 숫자 오탐 | 시각 숫자를 routing intent에서 제거; 비공개 enum 문자열 추측 금지 |

`vehicle-07-05`의 evidence는 “Switched it back to English”로 모듈을 말하지 않는다.
공개 schema에서 center display와 instrument panel language Tool은 설명과 인자가
같다. `vehicle-07-06`의 evidence도 “I set the AC to 18°C”로 zone을 말하지 않는다.
따라서 이 두 값을 Gold에 맞춰 채우는 것은 범용 binding 개선이 아니다.

## 범용 원인

### 1. query-only selector가 값·문맥 토큰에 쉽게 끌린다

`memory_router.py`는 query token과 Tool domain/topic/slot/description token을
결합한다. 숫자도 token으로 유지되므로 시각의 `1`이 schema 설명의 level 범위와
일치한다. `right` 같은 방향 단어도 충분한 domain/action 근거 없이 강한 점수를
얻는다. 반면 선택된 Fact의 predicate와 capability는 selector 후보를 보강하지
않는다.

### 2. 잘못된 route가 late binding의 유효 후보 탐색을 막는다

`tool_memory_execution.py`는 routed Tool에 `+25`를 주고 최고 점수 동률 집합만
schema validation한다. 최고 집합이 모두 invalid여도 다음 점수의 Tool은 검사하지
않는다. `18°C` Fact가 temperature Tool 대신 routed circulation/mode에서 먼저
검사되고 폐기된 사례가 여기에 해당한다.

### 3. 시간 조건을 단일 exact bucket으로 비교한다

현재 8 PM은 먼저 `evening`으로 정규화되고, stored `night`와 exact match가 아니면
Fact가 즉시 제외된다. 같은 query의 “dark outside”는 사용되지 않는다. evening과
night를 전부 합치는 대신 clock과 명시적 환경 표현을 함께 보존해야 한다.

### 4. 정보가 없는 필수 인자는 안전하게 거절되고 있다

현재 role binding은 query/entity role에서 구할 수 있는 값만 채운다. 이 때문에
target, zone, 숨겨진 enum이 없는 경우 hint가 거절된다. 정답률만 보고 기본값을
추가하기보다 이 보수적 동작을 유지하는 것이 맞다.

## 구현 후보와 overfit 방지선

1. **Routing hygiene**: 시각 span과 숫자-only token은 lexical Tool 선택에서
   제외한다. `left/right` 같은 약한 slot 단어는 domain/action anchor가 함께 있을
   때만 결정 근거로 사용한다.
2. **Fact-assisted candidate union**: query route와 selected Fact의
   predicate/capability로 만든 schema 후보를 합친다. routed Tool은 우선순위일 뿐
   hard gate로 쓰지 않고, 상위 점수 후보가 모두 invalid면 다음 점수 tier를
   검증한다. 단, 유효한 sibling이 여럿이면 계속 ambiguity로 거절한다.
3. **Condition signal 보존**: clock bucket 외에 `night`, `dark outside` 같은 명시적
   환경 신호를 조건 판정에 사용한다. 모든 evening Fact를 night와 동일시하지 않는다.
4. **추측 금지**: `unspecified→center display`, 누락 zone의 `all` default,
   benchmark simulator의 숨겨진 air-direction enum은 추가하지 않는다.

이 세부 후보는 바로 구현하지 않는다. 후속 Scenario 1–5 비교에서도 selector 누락은
22/50으로 반복됐지만, 숫자·방향어·시간 조건 각각의 직접 사례 수는 작았다.
[Scenario 1–5 비교 결과](memory-methodology-s1-s5-comparison-results.md)에 따라 먼저
Scenario 1–10 전체 실패를 같은 taxonomy로 분류하고, 여러 Scenario를 동시에 덮는
수정만 선택한다.

후속 100-task 분석 결과와 방법론 차이에 기반한 구조적 개선 방향은
[Scenario 1–10 구조적 병목 비교](memory-methodology-s1-s10-structural-bottleneck-comparison.md)를
따른다.

인물 role parser가 가까운 두 인물을 모두 driver로 잡거나, “feet warm”을 weather
`hot`으로 읽는 부수 오탐도 trace에서 확인됐다. 다만 7건의 직접 원인은 아니므로
이번 최소 P1 구현 범위에는 넣지 않고 별도 회귀 지표로 감시한다.

## 구현 전 검증 기준

P1 수정은 먼저 Agent LLM 없이 고정 cache를 replay한다.

- 전체 50건 reference-Tool coverage를 보고, 7개 task만의 성공 여부로 튜닝하지 않는다.
- 기존 selector가 맞았던 task의 Tool을 제거하지 않는다.
- 모든 emitted hint는 public JSON schema를 통과해야 한다.
- target·필수 인자가 없는 negative fixture는 계속 ambiguity 또는 invalid로 남아야 한다.
- task ID, 인물 이름, Gold argument literal이 코드·fixture의 분기 조건에 없어야 한다.

이 gate를 통과한 뒤에만 P0와 같은 고정-cache Agent 3회 평가로 ESM 영향을 확인한다.
