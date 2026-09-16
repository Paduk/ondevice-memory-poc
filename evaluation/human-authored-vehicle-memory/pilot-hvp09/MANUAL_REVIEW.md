# HVP09 manual-style review

검토 결과: **external-adapted pilot 용도로 PASS, human-authored test 용도로는 제외**.

- 50 Turn, 인간/합성 원문 25/25, UPDATE/NO_OP 10/40을 확인했다.
- 팬, 조명, 트렁크, 선루프, 음성 안내, 잠금, 핸들 열선 7개 slot이 원문과 대응한다.
- 팬 6→8, 주차 잠금→출발 시 잠금 해제, 일반 열선 off→추운 날 on은 시간순
  REPLACE다.
- 붉은색 hex 값은 VehicleWorld의 동치 열거값 `red`로 정규화했다.
- 발렌타인 맥락의 붉은 조명을 평상시 설정으로 과도하게 일반화하지 않고 `romantic cabin
  mood` 조건에 묶었다.
- “핸들이 너무 뜨겁다”는 원문에 없던 추운 날 조건을 초기 memory에서 제거했다.
- 잠금 상태 slot을 주차에 편향되지 않은 `vehicle_access`로 바꾸고 Quiz 10개를 실제
  activation condition이 드러나도록 다시 작성했다.
- 기본값과 같은 최종 호출만 단독으로 묻지 않도록 Quiz 순서를 점검했다.

지속성 문장과 memory/Quiz label은 모델 작성이므로 실제 human-authored test에는 포함하지
않는다.
