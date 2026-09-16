# HVP19 manual-style review

검토 결과: **external-adapted pilot 용도로 PASS, human-authored test 용도로는 제외**.

- 50 Turn을 순서대로 읽고 인간/합성 원문 25/25, UPDATE/NO_OP 10/40을 확인했다.
- 생일 선물 주문, 다음 날 알람, 수면 playlist, 수면 조회, Sentry Mode, 즉시 도어
  해제는 차량의 지속 선호가 아닌 일회성 대화·실행 요청이므로 NO_OP가 적절하다.
- 조명, rear-right 좌석 환기, 순환 모드, front defrost, front trunk, 도어 잠금,
  라디오의 7개 활성 slot이 외부 원문의 동작과 대응한다.
- 좌석 냉방 `level=2`를 단순 enabled 상태로 축소하던 문제를 수정해 VehicleMemBench의
  `rear_right ventilation speed=2`로 값과 위치를 모두 보존했다.
- Bioweapon Defense를 inside circulation으로 근사하던 두 발화를 제거했다. 실제
  `setRecirculation` 원문을 사용해 `recirculation on→off`를 VehicleMemBench의
  `inside→outside`와 직접 대응시켰다.
- 세 REPLACE는 `ambient color: white→red→white`와 `circulation: inside→outside`다.
  모두 동일 slot, Tool, context arguments, activation condition을 유지한다.
- 조명 밝기는 발화별 일회성 값으로 두고 색상만 지속값으로 보존했다. Media source의
  Radio 선택은 VehicleMemBench에서 같은 효과를 내는 radio-on 동작으로 정규화했다.
- Turn Quiz 4개와 Final Quiz 6개의 cutoff, evidence slot과 최종값을 대조했다.
  마지막 복합 Quiz는 parked access와 background-noise 조건을 모두 활성화한다.
- 공식 Tool JSON schema, VehicleWorld 실행, Patch replay, NO_OP identity, Quiz cutoff를
  포함한 자동 검증을 모두 통과했다.

외부 원문의 문법과 말투는 source trace에 그대로 보존했다. 지속성 문장, memory label,
Quiz가 모델 작성이므로 실제 human-authored test로 간주하지 않는다.
