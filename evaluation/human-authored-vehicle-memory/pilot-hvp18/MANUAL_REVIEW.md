# HVP18 manual-style review

검토 결과: **external-adapted pilot 용도로 PASS, human-authored test 용도로는 제외**.

- 50 Turn을 순서대로 읽고 인간/합성 원문 25/25, UPDATE/NO_OP 10/40을 확인했다.
- 여행 준비, sleep cast, 다음 날 알람, HRV 조회, Smart Summon, 충전 전류 조정은
  차량의 지속 선호가 아니라 일회성 대화·즉시 실행이므로 NO_OP가 적절하다.
- 조명, rear-right 좌석 환기, 온도, 순환 모드, front trunk, 선루프, rear defrost의
  7개 활성 slot이 외부 원문의 정답 호출과 대응한다.
- 최초 구성은 좌석 냉방 `level=2`를 단순 enabled 상태로 축소하고, 이후 아들의 좌석
  냉방 OFF를 Harper 본인의 정상값으로 합쳐 owner와 속성이 모두 불안정했다. 이를
  `rear_right ventilation speed=2`로 정확히 보존하고, 아들 관련 발화는 제거했다.
- 대체한 외부 온도 발화를 이용해 온도 REPLACE를 24→21→28로 구성했다. 조명은
  pink→red이며, 세 REPLACE 모두 같은 slot, Tool, context arguments, activation
  condition을 유지한다.
- 밝기와 HVAC mode는 해당 발화의 일회성 복합 속성으로 두고, 명시적으로 지속화한
  색상·온도만 memory에 보존했다.
- `side-window visibility→outside circulation`, `locked front trunk→open`,
  `quieter→sunroof 0`, `rear visibility→rear defrost`의 의미 대응을 확인했다.
- Turn Quiz 4개와 Final Quiz 6개의 cutoff, evidence slot, 활성 조건과 최종값을
  대조했다. 기본 simulator 상태와 같은 단독 호출은 피하면서 모든 활성 slot을 다룬다.
- 공식 Tool JSON schema, VehicleWorld 실행, Patch replay, NO_OP identity, Quiz cutoff를
  포함한 자동 검증을 모두 통과했다.

외부 원문의 문법과 말투는 source trace에 그대로 보존했다. 지속성 문장, memory label,
Quiz가 모델 작성이므로 실제 human-authored test로 간주하지 않는다.
