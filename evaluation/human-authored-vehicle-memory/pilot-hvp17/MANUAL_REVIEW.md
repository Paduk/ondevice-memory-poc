# HVP17 manual-style review

검토 결과: **external-adapted pilot 용도로 PASS, human-authored test 용도로는 제외**.

- 50 Turn을 순서대로 읽고 인간/합성 원문 25/25, UPDATE/NO_OP 10/40을 확인했다.
- 수면 podcast, 날씨·여행 준비, 도착 문자, 수면 조회, 즉시 프런트 트렁크 열기,
  자동 주차는 일회성 대화 또는 즉시 실행 요청이므로 NO_OP가 적절하다.
- 조명, 온도, rear-right 창문, 기본 목적지, defrost, 순환 모드, rear trunk의
  7개 활성 slot은 각 Audio2Tool 원본의 정답 호출과 대응한다.
- `Passenger_Rear=60`은 VehicleMemBench의 `rear_right=60`, front/rear defrost는
  `zone=all`, recirculation disabled는 `circulation=outside`, rear cargo close는
  `rear trunk=false`로 의미 보존 정규화했다.
- 일회성 복합 명령의 밝기와 HVAC mode는 지속 memory에서 제외하고, 원문에 명시된
  색상과 온도만 지속값으로 남겼다.
- 빨간 조명 원문에는 밝기 요청이 없는데도 추가 문장이 밝기를 언급하던 문제를 발견해
  해당 표현을 제거했다.
- 세 REPLACE는 `ambient color: pink→red→white`와 `temperature: 18→27`이다.
  각 전이는 동일 slot, Tool, context arguments, activation condition을 유지한다.
- Turn Quiz 4개와 Final Quiz 6개의 cutoff, evidence slot, 활성 조건과 최종값을
  대조했다. 마지막 복합 Quiz는 stale-air headache와 normal driving을 모두 명시한다.
- 공식 Tool JSON schema, VehicleWorld 실행, Patch replay, NO_OP identity, Quiz cutoff를
  포함한 자동 검증을 모두 통과했다.

외부 원문의 문법과 말투는 source trace에 그대로 보존했다. 지속성 문장, memory label,
Quiz가 모델 작성이므로 실제 human-authored test로 간주하지 않는다.
