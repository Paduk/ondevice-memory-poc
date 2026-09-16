# HVP14 manual-style review

검토 결과: **external-adapted pilot 용도로 PASS, human-authored test 용도로는 제외**.

- 50 Turn을 순서대로 읽고 인간/합성 원문 25/25, UPDATE/NO_OP 10/40을 확인했다.
- 날씨·영화 예약 대화에는 지속적인 차량 설정이 없다. 충전 케이블 해제, 즉시 충전,
  remote summon도 현재 행동 요청일 뿐 반복 선호가 아니므로 NO_OP가 적절하다.
- 선루프, 기본 목적지, front trunk, navigation voice, rear defrost, rear trunk, 핸들
  열선의 7개 활성 slot이 원래 Tool 호출과 대응한다.
- 선루프 30→18, 목적지 Work→Home, 핸들 열선 off→on은 동일 slot, Tool, context
  arguments, activation condition을 유지하는 시간순 REPLACE다.
- Front trunk 조건을 원문에 없던 `groceries`로 한정했던 표현을 일반적인
  `when Parker loads cargo`로 고쳐 임의의 화물 종류를 제거했다.
- Rear-only defrost는 원문의 `front=False, rear=True`를 VehicleMemBench의
  `zone=rear, mode=defrost`로 손실 없이 정규화했다.
- 마지막 핸들 열선 원문에는 추위를 느끼는 현재 상황이 있지만, 덧붙인 마지막 문장에서
  사용자가 이를 새로운 평상시 설정으로 명시하므로 REPLACE label과 충돌하지 않는다.
- 기본 상태와 같은 rear-trunk close는 중간 Quiz에서 제외했고, Final Quiz에서는 현재
  핸들 열선 설정과 함께 명시적으로 묻는다.
- Turn Quiz 4개와 Final Quiz 6개의 조건, evidence slot, gold call을 각각 대조했다.
- 공식 Tool JSON schema, VehicleWorld 실행, Patch replay, NO_OP identity, Quiz cutoff를
  포함한 자동 검증을 모두 통과했다.

외부 원문의 문법과 표현은 source trace에 그대로 보존했다. 지속성 문장, memory label,
Quiz는 모델 작성이므로 실제 human-authored test로 간주하지 않는다.
