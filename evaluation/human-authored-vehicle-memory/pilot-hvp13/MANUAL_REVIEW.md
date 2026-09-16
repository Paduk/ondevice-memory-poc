# HVP13 manual-style review

검토 결과: **external-adapted pilot 용도로 PASS, human-authored test 용도로는 제외**.

- 50 Turn을 순서대로 읽고 인간/합성 원문 25/25, UPDATE/NO_OP 10/40을 확인했다.
- 파티·여행 대화의 기호와 일정은 차량 Tool의 지속 설정이 아니며, 세 개의 차량 대화도
  배터리 preconditioning, 장소 검색, OTA 상태 확인이라는 일회성 요청이므로 NO_OP가
  적절하다.
- 팬, driver window, rear-right 온도, defrost, navigation voice, playback, 핸들 열선의
  7개 활성 slot이 원문 Tool 호출과 대응한다.
- 팬 5→1→3과 driver window 55→90은 동일 slot, Tool, context arguments, activation
  condition을 유지하는 시간순 REPLACE다.
- rear-right 온도와 front-and-rear defrost의 zone을 VehicleMemBench ontology에 맞게
  정규화했고 사용자 귀속 오류는 없다.
- 이야기 도중 navigation 안내가 끼어든 원문을 포괄적인 `spoken audio` 조건으로 넓혔던
  표현을 `when Cameron is telling a story`로 좁혔다.
- `resume playing` 원문을 포괄적인 selected-audio 선호로 넓혔던 표현을
  `when Cameron's audio playback has stopped`로 좁혔다.
- Turn Quiz 4개는 해당 시점의 초기값을, Final Quiz 6개는 최종 활성값만 참조하며 모든
  질문에 activation condition이 명시돼 있다.
- 공식 Tool JSON schema, VehicleWorld 실행, Patch replay, NO_OP identity, Quiz cutoff를
  포함한 자동 검증을 모두 통과했다.

원문의 문법 오류와 말투 차이는 외부 표현 전이를 위해 그대로 보존했다. 이 시나리오는
지속성 문장, memory label, Quiz가 모델 작성이므로 실제 인간 검증셋의 대체물이 아니다.
