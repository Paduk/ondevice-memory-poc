# HVP16 manual-style review

검토 결과: **external-adapted pilot 용도로 PASS, human-authored test 용도로는 제외**.

- 50 Turn을 순서대로 읽고 인간/합성 원문 25/25, UPDATE/NO_OP 10/40을 확인했다.
- 공연·영화 예약과 walking direction은 일회성 요청이다. Spotify 전환도 `지금` 수행하는
  요청일 뿐 지속 선호라는 명시가 없으며, 배터리·odometer 조회도 저장 대상이 아니므로
  NO_OP가 적절하다.
- 순환 모드, 선루프, navigation voice, playback, defrost, driver window, 핸들 열선의
  7개 활성 slot이 원래 Tool 호출과 대응한다.
- 창문 김서림 방지를 위한 outside circulation과 식당 냄새 차단을 위한 inside
  circulation은 조건이 다르므로 REPLACE로 묶지 않았다. 김서림 조건의 outside
  circulation만 독립 memory로 유지했다.
- 비가 올 때 창문을 닫는 원문을 평상시 driver-window 설정으로 강제했던 REPLACE를
  제거했다. 5턴 창문 대화는 앞의 4턴을 NO_OP로 두고 값과 대상이 확정된 마지막 턴만
  `driver window=50` ADD로 처리했다.
- 과도하게 넓었던 `spoken audio`와 `selected audio` 조건을 각각 `telling a story`와
  `audio playback has stopped`로 좁혀 원문 조건을 보존했다.
- 세 REPLACE는 모두 선루프의 평상시 설정 10→15→30→0으로 재구성했다. 각 단계는
  동일 slot, Tool, context arguments, activation condition을 유지하고 값만 변경한다.
- Turn Quiz 4개와 Final Quiz 6개의 cutoff, evidence slot, 최종 활성값을 대조했다.
- 공식 Tool JSON schema, VehicleWorld 실행, Patch replay, NO_OP identity, Quiz cutoff를
  포함한 자동 검증을 모두 통과했다.

외부 원문의 문법과 말투는 source trace에 보존했다. 지속성 문장, memory label, Quiz가
모델 작성이므로 실제 human-authored test로 간주하지 않는다.
