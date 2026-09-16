# HVP12 manual-style review

검토 결과: **external-adapted pilot 용도로 PASS, human-authored test 용도로는 제외**.

- 50 Turn, 인간/합성 원문 25/25, UPDATE/NO_OP 10/40을 확인했다.
- 온도, 목적지, audiobook 재생, 안내 음성, defrost, 핸들 열선, 잠금 7개 slot을
  확인했다.
- 목적지 Home→Riverside Dog Park, 평상시 열선 off→on, 주차 상태 unlock→lock은
  전후 조건과 Tool 속성이 같은 시간순 REPLACE다.
- audiobook 재생을 일반 music 선호로 넓히지 않고 동일 media 조건에 묶었다.
- “안내가 너무 잦다”는 발화를 `obvious route`로 바꾸지 않고 prompt frequency 조건을
  그대로 유지했다.
- 열선 초기 발화에 근거 없이 붙었던 cold-drive 조건을 제거하고, 도어 전후 조건을 모두
  동일한 parked 상태로 맞췄다.
- `(Names film...)` 메타 응답이 포함된 인간 원문을 실제 상영 시간으로 답하는 자연스러운
  2턴 cinema record로 교체했다.
- Quiz 10개를 실제 activation condition이 드러나도록 다시 작성했다.
- 상태 기본값과 같은 초기 설정은 Turn Quiz 대상에서 제외했고 최종 호출은 모두 재생된다.
- Turn/Final Quiz가 각 cutoff에서 활성인 evidence만 참조하는지 확인했다.

말투와 assistant 이름이 세션마다 달라질 수 있으며, 실제 인간 검증셋의 대체물은 아니다.
