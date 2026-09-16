# HVP15 manual-style review

검토 결과: **external-adapted pilot 용도로 PASS, human-authored test 용도로는 제외**.

- 50 Turn을 순서대로 읽고 인간/합성 원문 25/25, UPDATE/NO_OP 10/40을 확인했다.
- 2026 시나리오에 부자연스러운 `Covid kit` 조언이 포함된 9턴 여행 대화를, 길이가 같고
  중복되지 않는 인간 작성 Bologna 여행 대화로 교체했다.
- 여행·영화 대화에는 지속 차량 설정이 없다. 충전 포트 열기, 배터리 상태 확인,
  odometer 확인도 현재 행동·조회 요청이므로 NO_OP가 적절하다.
- 순환 모드, 팬, driver-zone 온도, navigation voice, audiobook playback, 핸들 열선,
  rear trunk의 7개 활성 slot이 원래 Tool 호출과 대응한다.
- 순환 모드 outside→inside, 팬 7→8, 온도 19→18은 동일 slot, Tool, context arguments,
  activation condition을 유지하는 시간순 REPLACE다.
- 5턴 팬 대화는 값이 확정되지 않은 앞의 4턴을 NO_OP로 두고, 사용자가 `8`을 확정한
  마지막 턴만 REPLACE로 처리했다.
- `forgot gloves`, `obvious route`, `audiobook stopped` 조건은 원문보다 넓히지 않고 그대로
  memory와 Quiz에 유지했다.
- 기본 상태와 같은 rear-trunk close는 중간 Quiz에서 제외하고, Final Quiz에서는 조건을
  명시해 핸들 열선과 함께 평가한다.
- Turn Quiz 4개와 Final Quiz 6개의 cutoff, evidence slot, 최종 활성값을 대조했다.
- 공식 Tool JSON schema, VehicleWorld 실행, Patch replay, NO_OP identity, Quiz cutoff를
  포함한 자동 검증을 모두 통과했다.

외부 원문의 문법과 말투는 source trace에 보존했다. 지속성 문장, memory label, Quiz가
모델 작성이므로 실제 human-authored test로 간주하지 않는다.
