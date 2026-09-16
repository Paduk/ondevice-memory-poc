# HVP04 manual-style review

검토 결과: **HVP05–HVP20과 같은 external-adapted pilot 규격으로 PASS,
human-authored test 용도로는 제외**.

- 50 Turn, 인간/합성 원문 25/25, UPDATE/NO_OP 10/40, ADD/REPLACE 7/3,
  활성 slot·Tool 7개를 확인했다.
- driver temperature, front trunk, steering-wheel heat, circulation, driver window,
  fan, parked-door access의 7개 활성 설정이 외부 원문의 동작과 대응한다.
- `Vent windows→Affirmative` 원문에는 15%가 없는데 all-window 15% 지속값을 만들었던
  문제를 확인해 해당 마지막 Turn을 원문 그대로 NO_OP로 되돌렸다.
- 독립적인 rear-right temperature ADD 대신 외부 rain/driver-window-close 레코드를
  사용해 기존 normal driver-window slot의 REPLACE로 연결했다.
- driver-window는 20→0→55→90의 세 REPLACE로 구성되며 Tool, selector와 normal-driving
  condition이 모든 단계에서 동일하다.
- Turn/Final Quiz를 4/6, Gold Tool call 11개로 재구성하고 7개 최종 활성 slot을 모두
  평가하도록 했다. 기본값인 door-unlocked 호출은 fan 설정과 결합해 simulator에서도
  유효한 목표 상태를 만든다.
- 외부 원문, cutoff evidence, 최종 활성값을 대조하고 공식 schema, VehicleWorld,
  Patch replay와 NO_OP identity를 포함한 8개 자동 검증을 통과했다.

외부 원문 중 25 Turn만 인간 작성이고 조립·지속성 문장·Gold label·Quiz는 모델
작성이다. 따라서 실제 human-authored test로 간주하지 않는다.
