# HVP03 manual-style review

검토 결과: **HVP05–HVP20과 같은 external-adapted pilot 규격으로 PASS,
human-authored test 용도로는 제외**.

- 50 Turn, 인간/합성 원문 25/25, UPDATE/NO_OP 10/40, ADD/REPLACE 7/3,
  활성 slot·Tool 7개를 확인했다.
- temperature, fan, circulation, window, seat ventilation, music volume,
  steering-wheel heat의 7개 활성 설정이 외부 원문의 동작과 대응한다.
- steering-wheel heat의 ADD 조건이 `normal driving`, REPLACE 조건이 `cold morning`으로
  달랐던 문제를 수정해 두 단계 모두 cold-morning 조건으로 통일했다.
- REPLACE는 fan 5→3, passenger-seat ventilation 3→2, steering-wheel heat
  false→true이며 각각 동일 slot, Tool, selector, condition을 유지한다.
- Turn Quiz의 불필요한 이중 호출을 하나로 줄여 Turn/Final Quiz 4/6,
  Gold Tool call 11개로 표준화했다.
- 외부 원문, cutoff evidence, 최종 활성값을 대조하고 공식 schema, VehicleWorld,
  Patch replay와 NO_OP identity를 포함한 8개 자동 검증을 통과했다.

외부 원문 중 25 Turn만 인간 작성이고 조립·지속성 문장·Gold label·Quiz는 모델
작성이다. 따라서 실제 human-authored test로 간주하지 않는다.
