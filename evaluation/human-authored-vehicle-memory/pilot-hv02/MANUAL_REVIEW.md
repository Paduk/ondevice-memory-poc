# HVP02 manual-style review

검토 결과: **HVP03–HVP20과 같은 external-adapted pilot 규격으로 PASS,
human-authored test 용도로는 제외**.

- 50 Turn, 인간/합성 원문 25/25, UPDATE/NO_OP 10/40, ADD/REPLACE 7/3,
  활성 slot·Tool 7개를 확인했다.
- 음악 볼륨, rear-right 온도, 조수석 환기, 프런트·리어 트렁크, 도어 잠금,
  내비게이션 음성의 7개 활성 설정을 원문 및 지속성 문장과 대조했다.
- REPLACE 세 건은 모두 같은 음악 볼륨 slot에서 52→25→70→5로 진행되며,
  Tool attribute와 condition을 바꾸지 않는다.
- 조수석 발화의 heat-off는 해당 운행에만 적용하고, 명시적으로 지속성을 부여한
  ventilation speed 1만 memory에 남겼다.
- Turn/Final Quiz 4/6은 각 cutoff의 활성 memory만 사용한다. 초기 상태와 같은
  리어 트렁크 닫힘은 rear-right 온도와 결합해 simulator에서도 실행 결과를 검증했다.
- 외부 원문, cutoff evidence, 최종 활성값을 대조하고 공식 schema, VehicleWorld,
  Patch replay와 NO_OP identity를 포함한 8개 자동 검증을 통과했다.

외부 원문 중 25 Turn만 인간 작성이고 조립·지속성 문장·Gold label·Quiz는 모델
작성이다. 따라서 실제 human-authored test로 간주하지 않는다. 이전 모델 작성 HVP02는
`model-authored-v0/`에 보존했다.
