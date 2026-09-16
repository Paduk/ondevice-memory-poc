# HVP01 manual-style review

검토 결과: **HVP03–HVP20과 같은 external-adapted pilot 규격으로 PASS,
human-authored test 용도로는 제외**.

- 50 Turn, 인간/합성 원문 25/25, UPDATE/NO_OP 10/40, ADD/REPLACE 7/3,
  활성 slot·Tool 7개를 확인했다.
- 좌석 환기, 음악 볼륨, 프런트 트렁크, 공조 모드, 창문, 순환 모드,
  리어 트렁크의 7개 활성 설정이 외부 원문 동작 및 추가된 지속성 문장과 대응한다.
- REPLACE는 음악 볼륨 11→58, 공조 모드 auto→defrost, 순환 모드
  inside→outside이며 동일 slot, Tool attribute, selector, condition을 유지한다.
- 원문상 단발성 명령은 추가 문장에서 반복 조건을 명시한 경우에만 UPDATE로 두고,
  나머지 40 Turn은 NO_OP로 유지했다.
- Turn/Final Quiz 4/6은 각 cutoff의 활성 memory만 사용한다. 초기 상태와 같은
  창문 닫힘은 프런트 트렁크 열기와 결합해 simulator에서도 실행 결과를 검증했다.
- 외부 원문, cutoff evidence, 최종 활성값을 대조하고 공식 schema, VehicleWorld,
  Patch replay와 NO_OP identity를 포함한 8개 자동 검증을 통과했다.

외부 원문 중 25 Turn만 인간 작성이고 조립·지속성 문장·Gold label·Quiz는 모델
작성이다. 따라서 실제 human-authored test로 간주하지 않는다. 이전 모델 작성 HVP01은
`model-authored-v0/`에 보존했다.
