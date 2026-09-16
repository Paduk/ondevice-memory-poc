# HVP08 manual-style review

검토 결과: **external-adapted pilot 용도로 PASS, human-authored test 용도로는 제외**.

- 50 Turn, 인간/합성 원문 25/25, UPDATE/NO_OP 10/40을 확인했다.
- 온도, 창문, 목적지, 순환, 조명, 트렁크, 도어 잠금 7개 slot이 근거 원문과 대응한다.
- 창문 60→10, 목적지 Riverside Dog Park→Eastside Library, 주차 시 lock→출발 시
  unlock은 시간순 REPLACE다.
- 음식 냄새 때문에 선택한 내기순환을 평상시 설정으로 일반화하지 않고 동일 조건에 묶었다.
- 밝기 50을 누락하던 조명 record와, 아들의 좌석 냉방을 Avery에게 잘못 귀속하던 record를
  각각 단일 색상 record와 명확한 잠금/해제 record로 교체했다.
- 일반적인 factory 문장이던 Quiz 10개를 실제 activation condition이 드러나도록 다시 썼다.
- Turn Quiz는 REPLACE 이전 상태를, Final Quiz는 최종 상태를 사용한다.

여러 외부 화자를 Avery 한 명으로 정규화했고 UPDATE 원문은 합성 데이터이므로, 실제
human validation 결과로 보고해서는 안 된다.
