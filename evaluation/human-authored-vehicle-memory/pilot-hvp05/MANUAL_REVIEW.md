# HVP05 manual review

검토 결과: **external-adapted pilot 용도로 PASS, human-authored test 용도로는 제외**.

## 한 턴씩 확인한 결과

- S1의 여행 예산·시기, S2의 당일 경로, S3의 `work location today`, S4의 영화
  조회는 모두 해당 세션의 일회성 정보이거나 VehicleMemBench Tool 범위 밖이므로 NO_OP가
  적절하다.
- S5의 배터리 상태와 S6의 에너지 소비량은 현재 상태 조회이므로 지속 memory에 저장하지
  않는 것이 맞다.
- 음악 볼륨 30→75와 평일 목적지 `123 Main St`→`Work`는 동일 slot의 시간순
  REPLACE로 일관된다.
- 도어 잠금, front trunk, sunroof 매핑은 외부 Tool 의미와 VehicleMemBench Tool 의미가
  직접 대응한다.
- 네 Turn Quiz는 각 cutoff 시점의 당시 값을 사용하고, 여섯 Final Quiz는 최종 활성 값만
  사용한다.

## 수작업 검토에서 수정한 부분

- 원문과 추가 문장 사이의 구두점 누락을 수정했다.
- 모든 UPDATE에 반복되던 `remember`, `replace`, `keep` 문구를 줄이고, 평소 행동이나
  현재 선호가 자연스럽게 드러나도록 표현을 다양화했다.
- Quiz의 `currently remembered`, `latest destination` 같은 기계적인 표현을 실제 요청에
  가까운 문장으로 바꿨다.

## 남은 한계

- 서로 다른 원문 작성자를 하나의 운전자 `Casey`로 정규화했기 때문에 세션별 말투와
  assistant 호출명이 달라진다.
- user–assistant 대화라서 기존의 인간–인간 V2 대화와 speaker 분포가 다르다.
- correction source는 한 발화짜리 세션이므로 실제 장문 대화보다 변화가 명시적이다.
- 최종 활성 slot이 5개여서 HVP01/HVP02보다 memory 다양성이 작다.

따라서 현재 버전은 외부 언어 전이 및 형식 검증에는 사용할 수 있지만, 인간 검증셋에
넣으려면 사람이 전체 세션을 하나의 일관된 화자와 상황으로 다시 작성해야 한다.
