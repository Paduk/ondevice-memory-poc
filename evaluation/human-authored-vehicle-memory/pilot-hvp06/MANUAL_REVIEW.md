# HVP06 manual review

검토 결과: **external-adapted pilot 용도로 PASS, human-authored test 용도로는 제외**.

## 한 턴씩 확인한 결과

- 항공·호텔 예약, 대중교통 경로, 휴게소 검색과 당일 교통정보는 VehicleMemBench의
  지속 차량 설정 범위 밖이거나 일회성 요청이므로 NO_OP가 적절하다.
- 도어 잠금 및 Sentry 상태를 묻는 발화는 현재 상태 조회이므로 NO_OP가 맞다.
- 목적지, defrost, circulation, rear trunk의 Tool 매핑은 외부 Tool 의미와 직접
  대응한다.
- 목적지, defrost와 circulation은 각각 같은 slot 안에서 과거 값이 사라지고 최신 값만
  남는다.
- Turn Quiz는 갱신 전 값을, Final Quiz는 모든 REPLACE 이후 값을 사용한다.

## 수작업 검토에서 수정한 부분

- 원문에 없던 `heavy traffic` 조건을 제거하고, circulation을 일반적인 cabin-air
  preference로 다시 정의했다.
- `Baker Street → Home`을 부자연스러운 주말 목적지 변경이 아니라, 목적지를 생략했을
  때 사용하는 default destination 변경으로 수정했다.
- rear trunk 조건을 `급할 때`에서 `짐을 실을 때`로 정리해 Quiz 조건과 일치시켰다.
- 반복되던 `save`, `keep`, `replace` 문구를 평소 행동과 선호 변화가 드러나는 자연스러운
  문장으로 교체했다.
- Quiz 문장에서 `remembered/latest` 같은 내부 memory 표현을 줄였다.

## 남은 한계

- 여러 원문 작성자를 하나의 운전자 `Riley`로 정규화해 세션별 말투와 assistant 이름이
  일관되지 않다.
- 인간 작성 NO_OP 대화와 차량 UPDATE 대화의 출처가 분리되어 있어, 한 대화 안에서
  자연스럽게 섞인 실제 인간 대화는 아니다.
- 세 correction은 한 발화짜리 세션이고, 최종 활성 slot도 4개로 다양성이 작다.

따라서 외부 표현에 대한 전이 확인에는 사용할 수 있지만, human validation의 직접적인
근거로 사용해서는 안 된다.
