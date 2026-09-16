# HVP07 manual review

검토 결과: **external-adapted pilot 용도로 PASS, human-authored test 용도로는 제외**.

## 한 턴씩 확인한 결과

- go-karting 예약, 대중교통 지연 알림과 판매점 검색은 일회성 외부 업무이므로 NO_OP가
  적절하다.
- 소프트웨어 업데이트와 차량 정보 요청도 현재 요청·조회이며, 지원되는 지속 차량 설정이
  아니므로 NO_OP가 맞다.
- navigation voice mute, sunroof, circulation, music playback과 front trunk는 외부
  Tool에서 VehicleMemBench Tool로 의미 손실 없이 매핑된다.
- sunroof 50→18과 circulation inside→outside는 동일 slot의 시간순 REPLACE다.
- 네 Turn Quiz는 두 REPLACE 이전 값을 포함하고, Final Quiz는 수정된 최종 값만 사용한다.

## 수작업 검토에서 수정한 부분

- `bad odor` 조건에서 inside→outside로 바뀌어 비상식적이던 circulation Quiz를 발견했다.
  이를 특정 악취 조건이 아니라 평상시 cabin-air preference 변화로 재정의했다.
- 음악 source에 없던 `trip start` 조건을 제거하고, 원문에 실제로 등장하는 `playlist를
  선택한 경우`로 조건을 좁혔다.
- 반복적인 `keep/replace/remembered/latest` 표현을 평소 행동과 선호 변화 중심의
  자연스러운 문장으로 바꿨다.
- 원문과 추가 문장의 중복 표현을 줄이고 Quiz도 실제 사용 요청에 가깝게 수정했다.

## 남은 한계

- 여러 원문 작성자를 운전자 `Morgan` 한 명으로 합쳐 말투와 assistant 호출명이
  세션마다 달라진다.
- 인간 작성 25 Turn은 자연스러운 NO_OP 문맥이고, 실제 UPDATE Turn은 Audio2Tool의
  합성 query에서 왔다.
- correction source 세 개가 한 발화 세션이어서 실제 인간의 장기적 선호 변화보다
  명시적이다.

따라서 HVP07은 외부 표현과 correction에 대한 전이 파일럿으로는 사용할 수 있지만,
human validation 결과로 직접 보고해서는 안 된다.
