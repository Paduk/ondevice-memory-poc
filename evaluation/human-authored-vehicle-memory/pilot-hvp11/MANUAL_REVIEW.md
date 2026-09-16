# HVP11 manual-style review

검토 결과: **external-adapted pilot 용도로 PASS, human-authored test 용도로는 제외**.

- 50 Turn, 인간/합성 원문 25/25, UPDATE/NO_OP 10/40을 확인했다.
- 좌석 열선, 온도, 조명, 안내 음성, 팬, 도어 잠금, 기본 목적지 7개 slot을 확인했다.
- fan 6→8, 주차 시 lock→출발 시 unlock, 기본 목적지 Home→Eastside Library는
  Tool 속성이 일치하는 시간순 REPLACE다.
- 아들의 뒷좌석 냉방을 Quinn에게 귀속하고 level을 on/off로 축약하던 pair를 제거했다.
- 서로 다른 조건이던 햇빛 차단용 sunroof 15와 소음 감소용 sunroof 0도 REPLACE에서
  제거했다.
- 밝기 100을 누락하던 white 조명 replacement를 제거하여 blue 색상만 보존했다.
- 아파트 음악 대화는 in-car 문맥으로도 자연스러운 3턴 cinema 대화로 교체했다.
- Quiz 10개를 실제 activation condition이 드러나게 다시 쓰고 모두 실행되는지 확인했다.

인간 작성 25 Turn은 자연스러운 NO_OP 문맥이고 UPDATE Turn은 Audio2Tool 합성 query다.
