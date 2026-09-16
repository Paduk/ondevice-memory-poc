# HVP10 manual-style review

검토 결과: **external-adapted pilot 용도로 PASS, human-authored test 용도로는 제외**.

- 50 Turn, 인간/합성 원문 25/25, UPDATE/NO_OP 10/40을 확인했다.
- 온도, 팬, 핸들 열선, 두 창문, 선루프, rear defrost 7개 slot이 원문과 대응한다.
- 팬 1→7, 선루프 10→30, passenger-front window 60→10은 동일 slot의 시간순
  REPLACE다.
- rear-only와 front-only defrost는 서로 다른 가시성 조건이므로 REPLACE로 묶지 않았다.
- 자녀가 답답하다고 한 rear-right window 설정을 Taylor 본인이 아닌 자녀 상황에 귀속했다.
- 핸들 열선 조건을 포괄적인 cold drive에서 원문의 `hands feel cold`로 좁혔다.
- `[insert name]`, `answers`, `books tickets`가 포함된 미완성 인간 원문을 자연스러운
  7턴 cinema record로 교체했다.
- 모든 정수값과 zone을 VehicleMemBench Tool schema 및 VehicleWorld에서 재생했다.
- Quiz 10개를 실제 activation condition이 드러나게 다시 쓰고 evidence cutoff와 최종
  활성값을 확인했다.

외부 표현 전이 검사용 파일럿이며, 인간 작성 UPDATE 대화로 해석하면 안 된다.
