# HVP20 manual-style review

검토 결과: **external-adapted 구조 파일럿 용도로 PASS, 독립적인 텍스트 다양성 및
human-authored test 용도로는 제외**.

- 50 Turn을 순서대로 읽고 인간/합성 원문 25/25, UPDATE/NO_OP 10/40을 확인했다.
- 당일 영화 예약, 취침용 음악·난방 조절, 요리 중 음악 재생, 수면 조회, 즉시 glovebox
  잠금, cruise speed 65 설정은 지속 차량 선호가 아닌 일회성 대화·실행이므로 NO_OP가
  적절하다.
- 온도, 조명, 기본 목적지, navigation voice, rear defrost, 순환 모드, 도어 잠금의
  7개 활성 slot이 외부 원문의 정답 호출과 대응한다.
- `setAmbientLighting(white, 100)`에서 색상만 지속값으로 두고 물건을 찾기 위한 밝기는
  일회성으로 분리했다. `setClimate`에서도 온도만 지속화하고 COOL/HEAT/AUTO/MIN mode는
  각 발화의 일회성 속성으로 유지했다.
- `front=false, rear=true` defrost는 `zone=rear`, recirculation disabled는
  `circulation=outside`, corrected unlock은 `door=all, locked=false`로 의미 보존
  정규화했다.
- 세 REPLACE는 모두 정상 cabin temperature의 18→27→24→21 체인이다. 동일 slot,
  Tool, `zone=all`, activation condition을 유지하고 값만 시간순으로 변경한다.
- Turn Quiz 4개와 Final Quiz 6개의 cutoff, evidence slot, 조건 및 최종값을 대조했다.
  마지막 복합 Quiz는 stale-air headache와 parked 조건을 모두 명시한다.
- 공식 Tool JSON schema, VehicleWorld 실행, Patch replay, NO_OP identity, Quiz cutoff를
  포함한 자동 검증을 모두 통과했다.

## 텍스트 중복 제한

Audio2Tool은 같은 query를 서로 다른 화자가 녹음한 복수 record를 제공한다. HVP20의
UPDATE record ID와 음원은 HVP03–HVP19와 겹치지 않지만, text-only `original_text`는
앞선 시나리오에 사용된 paired record와 중복된다. 이는 상태 전이와 파이프라인을 검증하는
external-adapted pilot에는 허용하지만, HVP20을 독립적인 언어 다양성 표본으로 세거나
성능 표본 수를 늘리는 근거로 사용해서는 안 된다.

외부 원문의 문법과 말투는 source trace에 그대로 보존했다. 지속성 문장, memory label,
Quiz가 모델 작성이므로 실제 human-authored test로 간주하지 않는다.
