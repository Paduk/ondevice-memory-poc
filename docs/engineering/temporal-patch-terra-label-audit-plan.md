# Temporal Patch Terra Label Audit 계획

## 목적

기존 Patch의 `decision/op/target/content`와 메모리 trajectory는 고정하고,
T1–T20의 `identity_key/temporal_action/temporal_cue`만 검수해 Temporal Patch
학습 라벨의 신뢰도를 높인다. 기존 deterministic 데이터는 덮어쓰지 않는다.

## 검수 순서

1. **기존 Terra plan 대조:** 각 T scenario의 `temporal-plan.json`에 있는
   `(source_event_id, source_update_index) -> action/cue`를 변환된 Patch와 매칭한다.
2. **결정론적 교정:** plan과 정확히 매칭되는 라벨은 plan 기준으로 교정한다.
   이는 이미 Terra가 생성한 416개 transition을 재사용하므로 추가 호출이 없다.
3. **Terra escalation:** plan 누락, 중복, operation 비호환처럼 자동 확정할 수 없는
   건만 `gpt-5.6-terra`가 `ACCEPT/CORRECT/DEFER`로 검수한다. Terra에도
   `op/target/content` 수정 권한은 주지 않는다.
4. **보수적 처리:** `DEFER`는 기존 라벨을 유지하되 학습 채택 대상과 분리해 기록한다.

Terra 입력은 현재 turn, 직전 memory, 고정 Patch, temporal plan/anchor, 기존 temporal
라벨이다. 출력은 verdict, 세 temporal 필드, 한 문장 reason으로 제한한다.

## 필수 gate

- temporary/end cue는 현재 turn의 exact substring이어야 한다.
- action과 `add/replace/delete` 조합이 Temporal Patch schema와 호환돼야 한다.
- 적용 전후 `op/target/content`, turn 수, UPDATE 수가 동일해야 한다.
- 전체 replay의 turn별/final memory SHA-256이 원본 Patch와 동일해야 한다.
- `ACCEPT/CORRECT/DEFER`, 교정 전후 action 분포와 토큰 사용량을 report로 남긴다.

## 산출물

- 원본: `...temporal-patch-v1` 유지
- 감사 기록: scenario/turn/operation별 JSONL과 요약 JSON
- 교정 데이터: 별도 `...temporal-patch-terra-audited-v2`
- 교정 데이터의 `catalog.sqlite` 재생성 후 학습 스크립트는 검증 완료된 v2를 사용

## 실행 단계

1. plan-first mapper와 strict schema/gate 구현
2. 미해결 건 Terra resumable audit 구현
3. audited dataset materialize 및 replay/hash 검증
4. T1 canary 후 T1–T20 전체 report 확인
5. 이상 없을 때만 Temporal Patch 학습 시작

## 구현 및 감사 결과 (2026-08-25)

- Terra plan transition 416/416건 매칭, live Terra 추가 호출 0건
- `ACCEPT` 306건, `CORRECT` 110건, `DEFER` 0건
- action 분포는 `durable 240 / conditional 64 / current 48 / temporary 32 /
  end-temporary 32`로 교정됨
- 335,504 turn 전체 replay 및 120개 scenario final-memory hash 검증 통과
- audited catalog 335,504 rows 생성 완료
