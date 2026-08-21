# Turn-wise Patch vs Summary 설계·구현·평가 계획

상태: `코어 구현 및 로컬 검증 완료`

## 1. 목표와 방법

날짜 경계를 제거하고, 시간순 history entry 한 줄을 하나의 turn으로 처리한다.
Quiz는 우선 기존 V1처럼 전체 turn 처리 후 실행한다.

| Profile | 매 turn 동작 |
| --- | --- |
| `cloud_turnwise_recursive_summary` | `현재 Summary + 새 turn → 전체 Summary rewrite 또는 no-op` |
| `cloud_turnwise_recursive_summary_patch` | `현재 Summary + 새 turn → 최소 Patch 또는 no-op`, Patch는 로컬에서 deterministic 적용 |

두 방식은 같은 모델 snapshot, 보존 정책, turn 순서를 사용한다. 차이는 update
표현이 전체 rewrite인지 최소 Patch인지뿐이다.

## 2. 구현 범위

1. `build_turn_history_batches()`를 추가해 entry별 batch를 만들고 기존
   Recursive Summary engine/storage를 재사용한다.
2. 날짜 표현을 제거한 Turn-wise 전용 prompt/version과 두 profile을 추가한다.
   `update_cadence=history_entry`를 cache signature에 넣어 기존 날짜별 cache와
   완전히 분리한다.
3. turn별 trace와 profile 집계를 추가한다.
   - 전체 turn, no-op, effective update 수·비율
   - 실패·repair·중복 update
   - Patch `add/replace/delete` 수·비율
   - no-op/update별 input·output tokens와 latency
   - 최종 Summary 크기와 hash
4. batch 생성, 순서 보존, no-op/update, deterministic Patch, cache 격리,
   집계 및 기존 baseline 비회귀 테스트를 추가한다.

구현된 profile은 CLI에서 각각 단독 실행한다. 두 profile은 같은
Recursive Summary engine을 사용하므로, cache 오염을 방지하기 위해 한
프로세스에서 동시 선택하지 않는다.

판정은 `no-op = Tool 호출 없음 + memory 불변`, `update = 유효 호출 + memory
hash 변경`으로 한다. 오류와 동일 내용을 다시 쓴 redundant update는 별도로
기록해 비율을 왜곡하지 않는다.

## 3. 평가 순서

1. fixture 및 실제 history 50-turn smoke test
2. Scenario 6에서 두 방식 각 1회 실행·semantic trace 확인
3. Scenario 1–10에서 각 1회 fresh 실행 후 ESM, State/Tool F1, Arg Exact,
   memory coverage, tokens, latency 및 update 분포 비교
4. 결과가 반복 편차와 구별되지 않을 때만 각 방식 1회 추가 실행

Scenario 1–10은 총 `27,303 turns`이므로 전체 1회 비교도 memory 호출
`54,606회`가 필요하다. turn은 scenario 안에서 순차 처리하되 scenario끼리는
병렬 실행한다.

## 4. 예상 결과

Patch는 rewrite보다 update output tokens와 decode latency가 작을 것으로
예상한다. 최종 정확도는 유사할 수 있지만, multi-turn 의미 누락과 반복
오갱신·오염 여부는 no-op/update trace와 실패 task를 함께 봐야 판단할 수 있다.
향후 V2에서는 중간 Quiz를 삽입해 Turn-wise 최신성의 실제 이점을 평가한다.
