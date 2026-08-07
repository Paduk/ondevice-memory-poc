# Ours × Recursive 결합 실험 계획

상태: `실험 완료`

작성일: 2026-07-30

## 목표

Scenario 6–10에서 Ours의 ESM `0.50`을 Recursive `0.62` 이상으로 개선할
가능성을 두 방법으로 검증한다. 기존 `cloud_fact_patch`와
`cloud_recursive_summary`의 구현·cache·결과는 변경하지 않는다.

## 1단계 — Ours + Recursive Hybrid

- 새 profile: `cloud_fact_recursive_hybrid`
- 질의별 Ours Fact Top-k와 query-independent 최종 Recursive Summary를
  출처가 구분된 하나의 Agent context로 제공한다.
- Ours의 Tool selector·execution hint는 유지한다.
- 같은 사실이 충돌하면 Agent가 출처를 볼 수 있게 하되 자동으로 값을
  덮어쓰지 않는다.
- Scenario 6 smoke 후 Scenario 6–10 50 task를 평가한다.

판정: ESM, State/Tool F1, Argument exact, Ours·Recursive 대비 taskwise
승패, context token 증가와 오류 유형을 보고한다.

결과: 원본 Ours Fact cache를 고정한 Scenario 6–10 50 task에서 ESM
`0.60`(30/50), State F1 `0.798`, Tool F1 `0.646`, Argument exact `0.52`.
Ours 대비 `+0.10`, Recursive 대비 `-0.02`다. Taskwise로 Ours 실패 8건을
복구하고 성공 3건을 잃었다. 비교 run ID는
`0500a4ff-48e1-4335-b775-142198afc45c`다.

## 2단계 — Recursive-assisted Ours

- 새 profile: `cloud_recursive_assisted_fact_patch`
- Recursive Summary는 Agent에게 직접 주입하지 않고 누락 Fact 후보 탐지에만
  사용한다.
- 후보는 반드시 원본 History evidence와 연결한 뒤 기존 Ours
  structure·validation·versioned DB·retrieval·binding 경로로 저장한다.
- Summary-only 추론이나 근거 없는 Fact는 저장하지 않는다.
- 기존 Ours 추출 후 보조 추출을 수행하고, 같은 identity의 중복 후보는
  제거한다. 따라서 Memory 생성 호출 비용은 Ours보다 커진다.

판정: Fact coverage `23/50→34/50 이상`을 1차 목표로 하고, Scenario 6
smoke 후 Scenario 6–10 50 task에서 ESM `0.62 이상`을 확인한다.

결과: 원본 Ours Fact DB를 seed로 고정한 run
`24dcf1f2-40af-48da-b893-420e4404525c`에서 ESM `0.60`, State F1
`0.728`, Tool F1 `0.573`, Argument exact `0.44`다. Ours보다 ESM
`+0.10`이지만 Recursive보다 `-0.02`이며 Hybrid와 같다.

- active Fact `47→79`, retrieval recall `0.49→0.603`
- 보조 후보 46건: 적용 36, NOOP 2, reject 2, review 6
- reviewed 정답 Fact 기준 엄격 coverage는 `23/50` 그대로다. 추가 Fact가
  정답 값과 유사해도 entity·predicate·condition 정규화가 달라 누락 27건의
  정답 record를 직접 복구하지 못했다.
- Ours 대비 taskwise 9건 복구, 4건 회귀했다.
- Fact extraction 호출 `449→898`로 증가해 성능 대비 비용 효율은 낮다.

Coverage audit는 frozen Retrieval·Stage Fact·Gate annotation을 기준으로
각 정답 identity를 assisted DB의 기존 linker에 read-only 대조했다. 기존
정답 23 task는 모두 유지됐지만, Stage Fact 26 task는 모두 여전히 `ADD`
판정이었고 gate-only 1 task도 active로 복구되지 않았다.

## 실행 순서와 중단 조건

1. Hybrid fixture·privacy·cache 불변성 테스트
2. Hybrid Scenario 6 smoke → 이상 없으면 50 task
3. 결과 고정 후 Recursive-assisted extraction 구현
4. assisted fixture·evidence 검증·smoke → 이상 없으면 50 task

최종 판단: Hybrid는 저비용 context 결합으로 유효한 +0.10 개선을 보였고,
assisted 방식은 Fact 수와 retrieval을 늘렸지만 구조 정규화와 binding
병목을 해결하지 못했다. 다음 개선은 Summary 후보를 더 생성하는 것보다
person identity, canonical predicate, applicability를 Ours schema로
정규화하고 검색 시 중복·충돌을 억제하는 데 우선순위를 둔다.

Gold Memory·Tool·argument는 생성이나 Agent 입력에 사용하지 않는다.
완료율 저하, raw History 유출, cache 오염, residual sensitive span이 있으면
전체 평가 전에 중단한다.
