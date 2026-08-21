# Compact A-MEM-style 개념 및 구현 계획

상태: 4회차 구현·Scenario 6 검증 완료

## 1. 짧은 개념

`Compact A-MEM-style`은 A-MEM의 structured note, semantic link, memory
evolution 개념을 유지하면서 on-device 자원에 맞게 기억 생성량과 LLM 호출을
줄이는 resource-matched baseline이다.

```text
History
→ generic relevance filtering + episode compaction
→ durable note construction
→ deterministic embedding link
→ correction/conflict일 때만 evolution
→ 기존 linked retrieval → 동일 Agent·Tool·scorer 평가
```

기존 `cloud_amem_style`은 history line마다 note를 생성하는 비교 기준으로 그대로
보존한다. Compact 버전은 여러 연속 대화를 하나의 episode로 묶고, 잡담·중복을
제거한 뒤 장기적으로 유효한 내용만 note로 만든다.

## 2. 방법론 경계

- profile 이름: `cloud_compact_amem_style`
- 기존 `cloud_amem`과 `cloud_amem_style`의 코드·cache·결과는 변경하지 않는다.
- Ours의 Vehicle Tool Schema, 전용 Fact ontology, Gold 정보는 사용하지 않는다.
- compaction은 범용 대화 규칙과 source evidence만 사용한다.
- retrieval 이후 Agent, Tool registry, Arguments와 scorer는 다른 baseline과 같다.
- 논문 A-MEM 재현이 아니라 **Compact A-MEM-style adaptation**으로 보고한다.

## 3. 구현 계획

### 1회차 — Compaction 계약

- [완료] chronological history를 bounded episode로 묶는다.
- [완료] durable preference, correction, condition만 남기는 strict JSON 응답을
  정의한다.
- [완료] 각 compact note에 원본 source line ID를 보존한다.
- [완료] Fake provider와 filtering/episode 단위 테스트를 추가한다.

확정 계약:

- episode 기본 상한: 16 entries, 8,000 characters, source 간격 6시간
- 한 episode compaction 호출이 0~16개 note를 직접 생성
- note 필드: content, context, keywords, tags, memory kind, source IDs
- source IDs는 해당 episode 내부의 chronological ID만 허용
- empty notes는 정상 filtering 결과로 허용
- OpenAI 입력은 PII redaction하며 history text를 untrusted data로 취급
- Vehicle Tool Schema와 Ours Fact ontology는 provider 입력에 포함하지 않음

### 2회차 — Compact graph 생성

- [완료] compact note를 기존 A-MEM note/version·embedding 저장 경로에
  연결한다. Compaction 결과가 metadata를 포함하므로 note별 construction LLM은
  다시 호출하지 않는다.
- [완료] link는 embedding similarity로 deterministic하게 생성한다.
- [완료] `correction`으로 분류되고 threshold 이상 후보가 있을 때만 LLM
  evolution을 실행한다.
- [완료] 전용 session, compact graph fingerprint와 episode 단위 resume 경로를
  추가한다.

확정 계약:

- episode compaction 전체를 atomic commit하며 실패 시 note/link가 남지 않음
- 완료 episode의 source ID가 같으면 재실행 시 provider 호출 없이 skip
- compact note와 모든 source line의 다대다 provenance를 SQLite에 보존
- link 기본 threshold `0.75`, 최대 후보 5개, decision `compact_embedding`
- 일반 A-MEM note가 있는 session에는 compact graph를 혼합하지 않음
- compaction·embedding·evolution usage를 기존 provider trace에 함께 기록

### 3회차 — VehicleMemBench 통합

- [완료] `cloud_compact_amem_style` profile과 CLI 설정을 추가했다.
- [완료] 전용 session에서 기존 A-MEM linked retrieval을 재사용한다.
- [완료] manifest에 raw/selected line 수, episode 수, compact note 수,
  압축률, evolution/link 수와 provider usage·비용을 기록한다.
- [완료] Fake Scenario E2E, privacy/no-Gold-leak와 episode 단위 cache resume를
  검증한다.

확정 CLI 설정:

- `--compact-amem-episode-max-entries` (기본 16)
- `--compact-amem-episode-max-chars` (기본 8,000)
- `--compact-amem-episode-max-gap-seconds` (기본 21,600)
- `--compact-amem-link-threshold` (기본 0.75)
- `--amem-note-limit`은 compact profile에서도 debug용 partial-history 제한으로만
  사용

검증 결과: Ruff 전체 통과, Ubuntu 테스트 300개 수집 중 299 passed·1 skipped.

### 4회차 — Scenario 6 검증

- [완료] 100-line smoke에서 7 episode, 29 note, 7 compaction call과
  privacy/retrieval 경로를 확인했다.
- [완료] Scenario 6 전체 2,698줄을 190 episode와 522 note로 구축하고 동일
  cache에서 10-task E2E를 실행했다.
- [완료] ESM `0.60`, State/Value F1 `0.776`, Tool F1 `0.633`, retrieval
  recall@k `0.75`를 확인했다.
- [완료] 모델이 evidence ID 집합을 비순차 배열로 반환하는 실전 오류를 발견해,
  episode 밖·중복 ID는 거부하면서 episode 순서로만 canonicalize하도록 graph
  policy v2와 회귀 테스트를 추가했다.
- [제한] 기존 `cloud_amem_style` Scenario 6 실측 artifact는 없다. 따라서 정확도
  직접 비교는 하지 않았고 dry-run 호출량과 비교했다. Compact의 정상
  compaction 190회는 line별 방식의 최소 2,698회 대비 92.96%, 보수적 최대
  5,395회 대비 96.48% 적다.

상세 결과와 재현 경로는
[Compact A-MEM-style Scenario 6 결과](compact-amem-style-s6-results.md)에
기록한다.

## 4. 완료 기준

- 기존 A-MEM profile의 결과와 cache key가 변하지 않는다.
- compact note가 모든 source evidence로 역추적된다.
- 생성 호출이 history line 수가 아니라 compact episode 수에 비례한다.
- Ours 전용 schema 없이 동일 VehicleMemBench 평가가 완료된다.
- 정확도와 함께 압축률·호출 수·token·latency를 결과표에 보고한다.
