# VehicleMemBench V2 Turn-wise Gold Memory 계획

세부 실행 계획: [VehicleMemBench V2 Consensus Gold 실행 계획](../engineering/vehiclemembench-v2-consensus-gold-execution-plan.md)

## V2 기본 단위와 입력

V2는 V1과 동일한 입력으로 매 turn의 신뢰 가능한 Gold Memory를 만든다.

```text
Luna 생성 입력 (V1과 동일)
- Current Memory: 확정된 M(t-1)
- New Conversation Turn: turn t
- V2 출력 형식을 지정하는 instruction만 추가

Luna 생성 출력
- Combined temporal patch operations + system이 계산한 Candidate M'(t) + reason
- 30-add compaction turn은 post-patch memory의 rewrite 후보 3개 추가 생성

검증 완료 출력
- 단일 Gold Memory M(t) + Combined Update Record + 검증 metadata
- 동일 M(t)를 Summary complete rewrite target으로 export
```

QA `gold_memory`, query, gold answer, 기존 trace와 미래 turn은 Luna, validator,
critic에 제공하지 않는다. 공식 QA는 scenario 생성 완료 후 외부 평가에만 사용한다.

## 생성 파이프라인

각 scenario는 첫 turn부터 순서대로 처리한다.

```text
V1과 동일한 입력 + V2 출력 instruction
                    ↓
 Luna Combined temporal patch 후보 3개
                    ↓
         Semantic Consensus
                    ↓
        Deterministic Validator
                    ↓
   PASS → 필요 시 30-add compaction 후보 3개
   FAIL → Terra → SOL → Human Review  ← 검증 정보만 사용
                    ↓
              단일 Gold M(t)
          ↙                     ↘
 Combined update record     Summary rewrite target
                    ↓
              turn t+1 진행
```

세 Luna 후보는 반드시 동일한 `M(t-1)`과 turn을 사용한다. Consensus는 문장
일치가 아니라 다음 의미 필드의 일치로 판정한다.

```text
operation + subject + attribute + normalized value/unit
+ condition/temporal scope + supersedes/delete target
+ evidence support + resulting canonical memory state
```

`identity_key`는 내부 추적용 이름이므로 consensus 의미 비교에서 제외한다. 그 외
memory text/value, 대상, operation, temporal scope 또는 적용 결과가 다르면 동일 후보로
보지 않는다.

세 후보는 **같은 model·prompt·schema·sampling 설정의 독립된 Responses 호출**로
생성한다. 현재는 API `seed`를 전제하지 않으며 후보별 prompt도 바꾸지 않는다.
서로 다른 prompt는 annotator policy 자체를 달라지게 하므로 별도 robustness 실험으로
분리한다.

- 3/3 의미 일치, 또는 정규화 가능한 형식 차이만 있으면 validator로 진행
- 2/3이 같아도 나머지 후보와 의미가 다르면 다수 후보를 preferred로 두고 Terra 검토
- 세 후보가 모두 다르면 Terra 검토, 해결되지 않으면 SOL과 Human Review로 진행

30-add compaction은 별도 경로를 사용한다. rewrite 후보가 3/3으로 같으면 대표 후보를
채택하고, 하나라도 다르면 Terra를 건너뛰고 SOL이 `SELECT / CORRECT / UNCERTAIN`으로
판단한다. SOL도 해결하지 못하면 Human Review에서 중단한다.

초기 S1–S10에서는 의미가 다른 2/3을 자동 채택하지 않는다. Human audit에서 충분한
정확도가 확인된 뒤에만 별도 version의 soft 2/3 정책을 검토한다.

## 단순한 Rule Validation

Validation gate는 Gold의 의미를 판단하지 않고, 코드로 확실히 확인 가능한 오류만
차단한다.

1. **Evidence Integrity**: evidence quote와 message ID가 current-turn user 원문과
   정확히 일치하며 금지된 source를 참조하지 않는가?
2. **Combined Integrity**:
   - temporal patch의 target·atomic apply·결과 hash를 확인
   - 30-add compaction rewrite는 schema, 길이와 hash만 확인

정보의 중요도, UPDATE/NO_OP 선택, value·condition·attribution·temporal 의미는 rule로
판정하지 않는다. 일반 turn의 의미 불일치는 Terra → SOL, compaction 불일치는
SOL로 바로 escalation한다. Rule gate는 `PASS / FAIL`만 반환하며 FAIL에는
재현 가능한 error code를 저장한다.

Summary는 별도 Gold trajectory를 만들지 않는다. 확정된 동일 `M(t)`를 UPDATE의 complete
`new_memory` target으로 사용하고, NO_OP이면 tool call 없이 이전 memory를 유지한다.

## Scenario별 순차 확정

`M(t)`가 통과하기 전에는 다음 turn으로 넘어가지 않는다. 문제가 생기면 해당
scenario만 중단하고 다른 scenario는 계속 처리한다. Human Review로 `M(t)`가
수정되면 그 확정 상태에서 이후 turn을 시작한다.

### Human Review와 자동 재개

`PAUSED_REVIEW`가 되면 중앙 SQLite queue에 current turn, 확정 `M(t-1)`, Luna 후보별
operation·gate 결과·적용 후 memory, Terra/SOL 결과를 저장한다. Reviewer는 로컬 웹에서
유효 후보 선택, 명시적 NO_OP, 직접 수정안 중 하나와 한 문장 근거를 제출한다.
30-add compaction review는 rewrite 후보 선택 또는 corrected compacted memory를 제출한다.

제출안은 request hash를 확인하고 동일 rule gate를 다시 통과해야만 turn에 atomic
commit된다. Controller는 제출을 감지해 해당 scenario의 tmux runner만 재시작하며,
commit 후 `t+1`부터 자동으로 계속한다. 실패한 제출은 `REJECTED`로 남기고 새 review
revision을 열어 audit 기록을 보존한다.

## 완료 후 QA Coverage Validation

scenario가 `COMPLETED`가 되면 SOL이 최종 `M(final)`에 공식 퀴즈 10개의 정답 정보가
포함되어 있는지 한 번 검증한다. 질문은 정답의 맥락 설명에만 사용하고,
`SUPPORTED` 근거는 반드시 최종 memory의 exact quote여야 한다.

- `SUPPORTED`: 정답에 필요한 정보가 모두 memory에 있음
- `PARTIAL`: 관련 정보는 있으나 값·사용자·조건·시간 범위 등이 부족함
- `MISSING`: 필요한 정보가 없음 또는 충돌함
- usable coverage는 `SUPPORTED + PARTIAL`로 집계
- `MISSING > 3`: `REGENERATION_RECOMMENDED`

결과는 `qa_coverage_validation.json`에 별도로 저장한다. QA는 완료된 Gold의 외부 품질
검사일 뿐이며 turn 생성·consensus·개별 label 수정에는 입력하지 않는다. 재생성 권고가
나와도 기존 artifact를 자동 삭제하거나 자동 재실행하지 않는다.

## Turn label

각 turn에는 최소한 다음 정보를 저장한다.

```json
{
  "scenario": 1,
  "turn_index": 17,
  "method": "canonical_combined",
  "decision": "UPDATE",
  "gold_memory": "...",
  "update_record": {
    "mode": "temporal_patch",
    "operations": [
      {
        "op": "replace",
        "target": "- instrument_panel_color: blue",
        "content": "- instrument_panel_color: green"
      }
    ],
    "compaction_triggered": false
  },
  "operation_reason": "Gary explicitly changes the instrument-panel color from blue to green.",
  "evidence": {
    "message_id": 18,
    "quote": "I want the instrument panel to be green now."
  },
  "validation_path": ["luna_consensus", "rule_validator"],
  "human_verified": false
}
```

`operation_reason`은 학습에 사용할 수 있는 짧은 한 문장으로 작성한다. 내부
추론 과정은 저장하지 않는다. `NO_OP`에도 새로운 durable vehicle memory가 없는
이유를 한 문장으로 기록한다.

## 완료 조건

- 모든 turn이 확정된 이전 Gold Memory에서 생성됨
- 모든 UPDATE가 원문 evidence와 연결됨
- 미래 turn과 평가 query가 생성 입력에 노출되지 않음
- correction, condition, 사용자와 temporal 상태가 보존됨
- unresolved turn이 없고 전체 trajectory를 결정론적으로 replay할 수 있음

V2의 핵심은 강한 모델 하나의 판단이 아니라, **V1과 동일한 입력에서 생성한
세 후보를 검증하고 turn별로 확정해 오류 전파를 막는 것**이다.
