from __future__ import annotations

import hashlib
import json

import pytest
from test_vehicle_v2_update_adjudicator import (
    audit_case,
    audit_decision,
    disagreed_consensus,
)

from palmclaw_ubuntu.vehicle_bench.v2_update_adjudicator import (
    UPDATE_ADJUDICATOR_PROMPT_VERSION,
    UPDATE_ADJUDICATOR_SCHEMA_VERSION,
)
from palmclaw_ubuntu.vehicle_bench.v2_update_review import (
    UpdateAuditHumanSubmission,
    UpdateAuditReviewQueue,
)


def _sol_defer() -> dict[str, object]:
    case = audit_case()
    consensus = disagreed_consensus()
    encoded = json.dumps(
        consensus,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "schema_version": UPDATE_ADJUDICATOR_SCHEMA_VERSION,
        "prompt_version": UPDATE_ADJUDICATOR_PROMPT_VERSION,
        "model_id": "test-sol",
        "response_id": "response-1",
        "case_id": case.case_id,
        "case_input_sha256": case.input_sha256,
        "consensus_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        "resolution": "DEFER_HUMAN",
        "decision": None,
        "reason": "Human review is needed.",
        "usage": {},
    }


def test_deferred_audit_can_be_submitted_immutably(tmp_path) -> None:
    queue = UpdateAuditReviewQueue(tmp_path / "audit.sqlite")
    record = queue.register_deferred(
        case=audit_case(),
        consensus=disagreed_consensus(),
        sol_report=_sol_defer(),
        checkpoint_path=tmp_path / "consensus.json",
    )

    submitted = queue.submit(
        record.review_id,
        UpdateAuditHumanSubmission.model_validate(
            {
                "reviewer": "reviewer",
                "reason": "The event directly supports the update.",
                "decision": audit_decision(),
            }
        ),
        expected_request_sha256=record.request_sha256,
    )

    assert submitted.status == "SUBMITTED"
    assert submitted.submission is not None
    assert submitted.submission.decision.verdict == "PASS"
    with pytest.raises(ValueError, match="already SUBMITTED"):
        queue.submit(
            record.review_id,
            submitted.submission,
            expected_request_sha256=record.request_sha256,
        )


def test_review_rejects_stale_hash_and_ungrounded_evidence(tmp_path) -> None:
    queue = UpdateAuditReviewQueue(tmp_path / "audit.sqlite")
    record = queue.register_deferred(
        case=audit_case(),
        consensus=disagreed_consensus(),
        sol_report=_sol_defer(),
        checkpoint_path=tmp_path / "consensus.json",
    )
    submission = UpdateAuditHumanSubmission.model_validate(
        {
            "reviewer": "reviewer",
            "reason": "The event directly supports the update.",
            "decision": audit_decision(),
        }
    )
    with pytest.raises(ValueError, match="hash has changed"):
        queue.submit(
            record.review_id,
            submission,
            expected_request_sha256="0" * 64,
        )

    invalid = audit_decision()
    invalid["candidate_updates"][0]["evidence"][0]["quote"] = "not present"
    with pytest.raises(ValueError, match="Evidence quote absent"):
        queue.submit(
            record.review_id,
            UpdateAuditHumanSubmission.model_validate(
                {
                    "reviewer": "reviewer",
                    "reason": "This evidence is invalid.",
                    "decision": invalid,
                }
            ),
            expected_request_sha256=record.request_sha256,
        )
