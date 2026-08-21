from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

from test_vehicle_v2_update_adjudicator import audit_case, audit_decision


def _runner_module():
    path = (
        Path(__file__).parents[1]
        / "evaluation/experiment-scripts/"
        / "run_vehiclemembench_v2_update_audit_judges.py"
    )
    spec = importlib.util.spec_from_file_location("update_audit_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sampled_no_op_error_records_stratum_expansion(tmp_path) -> None:
    runner = _runner_module()
    source = audit_case()
    case = replace(
        source,
        case_id="hybrid:s01:no-op-event",
        selection_reason="SAMPLED_NO_OP",
        expected_updates=(),
        candidate_updates=(),
        no_op_reason_codes=("NO_DURABLE_MEMORY",),
    )
    decision = {
        **audit_decision(),
        "case_id": case.case_id,
        "verdict": "FAIL",
        "expected_updates": [],
        "candidate_updates": [],
        "memory_result": "INFORMATION_LOSS",
    }
    consensus = {
        "case_id": case.case_id,
        "status": "AGREED",
        "adopted_decision": decision,
    }

    added = runner._record_no_op_expansions(
        tmp_path,
        cases=(case,),
        consensuses=[consensus],
        sol_reports={},
    )

    assert len(added) == 1
    assert added[0]["stratum"] == "NO_DURABLE_MEMORY"
    assert runner._load_no_op_expansions(tmp_path) == added
    assert (
        runner._record_no_op_expansions(
            tmp_path,
            cases=(case,),
            consensuses=[consensus],
            sol_reports={},
        )
        == ()
    )
