#!/usr/bin/env python3
"""Run/resume Luna+Terra UPDATE/NO_OP event audits and semantic consensus."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from palmclaw_ubuntu.vehicle_bench.v2_update_adjudicator import (
    UPDATE_ADJUDICATOR_PROMPT_VERSION,
    UPDATE_ADJUDICATOR_SCHEMA_VERSION,
    OpenAIV2UpdateAuditAdjudicator,
)
from palmclaw_ubuntu.vehicle_bench.v2_update_audit import (
    DEFAULT_NO_OP_EVENT_SAMPLE_RATE,
    DEFAULT_NO_OP_SAMPLE_SEED,
    EventAuditCase,
    build_update_audit_bundle,
    default_update_audit_artifact_paths,
    select_event_audit_cases,
)
from palmclaw_ubuntu.vehicle_bench.v2_update_judge import (
    UPDATE_CONSENSUS_SCHEMA_VERSION,
    UPDATE_JUDGE_PROMPT_VERSION,
    UPDATE_JUDGE_SCHEMA_VERSION,
    OpenAIV2UpdateEventJudge,
    build_dual_judge_consensus,
)
from palmclaw_ubuntu.vehicle_bench.v2_update_review import (
    UpdateAuditReviewQueue,
)

DEFAULT_EVALUATION_ROOT = Path("/mnt/data/hj153lee/PalmClaw/evaluation")
DEFAULT_OUTPUT_ROOT = (
    DEFAULT_EVALUATION_ROOT / "vehiclemembench-v2-three-way-evaluation" / "update-audit"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=DEFAULT_EVALUATION_ROOT,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--luna-model", default="gpt-5.6-luna")
    parser.add_argument("--terra-model", default="gpt-5.6-terra")
    parser.add_argument("--sol-model", default="gpt-5.6-sol")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument(
        "--no-op-event-sample-rate",
        type=float,
        default=DEFAULT_NO_OP_EVENT_SAMPLE_RATE,
    )
    parser.add_argument(
        "--no-op-sample-seed",
        default=DEFAULT_NO_OP_SAMPLE_SEED,
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("post_hoc", "hybrid", "native_turnwise"),
    )
    parser.add_argument("--scenarios", nargs="+", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.workers < 1 or args.max_attempts < 1:
        raise ValueError("workers and max attempts must be positive")
    output_root = args.output_root.expanduser().resolve()
    cases = _ready_cases(args)
    if not cases:
        raise ValueError("No UPDATE/NO_OP audit cases are ready")
    dry_run = _dry_run_summary(cases, args)
    if args.dry_run:
        return dry_run

    judges = {
        "luna": OpenAIV2UpdateEventJudge(
            args.luna_model,
            judge_role="luna",
            timeout_seconds=args.timeout_seconds,
            reasoning_effort="medium",
        ),
        "terra": OpenAIV2UpdateEventJudge(
            args.terra_model,
            judge_role="terra",
            timeout_seconds=args.timeout_seconds,
            reasoning_effort="high",
        ),
    }
    models = {
        "luna": args.luna_model,
        "terra": args.terra_model,
        "sol": args.sol_model,
    }
    reports: dict[tuple[str, str], dict[str, Any]] = {}
    work: list[tuple[EventAuditCase, str, Path, str]] = []
    for case in cases:
        directory = _case_directory(output_root, case)
        for role in ("luna", "terra"):
            checkpoint = directory / f"{role}.json"
            signature = _judge_signature(case, role=role, model=models[role])
            if checkpoint.is_file():
                stored = json.loads(checkpoint.read_text(encoding="utf-8"))
                if stored.get("evaluation_signature") != signature:
                    raise ValueError(f"Stale UPDATE Judge checkpoint: {checkpoint}")
                reports[(case.case_id, role)] = stored
            else:
                work.append((case, role, checkpoint, signature))

    def evaluate(item: tuple[EventAuditCase, str, Path, str]):
        case, role, checkpoint, signature = item
        last_error = None
        for attempt in range(1, args.max_attempts + 1):
            try:
                report = judges[role].audit(case)
                report.update(
                    {
                        "evaluation_signature": signature,
                        "attempt": attempt,
                        "method": case.method,
                        "scenario_index": case.scenario_index,
                    }
                )
                _write_json(checkpoint, report)
                return report
            except Exception as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    failures = []
    with ThreadPoolExecutor(max_workers=min(args.workers, len(work) or 1)) as pool:
        futures = {pool.submit(evaluate, item): item for item in work}
        for future in as_completed(futures):
            case, role, _, _ = futures[future]
            try:
                reports[(case.case_id, role)] = future.result()
            except Exception as exc:
                failures.append(f"{case.case_id}/{role}: {type(exc).__name__}: {exc}")

    consensuses = []
    for case in cases:
        luna = reports.get((case.case_id, "luna"))
        terra = reports.get((case.case_id, "terra"))
        if luna is None or terra is None:
            continue
        consensus = build_dual_judge_consensus(luna, terra)
        consensus.update(
            {
                "method": case.method,
                "scenario_index": case.scenario_index,
            }
        )
        _write_json(_case_directory(output_root, case) / "consensus.json", consensus)
        consensuses.append(consensus)

    case_by_id = {case.case_id: case for case in cases}
    adjudicator = OpenAIV2UpdateAuditAdjudicator(
        args.sol_model,
        timeout_seconds=args.timeout_seconds,
    )
    sol_reports: dict[str, dict[str, Any]] = {}
    sol_work = []
    for consensus in consensuses:
        if consensus["status"] != "DISAGREED":
            continue
        case = case_by_id[consensus["case_id"]]
        checkpoint = _case_directory(output_root, case) / "sol.json"
        signature = _sol_signature(case, consensus, model=args.sol_model)
        if checkpoint.is_file():
            stored = json.loads(checkpoint.read_text(encoding="utf-8"))
            if stored.get("evaluation_signature") != signature:
                raise ValueError(f"Stale UPDATE SOL checkpoint: {checkpoint}")
            sol_reports[case.case_id] = stored
        else:
            sol_work.append((case, consensus, checkpoint, signature))

    def adjudicate(item):
        case, consensus, checkpoint, signature = item
        last_error = None
        for attempt in range(1, args.max_attempts + 1):
            try:
                report = adjudicator.adjudicate(case, consensus)
                report.update(
                    {
                        "evaluation_signature": signature,
                        "attempt": attempt,
                        "method": case.method,
                        "scenario_index": case.scenario_index,
                    }
                )
                _write_json(checkpoint, report)
                return report
            except Exception as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    with ThreadPoolExecutor(max_workers=min(args.workers, len(sol_work) or 1)) as pool:
        futures = {pool.submit(adjudicate, item): item for item in sol_work}
        for future in as_completed(futures):
            case, consensus, checkpoint, signature = futures[future]
            try:
                sol_reports[case.case_id] = future.result()
            except Exception as exc:
                report = _deferred_sol_failure_report(
                    case,
                    consensus,
                    model=args.sol_model,
                    evaluation_signature=signature,
                    attempts=args.max_attempts,
                    error=exc,
                )
                _write_json(checkpoint, report)
                sol_reports[case.case_id] = report

    review_queue = UpdateAuditReviewQueue(
        output_root / "update-audit-review-queue.sqlite"
    )
    for consensus in consensuses:
        report = sol_reports.get(consensus["case_id"])
        if report is None or report["resolution"] != "DEFER_HUMAN":
            continue
        case = case_by_id[consensus["case_id"]]
        review_queue.register_deferred(
            case=case,
            consensus=consensus,
            sol_report=report,
            checkpoint_path=_case_directory(output_root, case) / "consensus.json",
        )

    new_expansions = _record_no_op_expansions(
        output_root,
        cases=cases,
        consensuses=consensuses,
        sol_reports=sol_reports,
    )
    if new_expansions:
        return run(args)

    summary = _aggregate(
        cases,
        reports,
        consensuses,
        sol_reports,
        human_review_count=sum(
            record.request["case"]["case_id"] in case_by_id
            for record in review_queue.list()
        ),
        failures=failures,
        models=models,
        dry_run=dry_run,
        no_op_expansions=_load_no_op_expansions(output_root),
    )
    _write_json(output_root / "dual-judge-summary.json", summary)
    if failures:
        raise RuntimeError(f"UPDATE Judge failures: {'; '.join(failures)}")
    return summary


def _ready_cases(args: argparse.Namespace) -> tuple[EventAuditCase, ...]:
    cases: dict[str, EventAuditCase] = {}
    output_root = args.output_root.expanduser().resolve()
    expansions = _load_no_op_expansions(output_root)
    scenario_indices = tuple(args.scenarios) if args.scenarios else None
    for paths in default_update_audit_artifact_paths(
        args.evaluation_root,
        scenario_indices=scenario_indices,
    ):
        if args.methods and paths.method not in args.methods:
            continue
        if args.scenarios and paths.scenario_index not in args.scenarios:
            continue
        if paths.stage2_artifact is None or not paths.stage2_artifact.is_file():
            continue
        if not paths.memory_artifact.is_file():
            continue
        if paths.method != "native_turnwise" and (
            paths.dialogue_root is None or not paths.dialogue_root.is_dir()
        ):
            continue
        bundle = build_update_audit_bundle(paths)
        if bundle.artifact_errors or any(
            audit.status == "FAIL" for audit in bundle.audits
        ):
            raise ValueError(
                f"Deterministic UPDATE audit failed before Judge: {paths.root}"
            )
        selected = select_event_audit_cases(
            bundle,
            no_op_event_sample_rate=args.no_op_event_sample_rate,
            no_op_sample_seed=args.no_op_sample_seed,
        )
        for case in selected:
            cases[case.case_id] = case
        matching_strata = {
            item["stratum"]
            for item in expansions
            if item["method"] == paths.method
            and item["scenario_index"] == paths.scenario_index
            and item["source_stage2_sha256"] == bundle.records[0].source_stage2_sha256
            and item["source_memory_artifact_sha256"]
            == bundle.records[0].source_memory_artifact_sha256
        }
        if matching_strata:
            for case in select_event_audit_cases(
                bundle,
                no_op_event_sample_rate=1.0,
                no_op_sample_seed=args.no_op_sample_seed,
            ):
                if (
                    case.selection_reason == "SAMPLED_NO_OP"
                    and _case_no_op_stratum(case) in matching_strata
                ):
                    cases[case.case_id] = case
    return tuple(cases[key] for key in sorted(cases))


def _dry_run_summary(
    cases: tuple[EventAuditCase, ...],
    args: argparse.Namespace,
) -> dict[str, Any]:
    counts = Counter((case.method, case.selection_reason) for case in cases)
    return {
        "schema_version": UPDATE_CONSENSUS_SCHEMA_VERSION,
        "status": "DRY_RUN",
        "case_count": len(cases),
        "initial_judge_call_count": len(cases) * 2,
        "models": {
            "luna": args.luna_model,
            "terra": args.terra_model,
            "sol": args.sol_model,
        },
        "case_counts": {
            f"{method}/{reason}": count
            for (method, reason), count in sorted(counts.items())
        },
        "case_ids": [case.case_id for case in cases],
    }


def _aggregate(
    cases: tuple[EventAuditCase, ...],
    reports: dict[tuple[str, str], dict[str, Any]],
    consensuses: list[dict[str, Any]],
    sol_reports: dict[str, dict[str, Any]],
    *,
    human_review_count: int,
    failures: list[str],
    models: dict[str, str],
    dry_run: dict[str, Any],
    no_op_expansions: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    case_by_id = {case.case_id: case for case in cases}
    for consensus in consensuses:
        case = case_by_id[consensus["case_id"]]
        groups[case.method].append(consensus)
    metrics = {}
    for method, selected in sorted(groups.items()):
        statuses = Counter(item["status"] for item in selected)
        verdicts = Counter(
            item["adopted_decision"]["verdict"]
            for item in selected
            if item["status"] == "AGREED"
        )
        final_decisions = []
        for item in selected:
            if item["status"] == "AGREED":
                final_decisions.append(item["adopted_decision"])
                continue
            sol = sol_reports.get(item["case_id"])
            if sol is not None and sol["resolution"] == "RESOLVE":
                final_decisions.append(sol["decision"])
        metrics[method] = {
            "case_count": len(selected),
            "consensus_statuses": dict(sorted(statuses.items())),
            "agreement_rate": statuses["AGREED"] / len(selected),
            "agreed_verdicts": dict(sorted(verdicts.items())),
            "sol_escalated_count": statuses["DISAGREED"],
            "sol_resolved_count": sum(
                sol_reports[item["case_id"]]["resolution"] == "RESOLVE"
                for item in selected
                if item["case_id"] in sol_reports
            ),
            "human_deferred_count": sum(
                sol_reports[item["case_id"]]["resolution"] == "DEFER_HUMAN"
                for item in selected
                if item["case_id"] in sol_reports
            ),
            "final_decision_count": len(final_decisions),
            "final_verdicts": dict(
                sorted(Counter(item["verdict"] for item in final_decisions).items())
            ),
            "final_memory_results": dict(
                sorted(
                    Counter(item["memory_result"] for item in final_decisions).items()
                )
            ),
        }
    usage = {}
    for role in ("luna", "terra"):
        selected = [
            report
            for (_, report_role), report in reports.items()
            if report_role == role
        ]
        usage[role] = {
            key: sum(int(item.get("usage", {}).get(key, 0)) for item in selected)
            for key in (
                "input_tokens",
                "cached_tokens",
                "output_tokens",
                "total_tokens",
            )
        }
    usage["sol"] = {
        key: sum(
            int(item.get("usage", {}).get(key, 0)) for item in sol_reports.values()
        )
        for key in (
            "input_tokens",
            "cached_tokens",
            "output_tokens",
            "total_tokens",
        )
    }
    return {
        "schema_version": UPDATE_CONSENSUS_SCHEMA_VERSION,
        "judge_schema_version": UPDATE_JUDGE_SCHEMA_VERSION,
        "prompt_version": UPDATE_JUDGE_PROMPT_VERSION,
        "status": "COMPLETED" if not failures else "PARTIAL",
        "models": models,
        "case_count": len(cases),
        "judge_report_count": len(reports),
        "consensus_count": len(consensuses),
        "sol_report_count": len(sol_reports),
        "human_review_count": human_review_count,
        "failure_count": len(failures),
        "failures": failures,
        "usage": usage,
        "metrics": metrics,
        "human_review_case_ids": [
            case_id
            for case_id, report in sol_reports.items()
            if report["resolution"] == "DEFER_HUMAN"
        ],
        "no_op_expansions": list(no_op_expansions),
        "dry_run": dry_run,
    }


def _case_directory(output_root: Path, case: EventAuditCase) -> Path:
    digest = hashlib.sha256(case.case_id.encode()).hexdigest()[:16]
    return (
        output_root
        / "judge"
        / case.method
        / f"s{case.scenario_index:02d}"
        / f"{case.event_id}-{digest}"
    )


def _judge_signature(case: EventAuditCase, *, role: str, model: str) -> str:
    body = {
        "judge_schema_version": UPDATE_JUDGE_SCHEMA_VERSION,
        "prompt_version": UPDATE_JUDGE_PROMPT_VERSION,
        "case_id": case.case_id,
        "case_input_sha256": case.input_sha256,
        "judge_role": role,
        "model": model,
    }
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sol_signature(
    case: EventAuditCase,
    consensus: dict[str, Any],
    *,
    model: str,
) -> str:
    body = {
        "schema_version": UPDATE_ADJUDICATOR_SCHEMA_VERSION,
        "prompt_version": UPDATE_ADJUDICATOR_PROMPT_VERSION,
        "model": model,
        "case_id": case.case_id,
        "case_input_sha256": case.input_sha256,
        "consensus": consensus,
    }
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _record_no_op_expansions(
    output_root: Path,
    *,
    cases: tuple[EventAuditCase, ...],
    consensuses: list[dict[str, Any]],
    sol_reports: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    consensus_by_id = {item["case_id"]: item for item in consensuses}
    requested = []
    for case in cases:
        if case.selection_reason != "SAMPLED_NO_OP":
            continue
        consensus = consensus_by_id.get(case.case_id)
        if consensus is None:
            continue
        decision = None
        if consensus["status"] == "AGREED":
            decision = consensus["adopted_decision"]
        else:
            sol = sol_reports.get(case.case_id)
            if sol is not None and sol["resolution"] == "RESOLVE":
                decision = sol["decision"]
        if decision is None or (
            decision["verdict"] == "PASS" and decision["memory_result"] == "FAITHFUL"
        ):
            continue
        requested.append(
            {
                "method": case.method,
                "scenario_index": case.scenario_index,
                "source_stage2_sha256": case.source_stage2_sha256,
                "source_memory_artifact_sha256": (case.source_memory_artifact_sha256),
                "stratum": _case_no_op_stratum(case),
                "trigger_case_id": case.case_id,
            }
        )
    if not requested:
        return ()
    path = output_root / "no-op-expansion.json"
    existing = list(_load_no_op_expansions(output_root))
    material_keys = {
        (
            item["method"],
            item["scenario_index"],
            item["source_stage2_sha256"],
            item["source_memory_artifact_sha256"],
            item["stratum"],
        )
        for item in existing
    }
    added = []
    for item in requested:
        key = (
            item["method"],
            item["scenario_index"],
            item["source_stage2_sha256"],
            item["source_memory_artifact_sha256"],
            item["stratum"],
        )
        if key in material_keys:
            continue
        material_keys.add(key)
        existing.append(item)
        added.append(item)
    if added:
        _write_json(
            path,
            {
                "schema_version": "vehiclemembench-v2-no-op-expansion-v1",
                "expansions": sorted(
                    existing,
                    key=lambda item: (
                        item["method"],
                        item["scenario_index"],
                        item["stratum"],
                    ),
                ),
            },
        )
    return tuple(added)


def _load_no_op_expansions(output_root: Path) -> tuple[dict[str, Any], ...]:
    path = output_root / "no-op-expansion.json"
    if not path.is_file():
        return ()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "vehiclemembench-v2-no-op-expansion-v1":
        raise ValueError(f"Invalid NO_OP expansion manifest: {path}")
    return tuple(payload.get("expansions", ()))


def _case_no_op_stratum(case: EventAuditCase) -> str:
    return "+".join(sorted(case.no_op_reason_codes))


def _deferred_sol_failure_report(
    case: EventAuditCase,
    consensus: dict[str, Any],
    *,
    model: str,
    evaluation_signature: str,
    attempts: int,
    error: Exception,
) -> dict[str, Any]:
    consensus_body = json.dumps(
        consensus,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "schema_version": UPDATE_ADJUDICATOR_SCHEMA_VERSION,
        "prompt_version": UPDATE_ADJUDICATOR_PROMPT_VERSION,
        "model_id": model,
        "response_id": None,
        "case_id": case.case_id,
        "case_input_sha256": case.input_sha256,
        "consensus_sha256": hashlib.sha256(consensus_body.encode("utf-8")).hexdigest(),
        "resolution": "DEFER_HUMAN",
        "decision": None,
        "reason": "SOL response failed schema or evidence grounding after retries.",
        "usage": {
            "input_tokens": 0,
            "cached_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        },
        "evaluation_signature": evaluation_signature,
        "attempt": attempts,
        "method": case.method,
        "scenario_index": case.scenario_index,
        "provider_failure": f"{type(error).__name__}: {error}",
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
