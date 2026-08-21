#!/usr/bin/env python3
"""Serve a local Human Review UI for VehicleMemBench V2."""

from __future__ import annotations

import argparse
import html
import json
import re
import sqlite3
import subprocess
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

from pydantic import ValidationError

from palmclaw_ubuntu.vehicle_bench.human_review import (
    HumanReviewQueue,
    HumanReviewRecord,
    HumanReviewSubmission,
)
from palmclaw_ubuntu.vehicle_bench.v2_update_review import (
    UpdateAuditHumanSubmission,
    UpdateAuditReviewQueue,
    UpdateAuditReviewRecord,
)

DEFAULT_QUEUE = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2/"
    "consensus-gold-v2/review_queue.sqlite"
)
DEFAULT_UPDATE_AUDIT_QUEUE = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/"
    "vehiclemembench-v2-three-way-evaluation/update-audit/"
    "update-audit-review-queue.sqlite"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument(
        "--update-audit-queue",
        type=Path,
        default=DEFAULT_UPDATE_AUDIT_QUEUE,
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    queue_store = HumanReviewQueue(args.queue)
    update_audit_store = UpdateAuditReviewQueue(args.update_audit_queue)

    class Handler(ReviewHandler):
        queue = queue_store
        update_audit_queue = update_audit_store

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Human Review UI: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


class ReviewHandler(BaseHTTPRequestHandler):
    queue: HumanReviewQueue
    update_audit_queue: UpdateAuditReviewQueue

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            review_status = (
                parse_qs(parsed.query).get("review_status", ["PENDING"])[0].upper()
            )
            if review_status not in {
                "PENDING",
                "SUBMITTED",
                "APPLIED",
                "REJECTED",
            }:
                self.send_error(HTTPStatus.BAD_REQUEST, "Invalid review status")
                return
            self._send_html(
                _render_index(
                    self.queue,
                    queues=_related_queues(self.queue),
                    update_audit_queue=self.update_audit_queue,
                    review_status=review_status,
                )
            )
            return
        if path.startswith("/review/"):
            review_id = unquote(path.removeprefix("/review/"))
            try:
                _, record = _find_review(
                    self.queue,
                    review_id,
                    update_audit_queue=self.update_audit_queue,
                )
            except KeyError as error:
                self.send_error(HTTPStatus.NOT_FOUND, str(error))
                return
            self._send_html(_render_review(record))
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if not path.startswith("/review/") or not path.endswith("/submit"):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        review_id = unquote(path.removeprefix("/review/").removesuffix("/submit"))
        length = int(self.headers.get("Content-Length", "0"))
        form = parse_qs(self.rfile.read(length).decode("utf-8"), keep_blank_values=True)
        try:
            review_queue, record = _find_review(
                self.queue,
                review_id,
                update_audit_queue=self.update_audit_queue,
            )
            review_kind = _review_kind(record.request)
            if review_kind == "UPDATE_AUDIT":
                submission = UpdateAuditHumanSubmission.model_validate(
                    {
                        "reviewer": _one(form, "reviewer"),
                        "reason": _one(form, "reason"),
                        "decision": json.loads(_one(form, "correction")),
                    }
                )
                review_queue.submit(
                    review_id,
                    submission,
                    expected_request_sha256=_one(form, "request_sha256"),
                )
                self.send_response(HTTPStatus.SEE_OTHER)
                self.send_header("Location", f"/review/{quote(review_id)}")
                self.end_headers()
                return
            action_value = _one(form, "action")
            selected_sample_id = None
            corrected_candidate = None
            corrected_memory = None
            if action_value.startswith("SELECT:"):
                action = "SELECT"
                selected_sample_id = action_value.split(":", 1)[1]
            elif action_value == "CORRECT":
                action = "CORRECT"
                correction = _one(form, "correction")
                if review_kind == "COMPACTION":
                    corrected_memory = correction
                else:
                    corrected_candidate = json.loads(correction)
            elif action_value == "NO_OP":
                action = "NO_OP"
            else:
                raise ValueError(f"Unsupported review action: {action_value}")
            submission = HumanReviewSubmission.model_validate(
                {
                    "reviewer": _one(form, "reviewer"),
                    "action": action,
                    "reason": _one(form, "reason"),
                    "selected_sample_id": selected_sample_id,
                    "corrected_candidate": corrected_candidate,
                    "corrected_memory": corrected_memory,
                }
            )
            review_queue.submit(
                review_id,
                submission,
                expected_request_sha256=_one(form, "request_sha256"),
            )
        except (KeyError, ValueError, ValidationError, json.JSONDecodeError) as error:
            self._send_html(
                _page("Review submission failed", f"<pre>{_escape(error)}</pre>"),
                status=HTTPStatus.BAD_REQUEST,
            )
            return
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", f"/review/{quote(review_id)}")
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        print(f"review-web: {format % args}", flush=True)

    def _send_html(
        self,
        content: str,
        *,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        encoded = content.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def _render_index(
    queue: HumanReviewQueue,
    *,
    queues: tuple[HumanReviewQueue, ...] | None = None,
    update_audit_queue: UpdateAuditReviewQueue | None = None,
    review_status: str = "PENDING",
) -> str:
    active_queues = queues or (queue,)
    update_records = update_audit_queue.list() if update_audit_queue is not None else ()
    gold_records = tuple(
        record for active_queue in active_queues for record in active_queue.list()
    )
    all_records = tuple(
        sorted(
            (*gold_records, *update_records),
            key=lambda record: (record.created_at, record.review_id),
        )
    )
    records = tuple(record for record in all_records if record.status == review_status)
    scenario_statuses = _pipeline_statuses(queue.path.parent)
    counts = {
        status: sum(record.status == status for record in all_records)
        for status in ("PENDING", "SUBMITTED", "APPLIED", "REJECTED")
    }
    cards = []
    for record in records:
        is_update_audit = _review_kind(record.request) == "UPDATE_AUDIT"
        turn_count, terra_calls, sol_calls = (
            (None, 1, 1) if is_update_audit else _checkpoint_stats(record)
        )
        review_turn = record.turn_index + 1
        progress = min(100.0, review_turn * 100 / turn_count) if turn_count else 0.0
        turn_label = (
            f"Event {record.request['case']['event_id']}"
            if is_update_audit
            else (
                f"{review_turn:,} / {turn_count:,}"
                if turn_count is not None
                else f"{review_turn:,} / ?"
            )
        )
        error = (
            f"<p class='error-note'>{_escape(record.apply_error)}</p>"
            if record.apply_error
            else ""
        )
        current_turn = record.request.get("current_turn", {}).get("raw", "")
        card_kind = "UPDATE Audit" if is_update_audit else "Human Review Turn"
        cards.append(
            f"""
            <a class="review-card" href="/review/{quote(record.review_id)}">
              <div class="card-top">
                <span class="scenario-pill">S{record.scenario_index}</span>
                {_status_badge(record.status)}
              </div>
              <div class="turn-row">
                <div><span class="label">{card_kind}</span>
                <strong>{turn_label}</strong></div>
                <span class="percent">{progress:.1f}%</span>
              </div>
              <div class="progress-track" aria-label="Scenario progress">
                <span style="width:{progress:.2f}%"></span>
              </div>
              <div class="call-grid">
                <div><span>Terra calls</span><strong>{terra_calls}</strong></div>
                <div><span>SOL calls</span><strong>{sol_calls}</strong></div>
              </div>
              <p class="turn-preview">{_escape(current_turn)}</p>
              {error}
              <div class="card-link">Open review <span>→</span></div>
            </a>
            """
        )
    empty = (
        f'<div class="empty-state">No {review_status.title()} reviews.</div>'
        if not cards
        else ""
    )
    status_rows = []
    for scenario in scenario_statuses:
        progress = scenario["progress"]
        qa_coverage = scenario["qa_coverage"] or "—"
        quiz_score = scenario["quiz_score"] or "Not evaluated for this run"
        status_rows.append(
            f"""
            <tr>
              <td><strong>S{scenario["scenario_index"]}</strong></td>
              <td>{_status_badge(str(scenario["status"]))}</td>
              <td>
                <div class="table-progress-label">
                  <span>{scenario["done"]:,} / {scenario["total"]:,}</span>
                  <strong>{progress:.1f}%</strong>
                </div>
                <div class="progress-track table-progress">
                  <span style="width:{progress:.2f}%"></span>
                </div>
              </td>
              <td>Combined · Temporal Patch + 30-add Compaction</td>
              <td>{scenario["updates"]:,} / {scenario["compactions"]:,}</td>
              <td>{_escape(qa_coverage)}</td>
              <td>{_escape(quiz_score)}</td>
              <td><span class="source-run">{_escape(scenario["source_run"])}</span></td>
            </tr>
            """
        )
    body = f"""
    <header class="hero">
      <p class="eyebrow">PALMCLAW · CONSENSUS GOLD V2</p>
      <div class="hero-title-row"><h1>Pipeline Status</h1>
      <a class="refresh-button" href="/?review_status={review_status}">Refresh</a></div>
      <p class="hero-copy">Inspect only the paused turn, make one decision,
      and let the checkpointed scenario resume automatically.</p>
      <div class="summary-grid">
        {_review_count_link("PENDING", counts["PENDING"], review_status)}
        {_review_count_link("SUBMITTED", counts["SUBMITTED"], review_status)}
        {_review_count_link("APPLIED", counts["APPLIED"], review_status)}
        {_review_count_link("REJECTED", counts["REJECTED"], review_status)}
      </div>
    </header>
    <section class="pipeline-panel">
      <div class="section-heading"><div><p class="eyebrow">LIVE CHECKPOINTS</p>
      <h2>Scenario progress</h2></div>
      <span class="muted">Updated only when Refresh is pressed</span></div>
      <div class="table-wrap"><table class="status-table">
        <thead><tr><th>Scenario</th><th>Status</th><th>Progress</th>
        <th>Method</th><th>Updates / Compactions</th><th>Final QA</th>
        <th>Quiz ESM / Arg</th><th>Latest run</th></tr></thead>
        <tbody>{"".join(status_rows)}</tbody>
      </table></div>
    </section>
    <section class="queue-heading"><p class="eyebrow">REVIEW HISTORY</p>
    <h2>{review_status.title()} reviews</h2></section>
    <main class="review-grid">{"".join(cards)}{empty}</main>
    """
    return _page("VehicleMemBench V2 Review", body)


def _render_review(record) -> str:
    request = record.request
    if _review_kind(request) == "UPDATE_AUDIT":
        return _render_update_audit_review(record)
    pending = request["pending_review"]
    gate_results = pending.get("payload", {}).get("gate_results", [])
    escalation_attempts = request.get("escalation_attempts", [])
    review_kind = _review_kind(request)
    candidate_sections = []
    available_ids = []
    for result in gate_results:
        sample_id = str(result.get("sample_id", "?"))
        if result.get("status") == "PASS":
            available_ids.append(sample_id)
        candidate = result.get("candidate", {}).get("payload", {})
        gate_status = str(result.get("status", "UNKNOWN"))
        candidate_sections.append(
            f"""
            <article class="candidate-card">
              <div class="candidate-head">
                <h3>Candidate {html.escape(sample_id)}</h3>
                <span class="gate-{gate_status.lower()}">{_escape(gate_status)}</span>
              </div>
              <p class="muted">Gate errors:
              {_escape(result.get("error_codes", []))}</p>
              <pre>{_escape(json.dumps(candidate, ensure_ascii=False, indent=2))}</pre>
              <details><summary>View resulting memory</summary>
              <pre>{_escape(result.get("after_memory") or "(unavailable)")}</pre>
              </details>
            </article>
            """
        )
    controls = ""
    if record.status == "PENDING":
        options = "".join(
            f"<option value='SELECT:{html.escape(sample_id)}'>"
            f"Select candidate {html.escape(sample_id)}</option>"
            for sample_id in available_ids
        )
        if review_kind == "COMPACTION":
            correction_template = pending["payload"]["post_patch_memory"]
            correction_label = "Corrected compacted memory"
            correction_option = "Submit corrected compacted memory"
            no_op_option = ""
        else:
            correction_template = json.dumps(
                {
                    "decision": "UPDATE",
                    "operations": [],
                    "reason": "One-line reason grounded in the current turn.",
                    "evidence": [{"quote": request["current_turn"]["content"]}],
                },
                ensure_ascii=False,
                indent=2,
            )
            correction_label = "Corrected Combined candidate JSON"
            correction_option = "Submit corrected candidate JSON"
            no_op_option = '<option value="NO_OP">Choose NO_OP</option>'
        controls = f"""
        <section class="panel decision-panel">
        <h2>Submit decision</h2>
        <form method="post" action="/review/{quote(record.review_id)}/submit">
          <input type="hidden" name="request_sha256"
                 value="{html.escape(record.request_sha256)}">
          <div class="form-grid">
          <label class="field">Reviewer<input required name="reviewer"
                 placeholder="Your name"></label>
          <label class="field">Action<select id="review-action" name="action">{options}
            {no_op_option}
            <option value="CORRECT">{correction_option}</option>
          </select></label>
          </div>
          <label class="field">One-line reason
          <input required name="reason"
                 placeholder="Why is this the correct decision?"></label>
          <div id="correction-fields" class="correction-fields">
            <label class="field">{correction_label}
            <textarea name="correction" rows="18">
            {_escape(correction_template)}</textarea></label>
          </div>
          <button type="submit">Submit immutable review <span>→</span></button>
        </form>
        </section>
        """
    elif record.submission is not None:
        controls = (
            '<section class="panel"><h2>Submitted decision</h2><pre>'
            + _escape(
                json.dumps(
                    record.submission.model_dump(mode="json"),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            + "</pre></section>"
        )
    turn_count, terra_calls, sol_calls = _checkpoint_stats(record)
    review_turn = record.turn_index + 1
    progress = min(100.0, review_turn * 100 / turn_count) if turn_count else 0.0
    turn_label = (
        f"{review_turn:,} / {turn_count:,}"
        if turn_count is not None
        else f"{review_turn:,} / ?"
    )
    body = f"""
    <nav><a class="back-link" href="/">← Human Review Queue</a></nav>
    <header class="detail-header">
      <div><p class="eyebrow">SCENARIO {record.scenario_index}</p>
      <h1>Turn {turn_label}</h1></div>
      {_status_badge(record.status)}
      <div class="wide-progress progress-track">
        <span style="width:{progress:.2f}%"></span>
      </div>
      <div class="detail-metrics">
        <span><strong>{progress:.1f}%</strong> scenario progress</span>
        <span><strong>{terra_calls}</strong> Terra calls</span>
        <span><strong>{sol_calls}</strong> SOL calls</span>
      </div>
    </header>
    <section class="panel"><h2>Current turn</h2>
    <pre>{_escape(request["current_turn"]["raw"])}</pre></section>
    <section class="panel"><h2>Pause reason</h2>
    <div class="reason-box">{_escape(pending["reason"])}</div></section>
    <section class="panel"><h2>Approved M(t-1)</h2>
    <details><summary>View approved memory</summary>
    <pre>{_escape(request["approved_memory_before"])}</pre></details></section>
    <section><h2>Candidates</h2>
    <div class="candidate-grid">{"".join(candidate_sections)}</div></section>
    <section class="panel"><h2>Terra / SOL attempts</h2>
    <details><summary>View resolver trace</summary>
    <pre>{_escape(json.dumps(escalation_attempts, ensure_ascii=False, indent=2))}</pre>
    </details></section>
    {controls}
    """
    return _page(f"Review {record.review_id}", body)


def _render_update_audit_review(record: UpdateAuditReviewRecord) -> str:
    request = record.request
    case = request["case"]
    consensus = request["dual_judge_consensus"]
    sol_report = request["sol_report"]
    dialogue = "\n".join(
        f"[{turn['turn_id']}] {turn['speaker_name']}: {turn['text']}"
        for turn in case.get("dialogue", [])
    )
    judge_decisions = {
        role: consensus["judge_reports"][role]["decision"] for role in ("luna", "terra")
    }
    controls = ""
    if record.status == "PENDING":
        template = json.loads(json.dumps(judge_decisions["terra"]))
        for expected in template["expected_updates"]:
            expected.pop("boundary_offset", None)
        decision_template = json.dumps(
            template,
            ensure_ascii=False,
            indent=2,
        )
        escaped_decision_template = _escape(decision_template)
        controls = f"""
        <section class="panel decision-panel">
          <h2>Submit final audit decision</h2>
          <p class="muted">Use only evidence present in this event dialogue.
          The original memory artifact is not modified by this review.</p>
          <form method="post" action="/review/{quote(record.review_id)}/submit">
            <input type="hidden" name="request_sha256"
                   value="{html.escape(record.request_sha256)}">
            <label class="field">Reviewer
              <input required name="reviewer" placeholder="Your name">
            </label>
            <label class="field">One-line reason
              <input required name="reason"
                     placeholder="Why is this the grounded final decision?">
            </label>
            <label class="field">Final audit decision JSON
              <textarea required name="correction"
                        rows="24">{escaped_decision_template}</textarea>
            </label>
            <button type="submit">Submit immutable audit review <span>→</span></button>
          </form>
        </section>
        """
    elif record.submission is not None:
        controls = (
            '<section class="panel"><h2>Submitted final audit decision</h2><pre>'
            + _escape(
                json.dumps(
                    record.submission.model_dump(mode="json"),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            + "</pre></section>"
        )
    updates = {
        "expected_updates": case["expected_updates"],
        "candidate_updates": case["candidate_updates"],
    }
    material_diff = json.dumps(
        consensus["material_diff"],
        ensure_ascii=False,
        indent=2,
    )
    body = f"""
    <nav><a class="back-link" href="/">← Human Review Queue</a></nav>
    <header class="detail-header">
      <div><p class="eyebrow">UPDATE / NO_OP AUDIT ·
      SCENARIO {record.scenario_index}</p>
      <h1>{_escape(case["event_id"])}</h1></div>
      {_status_badge(record.status)}
      <div class="detail-metrics">
        <span><strong>{_escape(case["method"])}</strong> method</span>
        <span><strong>{_escape(case["selection_reason"])}</strong> selection</span>
        <span><strong>{_escape(sol_report["resolution"])}</strong> SOL outcome</span>
      </div>
    </header>
    <section class="panel"><h2>Causal event dialogue</h2>
      <pre>{_escape(dialogue or "(empty event)")}</pre></section>
    <section class="panel"><h2>Memory transition</h2>
      <h3>Before</h3><pre>{_escape(case["event_before_memory"])}</pre>
      <h3>After</h3><pre>{_escape(case["event_after_memory"])}</pre>
    </section>
    <section class="panel"><h2>Expected and candidate updates</h2>
      <pre>{_escape(json.dumps(updates, ensure_ascii=False, indent=2))}</pre>
    </section>
    <section class="panel"><h2>Luna / Terra disagreement</h2>
      <details open><summary>Material diff</summary>
        <pre>{_escape(material_diff)}</pre>
      </details>
      <details><summary>Judge decisions</summary>
        <pre>{_escape(json.dumps(judge_decisions, ensure_ascii=False, indent=2))}</pre>
      </details>
    </section>
    <section class="panel"><h2>SOL adjudication</h2>
      <div class="reason-box">{_escape(sol_report["reason"])}</div>
    </section>
    {controls}
    """
    return _page(f"UPDATE audit {record.review_id}", body)


def _page(title: str, body: str) -> str:
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>{html.escape(title)}</title><style>
    :root{{--bg:#f3f6fb;--surface:#fff;--ink:#182235;--muted:#65728a;
    --line:#dde4ef;--navy:#19355d;--blue:#3c77d8;--cyan:#42b8c8;
    --shadow:0 14px 38px rgba(24,48,85,.09)}}
    *{{box-sizing:border-box}}body{{margin:0;color:var(--ink);background:var(--bg);
    font:15px/1.55 Inter,ui-sans-serif,system-ui,-apple-system,sans-serif}}
    body:before{{content:"";display:block;height:5px;
    background:linear-gradient(90deg,var(--navy),var(--blue),var(--cyan))}}
    body>header,body>main,body>nav,body>section{{max-width:1180px;margin-left:auto;
    margin-right:auto}}.hero{{padding:64px 24px 30px}}.eyebrow{{margin:0 0 8px;
    color:var(--blue);font-size:12px;font-weight:800;letter-spacing:.14em}}
    h1{{margin:0;font-size:clamp(30px,4vw,48px);line-height:1.1;letter-spacing:-.03em}}
    h2{{margin:0 0 16px;font-size:20px}}h3{{margin:0;font-size:17px}}
    .hero-copy{{max-width:680px;margin:14px 0 28px;color:var(--muted);font-size:17px}}
    .hero-title-row,.section-heading,.table-progress-label{{display:flex;
    align-items:center;justify-content:space-between;gap:16px}}
    .refresh-button{{padding:10px 16px;color:#fff;background:var(--navy);
    border-radius:10px;text-decoration:none;font-weight:800}}
    .refresh-button:hover{{background:#254e83}}
    .summary-grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}}
    .summary-grid a{{padding:17px 20px;color:inherit;text-decoration:none;
    border:1px solid var(--line);border-radius:14px;background:rgba(255,255,255,.7)}}
    .summary-grid a:hover,.summary-grid a.active{{border-color:var(--blue);
    background:#edf4ff}}
    .summary-grid strong{{display:block;font-size:25px}}
    .summary-grid span,.label,.call-grid span{{color:var(--muted);font-size:12px}}
    .review-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;
    padding:0 24px 64px}}.review-card{{display:block;padding:24px;color:inherit;
    text-decoration:none;background:var(--surface);border:1px solid var(--line);
    border-radius:20px;box-shadow:var(--shadow);transition:.18s ease}}
    .review-card:hover{{transform:translateY(-3px);border-color:#abc4ea}}
    .card-top,.turn-row,.candidate-head{{display:flex;align-items:center;
    justify-content:space-between;gap:12px}}.scenario-pill{{font-weight:850;color:var(--navy);
    font-size:18px}}.status-badge{{padding:5px 10px;border-radius:999px;font-size:11px;
    font-weight:800;letter-spacing:.06em}}.status-pending{{color:#9a5b05;background:#fff0c7}}
    .status-submitted{{color:#2458a6;background:#e5efff}}
    .status-applied{{color:#14704b;background:#dcf7e9}}
    .status-rejected,.status-failed,.status-stopped{{color:#a33434;background:#ffe5e5}}
    .status-running{{color:#2458a6;background:#e5efff}}
    .status-completed{{color:#14704b;background:#dcf7e9}}
    .status-paused_review{{color:#9a5b05;background:#fff0c7}}
    .turn-row{{margin-top:20px}}
    .turn-row strong{{display:block;margin-top:2px;font-size:22px}}
    .percent{{font-weight:800;color:var(--blue)}}
    .progress-track{{height:8px;overflow:hidden;margin:12px 0 20px;
    border-radius:99px;background:#e8edf5}}.progress-track span{{display:block;
    height:100%;
    border-radius:inherit;background:linear-gradient(90deg,var(--blue),var(--cyan))}}
    .call-grid{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}
    .call-grid div{{padding:12px 14px;border-radius:12px;background:#f6f8fc}}
    .call-grid strong{{display:block;font-size:19px}}.turn-preview{{min-height:48px;
    margin:16px 0;color:#3c485d}}.card-link{{display:flex;justify-content:space-between;
    padding-top:14px;border-top:1px solid var(--line);color:var(--blue);
    font-weight:750}}
    .error-note{{color:#a33434}}nav{{padding:28px 24px 0}}.back-link{{color:var(--blue);
    text-decoration:none;font-weight:700}}.detail-header{{display:grid;
    grid-template-columns:1fr auto;gap:18px;padding:34px 24px 24px}}
    .wide-progress,.detail-metrics{{grid-column:1/-1;margin:0}}
    .detail-metrics{{display:flex;flex-wrap:wrap;gap:20px;color:var(--muted)}}
    .detail-metrics strong{{color:var(--ink)}}.panel,body>section{{margin-top:18px;
    padding:24px;background:var(--surface);border:1px solid var(--line);
    border-radius:18px;box-shadow:0 8px 25px rgba(24,48,85,.055)}}
    .candidate-grid{{display:grid;gap:16px}}.candidate-card{{padding:20px;
    background:var(--surface);border:1px solid var(--line);border-radius:16px}}
    .gate-pass{{color:#14704b;background:#dcf7e9}}.gate-fail{{color:#a33434;
    background:#ffe5e5}}.gate-pass,.gate-fail{{padding:4px 9px;border-radius:99px;
    font-size:11px;font-weight:800}}.muted{{color:var(--muted)}}
    pre{{max-height:560px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;
    padding:16px;background:#f5f7fb;border:1px solid #e5eaf2;border-radius:12px;
    font:13px/1.55 ui-monospace,SFMono-Regular,Consolas,monospace}}
    details summary{{cursor:pointer;color:var(--blue);font-weight:700}}
    .reason-box{{padding:13px 16px;color:#744d08;background:#fff8e6;
    border-left:4px solid #e7ad3b;border-radius:8px}}
    .decision-panel{{margin-bottom:70px}}
    .form-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
    .field{{display:block;margin:14px 0;color:#334159;font-weight:700}}
    input,select,textarea{{display:block;width:100%;margin-top:7px;padding:11px 12px;
    color:var(--ink);background:#fff;border:1px solid #cbd5e3;border-radius:10px;
    font:inherit}}textarea{{resize:vertical;font-family:ui-monospace,monospace}}
    input:focus,select:focus,textarea:focus{{outline:3px solid #dbe9ff;
    border-color:var(--blue)}}button{{margin-top:16px;padding:12px 18px;color:#fff;
    background:var(--navy);border:0;border-radius:10px;font-weight:800;cursor:pointer}}
    button:hover{{background:#254e83}}.correction-fields{{display:none}}
    .empty-state{{grid-column:1/-1;padding:50px;text-align:center;color:var(--muted)}}
    .pipeline-panel{{padding:24px;background:var(--surface);
    border:1px solid var(--line);border-radius:18px;box-shadow:var(--shadow)}}
    .queue-heading{{padding:42px 24px 16px}}
    .table-wrap{{overflow-x:auto}}.status-table{{width:100%;border-collapse:collapse}}
    .status-table th,.status-table td{{padding:14px 12px;
    border-bottom:1px solid var(--line);text-align:left;vertical-align:middle}}
    .status-table th{{color:var(--muted);
    font-size:12px;white-space:nowrap}}.status-table td:nth-child(3){{min-width:210px}}
    .table-progress{{height:6px;margin:7px 0 0}}.table-progress-label{{font-size:13px}}
    .source-run{{color:var(--muted);font-size:12px;white-space:nowrap}}
    @media(max-width:720px){{.review-grid,.summary-grid,.form-grid{{grid-template-columns:1fr}}
    .hero{{padding-top:38px}}.detail-metrics{{display:grid;gap:6px}}
    .hero-title-row{{align-items:flex-start}}}}
    </style></head><body>{body}<script>
    const action=document.getElementById('review-action');
    const correction=document.getElementById('correction-fields');
    function syncCorrection(){{if(correction&&action){{
      correction.style.display=action.value==='CORRECT'?'block':'none';
    }}}}
    if(action){{action.addEventListener('change',syncCorrection);syncCorrection();}}
    </script></body></html>"""


def _status_badge(status: str) -> str:
    return (
        f'<span class="status-badge status-{status.lower()}">{_escape(status)}</span>'
    )


def _review_count_link(status: str, count: int, selected: str) -> str:
    active = " active" if status == selected else ""
    return (
        f'<a class="summary-card{active}" href="/?review_status={status}">'
        f"<strong>{count}</strong><span>{status.title()}</span></a>"
    )


def _checkpoint_stats(
    record: HumanReviewRecord,
) -> tuple[int | None, int | str, int | str]:
    try:
        uri = f"file:{Path(record.checkpoint_path).as_posix()}?mode=ro"
        with sqlite3.connect(uri, uri=True, timeout=2) as connection:
            connection.execute("BEGIN")
            state = connection.execute(
                "SELECT turn_count FROM run_state WHERE singleton = 1"
            ).fetchone()
            calls = connection.execute(
                """
                SELECT
                  SUM(CASE WHEN stage IN ('TERRA', 'COMPACTION_TERRA')
                           THEN 1 ELSE 0 END),
                  SUM(CASE WHEN stage IN ('SOL', 'COMPACTION_SOL')
                           THEN 1 ELSE 0 END)
                FROM model_attempts WHERE turn_index <= ?
                """,
                (record.turn_index,),
            ).fetchone()
        turn_count = int(state[0]) if state else None
        return turn_count, int(calls[0] or 0), int(calls[1] or 0)
    except sqlite3.Error:
        return None, "?", "?"


def _pipeline_statuses(output_root: Path) -> list[dict[str, object]]:
    latest: dict[int, dict[str, object]] = {}
    running_commands = _running_tmux_commands()
    run_roots = sorted(
        path for path in output_root.parent.glob("consensus-gold-v2*") if path.is_dir()
    )
    for run_root in run_roots:
        for scenario_dir in sorted(run_root.glob("scenario-*")):
            status = _read_pipeline_status(
                scenario_dir,
                run_root=run_root,
                running_commands=running_commands,
            )
            if status is None:
                continue
            scenario_index = int(status["scenario_index"])
            previous = latest.get(scenario_index)
            if previous is None or str(status["updated_at"]) > str(
                previous["updated_at"]
            ):
                latest[scenario_index] = status
    return [latest[index] for index in sorted(latest)]


def _read_pipeline_status(
    scenario_dir: Path,
    *,
    run_root: Path,
    running_commands: tuple[str, ...],
) -> dict[str, object] | None:
    try:
        scenario_index = int(scenario_dir.name.removeprefix("scenario-"))
    except ValueError:
        return None
    checkpoint = scenario_dir / "checkpoint.sqlite"
    if not checkpoint.is_file():
        return None
    try:
        uri = f"file:{checkpoint.as_posix()}?mode=ro"
        with sqlite3.connect(uri, uri=True, timeout=2) as connection:
            state = connection.execute(
                """
                SELECT status, next_turn_index, turn_count, updated_at
                FROM run_state WHERE singleton = 1
                """
            ).fetchone()
            counts = connection.execute(
                """
                SELECT
                  SUM(CASE WHEN json_extract(label_json, '$.decision') = 'UPDATE'
                           THEN 1 ELSE 0 END),
                  SUM(CASE WHEN json_extract(
                    label_json, '$.compaction_triggered') = 1 THEN 1 ELSE 0 END)
                FROM turn_results
                """
            ).fetchone()
        if state is None:
            return None
        status = str(state[0])
        if status == "RUNNING" and not _scenario_process_running(
            scenario_index,
            run_root=run_root,
            running_commands=running_commands,
        ):
            status = "STOPPED"
        done = int(state[1])
        total = int(state[2])
        return {
            "scenario_index": scenario_index,
            "status": status,
            "done": done,
            "total": total,
            "progress": min(100.0, done * 100 / total) if total else 0.0,
            "updates": int(counts[0] or 0),
            "compactions": int(counts[1] or 0),
            "qa_coverage": _qa_coverage_label(scenario_dir),
            "quiz_score": _quiz_score_label(scenario_dir, run_root=run_root),
            "updated_at": str(state[3]),
            "source_run": run_root.name,
        }
    except sqlite3.Error:
        return None


def _running_tmux_commands() -> tuple[str, ...]:
    result = subprocess.run(
        ["tmux", "list-panes", "-a", "-F", "#{pane_dead}\t#{pane_start_command}"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return ()
    return tuple(
        line.split("\t", 1)[1]
        for line in result.stdout.splitlines()
        if line.startswith("0\t") and "\t" in line
    )


def _scenario_process_running(
    scenario_index: int,
    *,
    run_root: Path,
    running_commands: tuple[str, ...],
) -> bool:
    scenario_pattern = re.compile(rf"--scenario\s+['\"]?{scenario_index}['\"]?(?:\s|$)")
    root_name = run_root.name
    return any(
        root_name in command and scenario_pattern.search(command)
        for command in running_commands
    )


def _related_queues(primary: HumanReviewQueue) -> tuple[HumanReviewQueue, ...]:
    queue_paths = sorted(
        path
        for path in primary.path.parent.parent.glob(
            "consensus-gold-v2*/review_queue.sqlite"
        )
        if path.is_file()
    )
    primary_path = primary.path.resolve()
    queues = [primary]
    queues.extend(
        HumanReviewQueue(path) for path in queue_paths if path.resolve() != primary_path
    )
    return tuple(queues)


def _find_review(
    primary: HumanReviewQueue,
    review_id: str,
    *,
    update_audit_queue: UpdateAuditReviewQueue | None = None,
) -> tuple[
    HumanReviewQueue | UpdateAuditReviewQueue,
    HumanReviewRecord | UpdateAuditReviewRecord,
]:
    for queue in _related_queues(primary):
        try:
            return queue, queue.get(review_id)
        except KeyError:
            continue
    if update_audit_queue is not None:
        try:
            return update_audit_queue, update_audit_queue.get(review_id)
        except KeyError:
            pass
    raise KeyError(f"Human review not found: {review_id}")


def _qa_coverage_label(scenario_dir: Path) -> str | None:
    path = scenario_dir / "qa_coverage_validation.json"
    if not path.is_file():
        return None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        counts = report.get("counts", {})
        supported = int(counts.get("SUPPORTED", 0))
        partial = int(counts.get("PARTIAL", 0))
        missing = int(counts.get("MISSING", 0))
        total = sum(int(value) for value in counts.values())
        usable = supported + partial
        recommendation = "REGENERATION_RECOMMENDED" if missing >= 4 else "PASS"
        coverage = usable * 100 / total if total else 0.0
        return (
            f"{usable}/{total} usable ({coverage:.0f}%) · "
            f"missing {missing} · {recommendation}"
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return "Unavailable"


def _quiz_score_label(scenario_dir: Path, *, run_root: Path) -> str | None:
    summary_path = scenario_dir / "run_summary.json"
    if not summary_path.is_file():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "COMPLETED":
            return None
        memory_sha256 = str(summary.get("final_memory_sha256") or "")
        if not memory_sha256:
            return None
        candidates: list[tuple[int, int, dict[str, object]]] = []
        manifest_pattern = f"*/{scenario_dir.name}/*/manifest.json"
        for manifest_path in run_root.parent.glob(manifest_pattern):
            metrics_path = manifest_path.with_name("metrics.json")
            if not metrics_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            snapshot = manifest.get("memory_snapshot", {})
            snapshot_sha256 = str(snapshot.get("memory_sha256") or "")
            if not re.fullmatch(r"[0-9a-f]{64}", snapshot_sha256):
                snapshot_sha256 = str(snapshot.get("cache_key") or "")
            if snapshot_sha256 != memory_sha256:
                continue
            source_run = str(snapshot.get("source_run") or "")
            if source_run and source_run != run_root.name:
                continue
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            profiles = metrics.get("profiles", {})
            profile = profiles.get(
                "cloud_turnwise_recursive_summary_patch_temporal_compact"
            )
            if not isinstance(profile, dict):
                continue
            exact_run_match = int(source_run == run_root.name)
            candidates.append(
                (exact_run_match, metrics_path.stat().st_mtime_ns, profile)
            )
        if not candidates:
            return None
        best_run_match = max(item[0] for item in candidates)
        eligible = [item for item in candidates if item[0] == best_run_match]
        _, _, profile = max(
            eligible,
            key=lambda item: (
                float(item[2]["exact_state_match"]),
                float(item[2]["argument_exact_match"]),
                -item[1],
            ),
        )
        esm = float(profile["exact_state_match"])
        argument_exact = float(profile["argument_exact_match"])
        return f"ESM {esm:.2f} / Arg {argument_exact:.2f}"
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return "Unavailable"


def _one(form: dict[str, list[str]], key: str) -> str:
    values = form.get(key)
    if not values or len(values) != 1:
        raise ValueError(f"Missing form field: {key}")
    return values[0]


def _escape(value: object) -> str:
    return html.escape(str(value))


def _review_kind(request: dict) -> str:
    if request.get("review_kind") == "UPDATE_AUDIT":
        return "UPDATE_AUDIT"
    return (
        request.get("pending_review", {}).get("payload", {}).get("review_kind", "TURN")
    )


if __name__ == "__main__":
    raise SystemExit(main())
