CREATE TABLE memory_gate_decisions (
    memory_id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    decision TEXT NOT NULL,
    evidence_relation TEXT NOT NULL,
    conflict_relation TEXT NOT NULL,
    reason TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    pii_categories_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX memory_gate_decisions_decision_idx
    ON memory_gate_decisions(decision, created_at);

CREATE INDEX memories_review_queue_idx
    ON memories(status, created_at)
    WHERE status = 'review';
