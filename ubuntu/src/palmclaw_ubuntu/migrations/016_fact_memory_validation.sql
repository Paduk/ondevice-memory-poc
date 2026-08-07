ALTER TABLE fact_memory_candidate_events
    RENAME TO fact_memory_candidate_events_r2;

CREATE TABLE fact_memory_candidate_events (
    id TEXT PRIMARY KEY,
    memory_patch_run_id TEXT NOT NULL
        REFERENCES memory_patch_runs(id) ON DELETE CASCADE,
    sequence_index INTEGER NOT NULL CHECK(sequence_index >= 0),
    candidate_json TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    decision TEXT NOT NULL
        CHECK(decision IN (
            'ADD',
            'UPDATE',
            'MERGE',
            'DELETE',
            'NOOP',
            'REVIEW',
            'REJECT'
        )),
    status TEXT NOT NULL
        CHECK(status IN ('applied', 'noop', 'review', 'rejected')),
    target_record_id TEXT REFERENCES fact_memory_records(id),
    merge_record_ids_json TEXT NOT NULL DEFAULT '[]',
    result_record_id TEXT REFERENCES fact_memory_records(id),
    match_score REAL,
    reason TEXT,
    validation_json TEXT NOT NULL DEFAULT '{}',
    idempotency_key TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    UNIQUE(memory_patch_run_id, sequence_index)
);

INSERT INTO fact_memory_candidate_events(
    id,
    memory_patch_run_id,
    sequence_index,
    candidate_json,
    evidence_json,
    decision,
    status,
    target_record_id,
    merge_record_ids_json,
    result_record_id,
    match_score,
    reason,
    validation_json,
    idempotency_key,
    created_at
)
SELECT
    id,
    memory_patch_run_id,
    sequence_index,
    candidate_json,
    evidence_json,
    decision,
    status,
    target_record_id,
    merge_record_ids_json,
    result_record_id,
    match_score,
    reason,
    '{}',
    idempotency_key,
    created_at
FROM fact_memory_candidate_events_r2;

DROP TABLE fact_memory_candidate_events_r2;

CREATE INDEX fact_memory_candidate_run_idx
    ON fact_memory_candidate_events(memory_patch_run_id, sequence_index);

CREATE INDEX fact_memory_candidate_result_idx
    ON fact_memory_candidate_events(result_record_id, created_at);
