ALTER TABLE memory_patch_proposals
    ADD COLUMN validation_json TEXT NOT NULL DEFAULT '{}';

ALTER TABLE memory_patch_proposals
    ADD COLUMN sequence_index INTEGER;

CREATE UNIQUE INDEX memory_patch_proposals_run_sequence_idx
    ON memory_patch_proposals(run_id, sequence_index)
    WHERE sequence_index IS NOT NULL;

CREATE TABLE tool_memory_record_status_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id TEXT NOT NULL
        REFERENCES tool_memory_records(id) ON DELETE CASCADE,
    status TEXT NOT NULL,
    reason TEXT,
    patch_proposal_id TEXT REFERENCES memory_patch_proposals(id),
    created_at TEXT NOT NULL
);

CREATE INDEX tool_memory_status_events_record_idx
    ON tool_memory_record_status_events(record_id, id);

CREATE INDEX tool_memory_status_events_proposal_idx
    ON tool_memory_record_status_events(patch_proposal_id, id);

INSERT INTO tool_memory_record_status_events(
    record_id,
    status,
    reason,
    patch_proposal_id,
    created_at
)
SELECT
    id,
    status,
    'migration_backfill',
    NULL,
    updated_at
FROM tool_memory_records;
