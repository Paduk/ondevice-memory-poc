CREATE TABLE tool_memory_records (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    record_key TEXT NOT NULL,
    user_id TEXT NOT NULL,
    tool_domain TEXT NOT NULL,
    topic TEXT NOT NULL,
    scope TEXT NOT NULL
        CHECK(scope IN ('global', 'vehicle', 'session', 'conditional')),
    scope_key TEXT NOT NULL,
    conditions_json TEXT NOT NULL,
    value_json TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    status TEXT NOT NULL
        CHECK(status IN (
            'active',
            'superseded',
            'merged',
            'deleted',
            'rejected'
        )),
    confidence REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
    version INTEGER NOT NULL CHECK(version > 0),
    supersedes_id TEXT REFERENCES tool_memory_records(id),
    merged_into_id TEXT REFERENCES tool_memory_records(id),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(record_key, version)
);

CREATE UNIQUE INDEX tool_memory_active_record_idx
    ON tool_memory_records(record_key)
    WHERE status = 'active';

CREATE INDEX tool_memory_partition_idx
    ON tool_memory_records(
        user_id,
        tool_domain,
        topic,
        scope,
        scope_key,
        status
    );

CREATE INDEX tool_memory_session_status_idx
    ON tool_memory_records(session_id, status, updated_at);

CREATE TABLE tool_memory_record_sources (
    record_id TEXT NOT NULL
        REFERENCES tool_memory_records(id) ON DELETE CASCADE,
    message_id INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    evidence_text TEXT NOT NULL,
    start_char INTEGER,
    end_char INTEGER,
    created_at TEXT NOT NULL,
    PRIMARY KEY(record_id, message_id, evidence_text)
);

CREATE INDEX tool_memory_sources_message_idx
    ON tool_memory_record_sources(message_id, record_id);

CREATE TABLE memory_patch_runs (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    source_turn_id TEXT REFERENCES turns(id) ON DELETE SET NULL,
    backend TEXT NOT NULL,
    model_id TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    status TEXT NOT NULL
        CHECK(status IN (
            'queued',
            'running',
            'completed',
            'failed',
            'interrupted'
        )),
    error TEXT,
    usage_json TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT
);

CREATE INDEX memory_patch_runs_session_status_idx
    ON memory_patch_runs(session_id, status, started_at);

CREATE INDEX memory_patch_runs_source_turn_idx
    ON memory_patch_runs(source_turn_id, started_at);

CREATE TABLE memory_patch_proposals (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES memory_patch_runs(id) ON DELETE CASCADE,
    operation TEXT NOT NULL CHECK(operation IN ('ADD', 'UPDATE', 'MERGE', 'DELETE')),
    target_record_id TEXT REFERENCES tool_memory_records(id),
    proposed_record_key TEXT,
    patch_json TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    confidence REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
    status TEXT NOT NULL
        CHECK(status IN ('pending', 'applied', 'rejected')),
    result_record_id TEXT REFERENCES tool_memory_records(id),
    rejection_reason TEXT,
    idempotency_key TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    applied_at TEXT
);

CREATE INDEX memory_patch_proposals_run_status_idx
    ON memory_patch_proposals(run_id, status, created_at);

CREATE INDEX memory_patch_proposals_target_idx
    ON memory_patch_proposals(target_record_id, created_at);
