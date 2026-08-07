CREATE TABLE fact_memory_records (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    record_key TEXT NOT NULL,
    user_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    predicate TEXT NOT NULL,
    identity_conditions_json TEXT NOT NULL DEFAULT '{}',
    applicability_json TEXT NOT NULL DEFAULT '{}',
    capability_hints_json TEXT NOT NULL DEFAULT '[]',
    value_json TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    status TEXT NOT NULL
        CHECK(status IN (
            'active',
            'superseded',
            'merged',
            'deleted',
            'review'
        )),
    confidence REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
    version INTEGER NOT NULL CHECK(version > 0),
    supersedes_id TEXT REFERENCES fact_memory_records(id),
    merged_into_id TEXT REFERENCES fact_memory_records(id),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(record_key, version)
);

CREATE UNIQUE INDEX fact_memory_active_record_idx
    ON fact_memory_records(record_key)
    WHERE status = 'active';

CREATE INDEX fact_memory_partition_idx
    ON fact_memory_records(
        session_id,
        user_id,
        entity_id,
        predicate,
        status
    );

CREATE INDEX fact_memory_capability_idx
    ON fact_memory_records(session_id, status, updated_at);

CREATE TABLE fact_memory_record_sources (
    record_id TEXT NOT NULL
        REFERENCES fact_memory_records(id) ON DELETE CASCADE,
    message_id INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    evidence_text TEXT NOT NULL,
    start_char INTEGER,
    end_char INTEGER,
    created_at TEXT NOT NULL,
    PRIMARY KEY(record_id, message_id, evidence_text)
);

CREATE INDEX fact_memory_sources_message_idx
    ON fact_memory_record_sources(message_id, record_id);

CREATE TABLE fact_memory_patch_events (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    operation TEXT NOT NULL CHECK(operation IN ('ADD', 'UPDATE', 'MERGE', 'DELETE')),
    target_record_id TEXT REFERENCES fact_memory_records(id),
    merge_record_ids_json TEXT NOT NULL DEFAULT '[]',
    proposed_record_key TEXT,
    patch_json TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    confidence REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
    status TEXT NOT NULL
        CHECK(status IN ('pending', 'applied', 'rejected', 'review')),
    result_record_id TEXT REFERENCES fact_memory_records(id),
    reason TEXT,
    idempotency_key TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    resolved_at TEXT
);

CREATE INDEX fact_memory_patch_session_status_idx
    ON fact_memory_patch_events(session_id, status, created_at);

CREATE INDEX fact_memory_patch_target_idx
    ON fact_memory_patch_events(target_record_id, created_at);

CREATE TABLE fact_memory_record_status_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id TEXT NOT NULL
        REFERENCES fact_memory_records(id) ON DELETE CASCADE,
    status TEXT NOT NULL,
    reason TEXT,
    patch_event_id TEXT REFERENCES fact_memory_patch_events(id),
    created_at TEXT NOT NULL
);

CREATE INDEX fact_memory_status_events_record_idx
    ON fact_memory_record_status_events(record_id, id);
