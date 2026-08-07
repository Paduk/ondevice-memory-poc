DROP INDEX IF EXISTS memories_session_scope_version_idx;

ALTER TABLE memories
    ADD COLUMN subject TEXT;

ALTER TABLE memories
    ADD COLUMN predicate TEXT;

ALTER TABLE memories
    ADD COLUMN value_json TEXT;

ALTER TABLE memories
    ADD COLUMN fact_key TEXT;

ALTER TABLE memories
    ADD COLUMN memory_type TEXT;

ALTER TABLE memories
    ADD COLUMN confidence REAL;

ALTER TABLE memories
    ADD COLUMN sensitivity TEXT;

ALTER TABLE memories
    ADD COLUMN supersedes_id TEXT REFERENCES memories(id);

ALTER TABLE memories
    ADD COLUMN consolidation_run_id TEXT REFERENCES consolidation_runs(id);

ALTER TABLE memories
    ADD COLUMN rejection_reason TEXT;

ALTER TABLE memories
    ADD COLUMN model_id TEXT;

ALTER TABLE memories
    ADD COLUMN schema_version TEXT;

CREATE UNIQUE INDEX memories_fact_version_idx
    ON memories(fact_key, version)
    WHERE fact_key IS NOT NULL;

CREATE UNIQUE INDEX memories_active_fact_idx
    ON memories(fact_key)
    WHERE fact_key IS NOT NULL AND status = 'verified';

CREATE INDEX memories_retrieval_idx
    ON memories(status, scope, session_id);

CREATE TABLE memory_sources (
    memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    message_id INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    evidence_text TEXT NOT NULL,
    start_char INTEGER,
    end_char INTEGER,
    created_at TEXT NOT NULL,
    PRIMARY KEY(memory_id, message_id, evidence_text)
);

CREATE INDEX memory_sources_message_idx
    ON memory_sources(message_id, memory_id);

CREATE TABLE memory_status_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    status TEXT NOT NULL,
    reason TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX memory_status_events_memory_idx
    ON memory_status_events(memory_id, id);

CREATE TABLE memory_embeddings (
    memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    model_id TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    vector_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(memory_id, model_id, dimensions)
);

CREATE TABLE retrieval_runs (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    turn_id TEXT REFERENCES turns(id) ON DELETE SET NULL,
    query TEXT NOT NULL,
    mode TEXT NOT NULL,
    top_k INTEGER NOT NULL,
    token_budget INTEGER NOT NULL,
    embedding_backend TEXT,
    embedding_model_id TEXT,
    selected_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    error TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT
);

CREATE INDEX retrieval_runs_session_started_idx
    ON retrieval_runs(session_id, started_at);

CREATE TABLE retrieval_candidates (
    retrieval_run_id TEXT NOT NULL
        REFERENCES retrieval_runs(id) ON DELETE CASCADE,
    memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    rank INTEGER,
    bm25_score REAL NOT NULL DEFAULT 0,
    embedding_score REAL NOT NULL DEFAULT 0,
    combined_score REAL NOT NULL DEFAULT 0,
    selected INTEGER NOT NULL DEFAULT 0,
    exclusion_reason TEXT,
    token_count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(retrieval_run_id, memory_id)
);

CREATE INDEX retrieval_candidates_rank_idx
    ON retrieval_candidates(retrieval_run_id, selected, rank);
