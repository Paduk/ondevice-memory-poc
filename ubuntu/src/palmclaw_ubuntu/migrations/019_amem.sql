CREATE TABLE amem_notes (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    source_message_id INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    speaker TEXT NOT NULL,
    content TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('active', 'deleted')),
    created_at TEXT NOT NULL,
    UNIQUE(session_id, source_message_id),
    UNIQUE(session_id, id)
);

CREATE INDEX amem_notes_session_source_idx
    ON amem_notes(session_id, source_message_id);

CREATE TABLE amem_construction_runs (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    source_message_id INTEGER NOT NULL,
    backend TEXT NOT NULL,
    model_id TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL
        CHECK(status IN ('running', 'completed', 'failed', 'interrupted')),
    note_id TEXT,
    output_json TEXT,
    usage_json TEXT,
    error TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    FOREIGN KEY(session_id, note_id)
        REFERENCES amem_notes(session_id, id)
);

CREATE INDEX amem_construction_runs_session_idx
    ON amem_construction_runs(session_id, source_message_id, started_at);

CREATE TABLE amem_evolution_events (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    new_note_id TEXT NOT NULL,
    backend TEXT NOT NULL,
    model_id TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL
        CHECK(status IN ('running', 'completed', 'failed', 'interrupted')),
    decision_json TEXT,
    usage_json TEXT,
    error TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    FOREIGN KEY(session_id, new_note_id)
        REFERENCES amem_notes(session_id, id) ON DELETE CASCADE
);

CREATE INDEX amem_evolution_events_note_idx
    ON amem_evolution_events(session_id, new_note_id, started_at);

CREATE TABLE amem_note_versions (
    id TEXT PRIMARY KEY,
    note_id TEXT NOT NULL REFERENCES amem_notes(id) ON DELETE CASCADE,
    version INTEGER NOT NULL CHECK(version > 0),
    context TEXT NOT NULL,
    keywords_json TEXT NOT NULL DEFAULT '[]',
    tags_json TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL CHECK(status IN ('active', 'superseded')),
    supersedes_id TEXT REFERENCES amem_note_versions(id),
    created_by_event_id TEXT REFERENCES amem_evolution_events(id),
    created_at TEXT NOT NULL,
    UNIQUE(note_id, version),
    UNIQUE(note_id, id)
);

CREATE UNIQUE INDEX amem_note_active_version_idx
    ON amem_note_versions(note_id)
    WHERE status = 'active';

CREATE INDEX amem_note_versions_note_idx
    ON amem_note_versions(note_id, version);

CREATE TABLE amem_note_embeddings (
    note_version_id TEXT NOT NULL
        REFERENCES amem_note_versions(id) ON DELETE CASCADE,
    model_id TEXT NOT NULL,
    dimensions INTEGER NOT NULL CHECK(dimensions > 0),
    vector_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(note_version_id, model_id, dimensions)
);

CREATE TABLE amem_links (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    left_note_id TEXT NOT NULL,
    right_note_id TEXT NOT NULL,
    created_by_note_id TEXT NOT NULL REFERENCES amem_notes(id),
    evolution_event_id TEXT REFERENCES amem_evolution_events(id),
    similarity_score REAL,
    decision TEXT NOT NULL,
    created_at TEXT NOT NULL,
    CHECK(left_note_id < right_note_id),
    CHECK(created_by_note_id = left_note_id OR created_by_note_id = right_note_id),
    UNIQUE(left_note_id, right_note_id),
    FOREIGN KEY(session_id, left_note_id)
        REFERENCES amem_notes(session_id, id) ON DELETE CASCADE,
    FOREIGN KEY(session_id, right_note_id)
        REFERENCES amem_notes(session_id, id) ON DELETE CASCADE
);

CREATE INDEX amem_links_session_left_idx
    ON amem_links(session_id, left_note_id);

CREATE INDEX amem_links_session_right_idx
    ON amem_links(session_id, right_note_id);

CREATE TABLE amem_retrieval_runs (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    query TEXT NOT NULL,
    mode TEXT NOT NULL,
    top_k INTEGER NOT NULL CHECK(top_k > 0),
    token_budget INTEGER NOT NULL CHECK(token_budget >= 0),
    embedding_model_id TEXT,
    embedding_dimensions INTEGER CHECK(embedding_dimensions > 0),
    status TEXT NOT NULL
        CHECK(status IN ('running', 'completed', 'failed', 'interrupted')),
    candidate_count INTEGER,
    selected_count INTEGER,
    selected_tokens INTEGER,
    latency_ms INTEGER,
    error TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT
);

CREATE INDEX amem_retrieval_runs_session_idx
    ON amem_retrieval_runs(session_id, started_at);

CREATE TABLE amem_retrieval_candidates (
    retrieval_run_id TEXT NOT NULL
        REFERENCES amem_retrieval_runs(id) ON DELETE CASCADE,
    note_id TEXT NOT NULL REFERENCES amem_notes(id) ON DELETE CASCADE,
    note_version_id TEXT NOT NULL
        REFERENCES amem_note_versions(id) ON DELETE CASCADE,
    source TEXT NOT NULL CHECK(source IN ('seed', 'linked')),
    rank INTEGER,
    embedding_score REAL NOT NULL DEFAULT 0,
    combined_score REAL NOT NULL DEFAULT 0,
    selected INTEGER NOT NULL DEFAULT 0 CHECK(selected IN (0, 1)),
    exclusion_reason TEXT,
    token_count INTEGER NOT NULL DEFAULT 0 CHECK(token_count >= 0),
    PRIMARY KEY(retrieval_run_id, note_id),
    FOREIGN KEY(note_id, note_version_id)
        REFERENCES amem_note_versions(note_id, id) ON DELETE CASCADE
);

CREATE INDEX amem_retrieval_candidates_rank_idx
    ON amem_retrieval_candidates(retrieval_run_id, selected, rank);
