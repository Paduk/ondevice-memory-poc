CREATE TABLE fact_memory_embeddings (
    record_id TEXT NOT NULL
        REFERENCES fact_memory_records(id) ON DELETE CASCADE,
    model_id TEXT NOT NULL,
    dimensions INTEGER NOT NULL CHECK(dimensions > 0),
    vector_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(record_id, model_id, dimensions)
);

CREATE TABLE fact_memory_retrieval_runs (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    turn_id TEXT REFERENCES turns(id) ON DELETE SET NULL,
    query TEXT NOT NULL,
    user_id TEXT NOT NULL,
    mode TEXT NOT NULL,
    top_k INTEGER NOT NULL CHECK(top_k > 0),
    token_budget INTEGER NOT NULL CHECK(token_budget >= 0),
    query_context_json TEXT NOT NULL,
    route_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('running', 'completed', 'failed')),
    candidate_count INTEGER,
    selected_count INTEGER,
    selected_tokens INTEGER,
    latency_ms INTEGER,
    error TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT
);

CREATE INDEX fact_memory_retrieval_runs_session_idx
    ON fact_memory_retrieval_runs(session_id, started_at);

CREATE TABLE fact_memory_retrieval_candidates (
    retrieval_run_id TEXT NOT NULL
        REFERENCES fact_memory_retrieval_runs(id) ON DELETE CASCADE,
    record_id TEXT NOT NULL
        REFERENCES fact_memory_records(id) ON DELETE CASCADE,
    rank INTEGER,
    bm25_score REAL NOT NULL DEFAULT 0,
    embedding_score REAL NOT NULL DEFAULT 0,
    entity_score REAL NOT NULL DEFAULT 0,
    condition_score REAL NOT NULL DEFAULT 0,
    route_score REAL NOT NULL DEFAULT 0,
    combined_score REAL NOT NULL DEFAULT 0,
    selected INTEGER NOT NULL DEFAULT 0,
    exclusion_reason TEXT,
    token_count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(retrieval_run_id, record_id)
);

CREATE INDEX fact_memory_retrieval_candidates_rank_idx
    ON fact_memory_retrieval_candidates(
        retrieval_run_id,
        selected,
        rank
    );
