CREATE TABLE compact_amem_episode_runs (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    episode_index INTEGER NOT NULL CHECK(episode_index >= 0),
    start_timestamp TEXT NOT NULL,
    end_timestamp TEXT NOT NULL,
    source_message_ids_json TEXT NOT NULL,
    backend TEXT NOT NULL,
    model_id TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL
        CHECK(status IN ('running', 'completed', 'failed', 'interrupted')),
    output_json TEXT,
    usage_json TEXT,
    error TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT
);

CREATE INDEX compact_amem_episode_runs_session_idx
    ON compact_amem_episode_runs(session_id, episode_index, started_at);

CREATE TABLE compact_amem_notes (
    note_id TEXT PRIMARY KEY REFERENCES amem_notes(id) ON DELETE CASCADE,
    episode_run_id TEXT NOT NULL
        REFERENCES compact_amem_episode_runs(id) ON DELETE CASCADE,
    note_ordinal INTEGER NOT NULL CHECK(note_ordinal >= 0),
    memory_kind TEXT NOT NULL CHECK(memory_kind IN (
        'preference', 'constraint', 'correction', 'commitment', 'profile',
        'stable_context'
    )),
    UNIQUE(episode_run_id, note_ordinal)
);

CREATE TABLE compact_amem_note_sources (
    note_id TEXT NOT NULL
        REFERENCES compact_amem_notes(note_id) ON DELETE CASCADE,
    source_message_id INTEGER NOT NULL CHECK(source_message_id >= 0),
    source_ordinal INTEGER NOT NULL CHECK(source_ordinal >= 0),
    PRIMARY KEY(note_id, source_message_id),
    UNIQUE(note_id, source_ordinal)
);

CREATE INDEX compact_amem_note_sources_source_idx
    ON compact_amem_note_sources(source_message_id, note_id);
