CREATE TABLE memory_patch_jobs (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    source_turn_id TEXT NOT NULL UNIQUE
        REFERENCES turns(id) ON DELETE CASCADE,
    source_message_cursor INTEGER NOT NULL,
    status TEXT NOT NULL
        CHECK(status IN (
            'queued',
            'running',
            'retryable',
            'completed',
            'failed'
        )),
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK(attempt_count >= 0),
    max_attempts INTEGER NOT NULL CHECK(max_attempts > 0),
    available_at TEXT NOT NULL,
    claimed_at TEXT,
    lease_expires_at TEXT,
    completed_at TEXT,
    last_error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX memory_patch_jobs_ready_idx
    ON memory_patch_jobs(status, available_at, source_message_cursor);

CREATE INDEX memory_patch_jobs_session_idx
    ON memory_patch_jobs(session_id, status, source_message_cursor);

ALTER TABLE memory_patch_runs
    ADD COLUMN job_id TEXT REFERENCES memory_patch_jobs(id) ON DELETE SET NULL;

CREATE INDEX memory_patch_runs_job_idx
    ON memory_patch_runs(job_id, started_at);

ALTER TABLE model_calls
    ADD COLUMN memory_patch_run_id TEXT
        REFERENCES memory_patch_runs(id) ON DELETE SET NULL;

CREATE INDEX model_calls_memory_patch_run_idx
    ON model_calls(memory_patch_run_id, created_at);
