CREATE TABLE evaluation_runs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    dataset_version TEXT NOT NULL,
    dataset_sha256 TEXT NOT NULL,
    execution_mode TEXT NOT NULL,
    profiles_json TEXT NOT NULL,
    seed INTEGER NOT NULL,
    repetitions INTEGER NOT NULL,
    status TEXT NOT NULL,
    metrics_json TEXT,
    artifact_dir TEXT,
    error TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT
);

CREATE INDEX evaluation_runs_started_idx
    ON evaluation_runs(started_at);

CREATE TABLE evaluation_cases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    evaluation_run_id TEXT NOT NULL
        REFERENCES evaluation_runs(id) ON DELETE CASCADE,
    profile TEXT NOT NULL,
    case_id TEXT NOT NULL,
    repetition INTEGER NOT NULL,
    status TEXT NOT NULL,
    latency_ms INTEGER NOT NULL,
    expected_json TEXT NOT NULL,
    actual_json TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    error TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(evaluation_run_id, profile, case_id, repetition)
);

CREATE INDEX evaluation_cases_run_profile_idx
    ON evaluation_cases(evaluation_run_id, profile, case_id);
