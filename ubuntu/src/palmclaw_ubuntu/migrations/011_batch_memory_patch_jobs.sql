ALTER TABLE memory_patch_jobs
    ADD COLUMN batch_run_id TEXT
        REFERENCES memory_patch_runs(id) ON DELETE SET NULL;

CREATE INDEX memory_patch_jobs_batch_run_idx
    ON memory_patch_jobs(batch_run_id, source_message_cursor);
