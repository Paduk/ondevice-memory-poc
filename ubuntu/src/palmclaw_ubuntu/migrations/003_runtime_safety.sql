ALTER TABLE tool_calls
    ADD COLUMN side_effect TEXT NOT NULL DEFAULT 'none';

ALTER TABLE tool_calls
    ADD COLUMN retry_safety TEXT NOT NULL DEFAULT 'safe';

ALTER TABLE tool_calls
    ADD COLUMN fingerprint TEXT;

CREATE INDEX tool_calls_fingerprint_idx
    ON tool_calls(turn_id, fingerprint, status);

