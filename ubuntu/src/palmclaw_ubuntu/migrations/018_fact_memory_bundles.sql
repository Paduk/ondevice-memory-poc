ALTER TABLE fact_memory_records
    ADD COLUMN bundle_id TEXT;

UPDATE fact_memory_records
SET bundle_id = COALESCE(
    (
        SELECT 'message:' || MIN(source.message_id)
        FROM fact_memory_record_sources AS source
        WHERE source.record_id = fact_memory_records.id
    ),
    'fact:' || id
);

CREATE INDEX fact_memory_bundle_idx
    ON fact_memory_records(session_id, bundle_id, status);
