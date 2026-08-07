ALTER TABLE tool_memory_records
    ADD COLUMN entity_id TEXT;

ALTER TABLE tool_memory_records
    ADD COLUMN identity_conditions_json TEXT NOT NULL DEFAULT '{}';

ALTER TABLE tool_memory_records
    ADD COLUMN applicability_json TEXT NOT NULL DEFAULT '{}';

ALTER TABLE tool_memory_records
    ADD COLUMN identity_family_key TEXT;

CREATE INDEX tool_memory_records_entity_idx
    ON tool_memory_records(user_id, entity_id, tool_domain, topic, status);

CREATE INDEX tool_memory_records_family_idx
    ON tool_memory_records(identity_family_key, status, version);
