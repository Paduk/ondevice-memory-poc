CREATE TABLE tool_schema_embeddings (
    tool_name TEXT NOT NULL,
    schema_fingerprint TEXT NOT NULL,
    model_id TEXT NOT NULL,
    dimensions INTEGER NOT NULL CHECK(dimensions > 0),
    vector_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(tool_name, schema_fingerprint, model_id, dimensions)
);

CREATE INDEX tool_schema_embeddings_model_idx
    ON tool_schema_embeddings(model_id, dimensions, tool_name);
