from __future__ import annotations

import json
from pathlib import Path

from palmclaw_ubuntu.vehicle_bench.v1_generation import load_persona_seeds
from palmclaw_ubuntu.vehicle_bench.v1_persona_source import (
    PERSONA_HUB_REPO,
    PERSONA_HUB_REVISION,
    PersonaHubChunk,
    PersonaHubShard,
    normalize_persona_hub_row,
    parse_persona_hub_chunk,
    write_persona_hub_sample,
)


def _shard() -> PersonaHubShard:
    return PersonaHubShard(
        path="ElitePersonas/elite_personas.part1.jsonl",
        size=10_000,
        oid="a" * 40,
        xet_hash="b" * 64,
    )


def test_parse_and_normalize_persona_hub_range_rows() -> None:
    data = (
        b'partial row\n{"persona":"A robotics researcher","domain":"CS"}\n'
        b'{"persona":"A historian","domain":"History"}\npartial row'
    )

    rows = parse_persona_hub_chunk(data)
    first = normalize_persona_hub_row(
        rows[0],
        shard=_shard(),
        byte_range=(100, 999),
    )
    repeated = normalize_persona_hub_row(
        rows[0],
        shard=_shard(),
        byte_range=(100, 999),
    )

    assert len(rows) == 2
    assert first == repeated
    assert first.source == f"{PERSONA_HUB_REPO}@{PERSONA_HUB_REVISION}"
    assert first.profile["source_shard"] == _shard().path


def test_sample_writer_preserves_license_and_loadable_seed_jsonl(
    tmp_path: Path,
) -> None:
    seeds = [
        normalize_persona_hub_row(
            {"persona": f"Persona {index}"},
            shard=_shard(),
            byte_range=(100, 999),
        )
        for index in range(2)
    ]
    chunk = PersonaHubChunk(
        shard=_shard(),
        start=100,
        end=999,
        sha256="c" * 64,
        parsed_rows=2,
    )

    seed_path, manifest_path = write_persona_hub_sample(
        tmp_path,
        seeds=seeds,
        chunks=[chunk],
        sample_count=2,
        chunk_bytes=900,
        sampling_seed=7,
        candidate_count=2,
    )
    manifest = json.loads(manifest_path.read_text())

    assert load_persona_seeds(seed_path) == tuple(seeds)
    assert manifest["source_license"] == "cc-by-nc-sa-4.0"
    assert manifest["sampling_provenance"] == "project_defined"
