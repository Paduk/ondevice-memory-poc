"""Reproducible bounded sampling from the official Persona-Hub Elite shards."""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

from palmclaw_ubuntu.vehicle_bench.v1_generation import V1PersonaSeed
from palmclaw_ubuntu.vehicle_bench.v1_reproduction import canonical_json_sha256

PERSONA_HUB_REPO = "proj-persona/PersonaHub"
PERSONA_HUB_REVISION = "16777e34bf5cb758b925cae5d84e868ee6c2100c"
PERSONA_HUB_LICENSE = "cc-by-nc-sa-4.0"
PERSONA_HUB_ELITE_DIRECTORY = "ElitePersonas"
PERSONA_HUB_SAMPLE_VERSION = "personahub-elite-stratified-range-sample-v1"

_PART_NUMBER = re.compile(r"part(\d+)\.jsonl$")


@dataclass(frozen=True)
class PersonaHubShard:
    path: str
    size: int
    oid: str
    xet_hash: str


@dataclass(frozen=True)
class PersonaHubChunk:
    shard: PersonaHubShard
    start: int
    end: int
    sha256: str
    parsed_rows: int


def sample_persona_hub_elite(
    output_dir: Path | str,
    *,
    sample_count: int = 600,
    chunk_bytes: int = 1_048_576,
    sampling_seed: int = 260323840,
    timeout_seconds: float = 60.0,
    client: Any | None = None,
) -> tuple[Path, Path]:
    """Sample every official shard without downloading the 301 GB dataset."""

    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if chunk_bytes < 65_536:
        raise ValueError("chunk_bytes must be at least 65536")
    if client is None:
        import httpx

        client = httpx.Client(follow_redirects=True, timeout=timeout_seconds)
        close_client = True
    else:
        close_client = False
    try:
        shards = fetch_persona_hub_shards(client)
        rng = random.Random(sampling_seed)
        chunks: list[PersonaHubChunk] = []
        candidates: dict[str, V1PersonaSeed] = {}
        for shard in shards:
            if shard.size <= chunk_bytes + 2:
                raise ValueError(
                    f"Persona-Hub shard is unexpectedly small: {shard.path}"
                )
            start = rng.randrange(1, shard.size - chunk_bytes - 1)
            end = start + chunk_bytes - 1
            data = fetch_persona_hub_range(
                client,
                shard=shard,
                start=start,
                end=end,
            )
            rows = parse_persona_hub_chunk(data)
            chunks.append(
                PersonaHubChunk(
                    shard=shard,
                    start=start,
                    end=end,
                    sha256=hashlib.sha256(data).hexdigest(),
                    parsed_rows=len(rows),
                )
            )
            for row in rows:
                normalized = normalize_persona_hub_row(
                    row,
                    shard=shard,
                    byte_range=(start, end),
                )
                candidates.setdefault(normalized.seed_id, normalized)
        if len(candidates) < sample_count:
            raise ValueError(
                f"only {len(candidates)} unique Persona-Hub rows were sampled; "
                f"need {sample_count}"
            )
        ordered = sorted(candidates.values(), key=lambda item: item.seed_id)
        selected = rng.sample(ordered, sample_count)
        selected.sort(key=lambda item: item.seed_id)
        return write_persona_hub_sample(
            output_dir,
            seeds=selected,
            chunks=chunks,
            sample_count=sample_count,
            chunk_bytes=chunk_bytes,
            sampling_seed=sampling_seed,
            candidate_count=len(candidates),
        )
    finally:
        if close_client:
            client.close()


def fetch_persona_hub_shards(client: Any) -> tuple[PersonaHubShard, ...]:
    url = (
        f"https://huggingface.co/api/datasets/{PERSONA_HUB_REPO}/tree/"
        f"{PERSONA_HUB_REVISION}/{PERSONA_HUB_ELITE_DIRECTORY}"
        "?recursive=false&expand=false&limit=100"
    )
    response = client.get(url)
    response.raise_for_status()
    raw = response.json()
    if not isinstance(raw, list):
        raise ValueError("Persona-Hub tree API did not return a list")
    shards = []
    for item in raw:
        if not isinstance(item, dict) or item.get("type") != "file":
            continue
        path = item.get("path")
        size = item.get("size")
        oid = item.get("oid")
        xet_hash = item.get("xetHash")
        if not isinstance(path, str) or _PART_NUMBER.search(path) is None:
            continue
        if not isinstance(size, int) or not isinstance(oid, str):
            raise ValueError(f"invalid Persona-Hub shard metadata: {item}")
        if not isinstance(xet_hash, str):
            raise ValueError(f"Persona-Hub shard is missing xetHash: {path}")
        shards.append(
            PersonaHubShard(
                path=path,
                size=size,
                oid=oid,
                xet_hash=xet_hash,
            )
        )
    shards.sort(key=lambda item: int(_PART_NUMBER.search(item.path).group(1)))
    if len(shards) != 19:
        raise ValueError(f"expected 19 Persona-Hub Elite shards, found {len(shards)}")
    return tuple(shards)


def fetch_persona_hub_range(
    client: Any,
    *,
    shard: PersonaHubShard,
    start: int,
    end: int,
) -> bytes:
    if start < 0 or end < start or end >= shard.size:
        raise ValueError("invalid Persona-Hub byte range")
    encoded_path = quote(shard.path, safe="/")
    url = (
        f"https://huggingface.co/datasets/{PERSONA_HUB_REPO}/resolve/"
        f"{PERSONA_HUB_REVISION}/{encoded_path}"
    )
    response = client.get(url, headers={"Range": f"bytes={start}-{end}"})
    response.raise_for_status()
    if response.status_code != 206:
        raise RuntimeError(
            f"Persona-Hub server ignored bounded range for {shard.path}: "
            f"status={response.status_code}"
        )
    expected = end - start + 1
    if len(response.content) != expected:
        raise RuntimeError(
            f"Persona-Hub range length mismatch for {shard.path}: "
            f"{len(response.content)} != {expected}"
        )
    return bytes(response.content)


def parse_persona_hub_chunk(data: bytes) -> tuple[dict[str, Any], ...]:
    lines = data.splitlines()
    if len(lines) < 3:
        raise ValueError("Persona-Hub chunk does not contain complete JSONL rows")
    rows: list[dict[str, Any]] = []
    for raw in lines[1:-1]:
        try:
            item = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid Persona-Hub JSONL row: {exc}") from exc
        if not isinstance(item, dict):
            raise ValueError("Persona-Hub JSONL row is not an object")
        persona = item.get("persona")
        if not isinstance(persona, str) or not persona.strip():
            raise ValueError("Persona-Hub row has no persona text")
        rows.append(item)
    return tuple(rows)


def normalize_persona_hub_row(
    row: dict[str, Any],
    *,
    shard: PersonaHubShard,
    byte_range: tuple[int, int],
) -> V1PersonaSeed:
    normalized = json.loads(json.dumps(row, ensure_ascii=False, sort_keys=True))
    row_hash = canonical_json_sha256(normalized)
    return V1PersonaSeed(
        seed_id=f"personahub-{row_hash[:24]}",
        source=f"{PERSONA_HUB_REPO}@{PERSONA_HUB_REVISION}",
        profile={
            "persona_hub_record": normalized,
            "source_shard": shard.path,
            "source_byte_range": [byte_range[0], byte_range[1]],
            "source_row_sha256": row_hash,
        },
    )


def write_persona_hub_sample(
    output_dir: Path | str,
    *,
    seeds: list[V1PersonaSeed],
    chunks: list[PersonaHubChunk],
    sample_count: int,
    chunk_bytes: int,
    sampling_seed: int,
    candidate_count: int,
) -> tuple[Path, Path]:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    seed_path = root / "persona_seeds.jsonl"
    manifest_path = root / "manifest.json"
    seed_text = "".join(
        json.dumps(
            seed.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
        for seed in seeds
    )
    seed_path.write_text(seed_text, encoding="utf-8")
    manifest = {
        "version": PERSONA_HUB_SAMPLE_VERSION,
        "source_repo": PERSONA_HUB_REPO,
        "source_revision": PERSONA_HUB_REVISION,
        "source_subset": "elite_persona",
        "source_license": PERSONA_HUB_LICENSE,
        "sampling_method": "one deterministic byte-range per official shard",
        "sampling_provenance": "project_defined",
        "sampling_seed": sampling_seed,
        "chunk_bytes": chunk_bytes,
        "candidate_count": candidate_count,
        "selected_count": sample_count,
        "seed_file_sha256": hashlib.sha256(seed_text.encode("utf-8")).hexdigest(),
        "selected_seed_ids_sha256": canonical_json_sha256(
            [seed.seed_id for seed in seeds]
        ),
        "chunks": [
            {
                **asdict(chunk),
                "shard": asdict(chunk.shard),
            }
            for chunk in chunks
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return seed_path, manifest_path
