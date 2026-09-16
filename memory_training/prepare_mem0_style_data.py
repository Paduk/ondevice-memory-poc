"""Export one-pass and two-stage Mem0-style SFT/evaluation views.

The source Patch trajectory remains the authority.  This exporter replays each
scenario, assigns stable record ids to grouped Markdown facts, and emits:

* ``fact_extraction.jsonl``: turn -> newly extracted facts;
* ``fact_manager.jsonl``: extracted facts + retrieved records -> CRUD events;
* ``joint_fact_update.jsonl``: turn + retrieved records -> facts and CRUD events;
* ``facts.jsonl``: immutable fact-version registry referenced by the views.

All source rows are retained.  Training code must filter ``split == "train"``
and ``train_eligible``; validation and test rows are chronological evaluation
trajectories and must not be shuffled during closed-loop evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .methods.operations import apply_operations, normalize_memory

SCHEMA_VERSION = "palmclaw-mem0-style-one-two-stage-v1"
EXTRACTION_SCHEMA_VERSION = "palmclaw-fact-extraction-sft-v1"
MANAGER_SCHEMA_VERSION = "palmclaw-fact-manager-sft-v1"
JOINT_SCHEMA_VERSION = "palmclaw-joint-fact-update-sft-v1"
FACT_SCHEMA_VERSION = "palmclaw-fact-version-registry-v1"

DEFAULT_SOURCE = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/"
    "grouped-s1-s100-plus-temporal-t1-t20-v2"
)
DEFAULT_OUTPUT = Path(
    "/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-v2-training/"
    "mem0-style-one-two-stage-s21t10-no-v1-v2"
)
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class AtomicFact:
    subject: str
    text: str

    @property
    def embedding_text(self) -> str:
        return f"{self.subject}: {self.text}"


@dataclass(frozen=True)
class FactVersion:
    fact_version_id: str
    record_id: str
    scenario_index: int
    version: int
    subject: str
    text: str
    created_sample_id: str
    supersedes_fact_version_id: str | None

    @property
    def fact(self) -> AtomicFact:
        return AtomicFact(self.subject, self.text)


@dataclass
class DerivedTurn:
    row: dict[str, Any]
    extracted_facts: list[dict[str, Any]]
    events: list[dict[str, Any]]
    event_metadata: list[dict[str, Any]]
    active_before: tuple[str, ...]
    active_after: tuple[str, ...]
    pure_delete_fallback: bool


class ScenarioFactState:
    """Replay one scenario as immutable versions of stable fact records."""

    def __init__(self, scenario_index: int) -> None:
        self.scenario_index = scenario_index
        self._next_record = 1
        self._versions: dict[str, FactVersion] = {}
        self._active_by_fact: dict[AtomicFact, str] = {}
        self._version_by_record: dict[str, int] = {}

    @property
    def versions(self) -> tuple[FactVersion, ...]:
        return tuple(self._versions.values())

    @property
    def active_version_ids(self) -> tuple[str, ...]:
        return tuple(self._active_by_fact.values())

    def version(self, fact_version_id: str) -> FactVersion:
        return self._versions[fact_version_id]

    def active_facts(self) -> tuple[AtomicFact, ...]:
        return tuple(self._active_by_fact)

    def add(self, fact: AtomicFact, *, sample_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        if fact in self._active_by_fact:
            raise ValueError(f"Fact already active: {fact}")
        record_id = f"s{self.scenario_index:03d}:f{self._next_record:04d}"
        self._next_record += 1
        version = self._new_version(
            record_id=record_id,
            fact=fact,
            sample_id=sample_id,
            supersedes=None,
        )
        self._active_by_fact[fact] = version.fact_version_id
        return (
            {"event": "ADD", "subject": fact.subject, "text": fact.text},
            {
                "event": "ADD",
                "gold_record_id": record_id,
                "new_fact_version_id": version.fact_version_id,
            },
        )

    def update(
        self,
        old: AtomicFact,
        new: AtomicFact,
        *,
        sample_id: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        old_version_id = self._require_active(old)
        old_version = self._versions[old_version_id]
        if new in self._active_by_fact and new != old:
            raise ValueError(f"Replacement fact already active: {new}")
        del self._active_by_fact[old]
        version = self._new_version(
            record_id=old_version.record_id,
            fact=new,
            sample_id=sample_id,
            supersedes=old_version_id,
        )
        self._active_by_fact[new] = version.fact_version_id
        return (
            {
                "event": "UPDATE",
                "id": old_version.record_id,
                "subject": new.subject,
                "text": new.text,
            },
            {
                "event": "UPDATE",
                "gold_record_id": old_version.record_id,
                "old_fact_version_id": old_version_id,
                "new_fact_version_id": version.fact_version_id,
            },
        )

    def delete(
        self, fact: AtomicFact
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        old_version_id = self._require_active(fact)
        old_version = self._versions[old_version_id]
        del self._active_by_fact[fact]
        return (
            {"event": "DELETE", "id": old_version.record_id},
            {
                "event": "DELETE",
                "gold_record_id": old_version.record_id,
                "old_fact_version_id": old_version_id,
            },
        )

    def _new_version(
        self,
        *,
        record_id: str,
        fact: AtomicFact,
        sample_id: str,
        supersedes: str | None,
    ) -> FactVersion:
        number = self._version_by_record.get(record_id, 0) + 1
        self._version_by_record[record_id] = number
        fact_version_id = f"{record_id}:v{number:02d}"
        version = FactVersion(
            fact_version_id=fact_version_id,
            record_id=record_id,
            scenario_index=self.scenario_index,
            version=number,
            subject=fact.subject,
            text=fact.text,
            created_sample_id=sample_id,
            supersedes_fact_version_id=supersedes,
        )
        self._versions[fact_version_id] = version
        return version

    def _require_active(self, fact: AtomicFact) -> str:
        try:
            return self._active_by_fact[fact]
        except KeyError as exc:
            raise ValueError(f"Fact is not active: {fact}") from exc


class LocalSentenceEmbedder:
    """Minimal Sentence-Transformers-compatible mean-pooling encoder."""

    def __init__(self, model_id: str, *, device: str, batch_size: int) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.model_id = model_id
        self.batch_size = batch_size
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        self.revision = getattr(self.model.config, "_commit_hash", None)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        chunks = []
        torch = self.torch
        for start in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(
                list(texts[start : start + self.batch_size]),
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            batch = {key: value.to(self.device) for key, value in batch.items()}
            with torch.inference_mode():
                hidden = self.model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            chunks.append(pooled.float().cpu().numpy())
        return np.concatenate(chunks, axis=0)


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = path.open("w", encoding="utf-8")
        self.digest = hashlib.sha256()
        self.count = 0

    def write(self, value: Mapping[str, Any]) -> None:
        payload = (
            json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        self.handle.write(payload.decode("utf-8"))
        self.digest.update(payload)
        self.count += 1

    def close(self) -> dict[str, Any]:
        self.handle.close()
        return {
            "path": self.path.name,
            "rows": self.count,
            "bytes": self.path.stat().st_size,
            "sha256": self.digest.hexdigest(),
        }


def parse_grouped_memory(memory: str) -> tuple[AtomicFact, ...]:
    subject: str | None = None
    facts = []
    for raw in normalize_memory(memory).splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("### "):
            subject = line[4:].strip()
            if not subject:
                raise ValueError("Grouped memory has an empty subject header")
            continue
        if line.startswith("- ") and subject is not None:
            fact = AtomicFact(subject=subject, text=line[2:].strip())
            if not fact.text:
                raise ValueError("Grouped memory has an empty fact")
            facts.append(fact)
            continue
        raise ValueError(f"Malformed grouped-memory line: {raw!r}")
    if len(facts) != len(set(facts)):
        raise ValueError("Grouped memory contains duplicate atomic facts")
    return tuple(facts)


def experiment_split(scenario_index: int) -> str:
    """Mirror the current multitask paper split, including temporal/V1 rows."""
    if 15 <= scenario_index <= 80 or 101 <= scenario_index <= 110:
        return "train"
    if scenario_index >= 200:
        return "train"
    if 81 <= scenario_index <= 85 or scenario_index == 111:
        return "validation"
    if 86 <= scenario_index <= 100 or 112 <= scenario_index <= 120:
        return "test"
    return "excluded"


def derive_scenario(rows: Sequence[dict[str, Any]]) -> tuple[
    list[DerivedTurn], tuple[FactVersion, ...]
]:
    if not rows:
        return [], ()
    scenario = int(rows[0]["scenario_index"])
    state = ScenarioFactState(scenario)
    derived = []
    previous_turn = -1
    for row in rows:
        if int(row["scenario_index"]) != scenario:
            raise ValueError("derive_scenario received mixed scenario ids")
        turn = int(row["global_turn_index"])
        if turn != previous_turn + 1:
            raise ValueError(f"Non-contiguous turn index at {row['sample_id']}")
        previous_turn = turn
        before = parse_grouped_memory(row["input"]["previous_memory"])
        if set(before) != set(state.active_facts()):
            raise ValueError(f"Fact replay mismatch before {row['sample_id']}")

        active_before = state.active_version_ids
        working = row["input"]["previous_memory"]
        events: list[dict[str, Any]] = []
        metadata: list[dict[str, Any]] = []
        extracted: list[dict[str, Any]] = []
        operations = row["target"].get("operations", [])
        for operation in operations:
            before_op = parse_grouped_memory(working)
            working, _ = apply_operations(working, [operation])
            after_op = parse_grouped_memory(working)
            removed = [fact for fact in before_op if fact not in set(after_op)]
            added = [fact for fact in after_op if fact not in set(before_op)]
            op = str(operation["op"])
            if op == "add" and removed:
                raise ValueError(f"ADD removed facts at {row['sample_id']}")
            if op == "delete" and added:
                raise ValueError(f"DELETE added facts at {row['sample_id']}")

            paired = min(len(removed), len(added)) if op == "replace" else 0
            for old, new in zip(removed[:paired], added[:paired], strict=True):
                event, detail = state.update(old, new, sample_id=row["sample_id"])
                events.append(event)
                metadata.append(detail)
                extracted.append(_extracted_fact(new))
            for old in removed[paired:]:
                event, detail = state.delete(old)
                events.append(event)
                metadata.append(detail)
            for new in added[paired:]:
                event, detail = state.add(new, sample_id=row["sample_id"])
                events.append(event)
                metadata.append(detail)
                extracted.append(_extracted_fact(new))

        decision = str(row["target"]["decision"])
        if (decision == "NO_OP") != (not events):
            raise ValueError(f"Decision/event mismatch at {row['sample_id']}")
        after = parse_grouped_memory(working)
        if set(after) != set(state.active_facts()):
            raise ValueError(f"Fact replay mismatch after {row['sample_id']}")

        pure_delete = bool(events) and not extracted
        if pure_delete:
            removed_subjects = [
                state.version(item["old_fact_version_id"]).subject
                for item in metadata
                if item["event"] == "DELETE"
            ]
            subject = removed_subjects[0] if removed_subjects else row["current_turn"]["speaker_name"]
            extracted = [
                {
                    "subject": subject,
                    "text": str(row["current_turn"]["text"]).strip(),
                    "ephemeral_invalidation": True,
                }
            ]
        derived.append(
            DerivedTurn(
                row=row,
                extracted_facts=extracted,
                events=events,
                event_metadata=metadata,
                active_before=active_before,
                active_after=state.active_version_ids,
                pure_delete_fallback=pure_delete,
            )
        )
    return derived, state.versions


def _extracted_fact(fact: AtomicFact) -> dict[str, Any]:
    return {"subject": fact.subject, "text": fact.text}


def _retrieval_query(item: DerivedTurn, *, manager: bool) -> str:
    if manager and item.extracted_facts:
        return "\n".join(
            f"{fact['subject']}: {fact['text']}" for fact in item.extracted_facts
        )
    turn = item.row["current_turn"]
    return f"{turn['speaker_name']}: {turn['text']}"


def _rank_active(
    query_embedding: np.ndarray,
    *,
    active_ids: Sequence[str],
    embedding_by_id: Mapping[str, np.ndarray],
    versions_by_id: Mapping[str, FactVersion],
    k: int,
) -> list[FactVersion]:
    scored = [
        (float(np.dot(query_embedding, embedding_by_id[fact_id])), fact_id)
        for fact_id in active_ids
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [versions_by_id[fact_id] for _, fact_id in scored[:k]]


def _retrieved_payload(versions: Iterable[FactVersion]) -> list[dict[str, Any]]:
    return [
        {
            "id": version.record_id,
            "fact_version_id": version.fact_version_id,
            "subject": version.subject,
            "text": version.text,
        }
        for version in versions
    ]


def _target_record_ids(item: DerivedTurn) -> set[str]:
    return {
        str(event["id"])
        for event in item.events
        if event["event"] in {"UPDATE", "DELETE"}
    }


def _target_is_retrieved(item: DerivedTurn, retrieved: Sequence[FactVersion]) -> bool:
    targets = _target_record_ids(item)
    return targets <= {version.record_id for version in retrieved}


def _base_row(item: DerivedTurn) -> dict[str, Any]:
    row = item.row
    split = experiment_split(int(row["scenario_index"]))
    return {
        "sample_id": row["sample_id"],
        "scenario_index": row["scenario_index"],
        "global_turn_index": row["global_turn_index"],
        "timestamp": row.get("timestamp"),
        "source_split": row.get("split"),
        "split": split,
        "train_eligible": bool(row.get("train_eligible", True))
        and split != "excluded",
    }


def _write_readme(path: Path, *, source: Path, retrieval_k: int) -> None:
    path.write_text(
        f"""# Mem0-style one-pass and two-stage data

Source: `{source}`

This package contains aligned data for two learned fact-memory baselines.

* `fact_extraction.jsonl`: current Turn -> extracted facts. Empty facts are NO_OP.
* `fact_manager.jsonl`: extracted facts + vector top-{retrieval_k} active records ->
  ADD/UPDATE/DELETE/NONE. It includes deterministic exact-duplicate NONE rows.
* `joint_fact_update.jsonl`: current Turn + vector top-{retrieval_k} records ->
  extracted facts and CRUD events in one response.
* `facts.jsonl`: immutable fact versions and stable record ids.

The experiment split is Train S21-S80 + T1-T10,
Validation S81-S85 + T11, and Test S86-S100 + T12-T20. S1-S20 are retained as
`excluded`. Filter training rows by `split=train` and `train_eligible=true`.
Validation/Test rows remain chronological for closed-loop evaluation.

Retrieval candidates are generated without learned task-specific retrieval:
fixed normalized embeddings, cosine similarity, and top-{retrieval_k}. The manager
uses the gold extracted fact as its training retrieval query; the joint view uses
the raw current Turn. Test-time retrieval must be recomputed from predicted active
facts rather than reusing these teacher-forced candidates.
""",
        encoding="utf-8",
    )


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    if output.exists():
        if not args.force:
            raise FileExistsError(output)
        if output == output.parent or output == Path("/"):
            raise ValueError("Refusing unsafe output replacement")
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    embedder = LocalSentenceEmbedder(
        args.embedding_model,
        device=args.embedding_device,
        batch_size=args.embedding_batch_size,
    )
    writers = {
        "facts": JsonlWriter(temporary / "facts.jsonl"),
        "extraction": JsonlWriter(temporary / "fact_extraction.jsonl"),
        "manager": JsonlWriter(temporary / "fact_manager.jsonl"),
        "joint": JsonlWriter(temporary / "joint_fact_update.jsonl"),
    }
    counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    retrieval_counts: Counter[str] = Counter()

    def process(rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        derived, versions = derive_scenario(rows)
        versions_by_id = {item.fact_version_id: item for item in versions}
        texts = [item.fact.embedding_text for item in versions]
        fact_matrix = embedder.encode(texts)
        embedding_by_id = {
            version.fact_version_id: fact_matrix[index]
            for index, version in enumerate(versions)
        }
        joint_queries = [_retrieval_query(item, manager=False) for item in derived]
        manager_queries = [_retrieval_query(item, manager=True) for item in derived]
        joint_matrix = embedder.encode(joint_queries)
        manager_matrix = embedder.encode(manager_queries)

        for version in versions:
            writers["facts"].write(
                {"schema_version": FACT_SCHEMA_VERSION, **asdict(version)}
            )
        for index, item in enumerate(derived):
            base = _base_row(item)
            split_counts[base["split"]] += 1
            counts["source_rows"] += 1
            counts["source_updates" if item.events else "source_no_ops"] += 1
            if item.pure_delete_fallback:
                counts["pure_delete_fallbacks"] += 1

            retrieved_joint = _rank_active(
                joint_matrix[index],
                active_ids=item.active_before,
                embedding_by_id=embedding_by_id,
                versions_by_id=versions_by_id,
                k=args.retrieval_k,
            )
            retrieved_manager = _rank_active(
                manager_matrix[index],
                active_ids=item.active_before,
                embedding_by_id=embedding_by_id,
                versions_by_id=versions_by_id,
                k=args.retrieval_k,
            )
            joint_recall = _target_is_retrieved(item, retrieved_joint)
            manager_recall = _target_is_retrieved(item, retrieved_manager)
            if _target_record_ids(item):
                retrieval_counts["target_rows"] += 1
                retrieval_counts[f"joint_target_recalled_{joint_recall}"] += 1
                retrieval_counts[f"manager_target_recalled_{manager_recall}"] += 1
                if base["train_eligible"]:
                    retrieval_counts["eligible_target_rows"] += 1
                    retrieval_counts[
                        f"eligible_joint_target_recalled_{joint_recall}"
                    ] += 1
                    retrieval_counts[
                        f"eligible_manager_target_recalled_{manager_recall}"
                    ] += 1

            extraction_row = {
                "schema_version": EXTRACTION_SCHEMA_VERSION,
                **base,
                "input": {"source_batch": [item.row["current_turn"]]},
                "target": {"facts": item.extracted_facts},
                "provenance": {
                    "source_schema_version": item.row.get("schema_version"),
                    "pure_delete_fallback": item.pure_delete_fallback,
                },
            }
            writers["extraction"].write(extraction_row)

            target_present = joint_recall or not _target_record_ids(item)
            joint_base = dict(base)
            joint_base["train_eligible"] = bool(base["train_eligible"]) and target_present
            writers["joint"].write(
                {
                    "schema_version": JOINT_SCHEMA_VERSION,
                    **joint_base,
                    "input": {
                        "source_batch": [item.row["current_turn"]],
                        "retrieved_facts": _retrieved_payload(retrieved_joint),
                    },
                    "target": {
                        "facts": item.extracted_facts,
                        "memory": item.events,
                    },
                    "gold_state": {
                        "active_before": list(item.active_before),
                        "active_after": list(item.active_after),
                    },
                    "provenance": {
                        "source_schema_version": item.row.get("schema_version"),
                        "retrieval_target_present": joint_recall,
                        "pure_delete_fallback": item.pure_delete_fallback,
                        "event_metadata": item.event_metadata,
                    },
                }
            )

            if item.events:
                manager_base = dict(base)
                manager_base["train_eligible"] = (
                    bool(base["train_eligible"]) and manager_recall
                )
                writers["manager"].write(
                    {
                        "schema_version": MANAGER_SCHEMA_VERSION,
                        **manager_base,
                        "input": {
                            "new_facts": item.extracted_facts,
                            "retrieved_facts": _retrieved_payload(retrieved_manager),
                        },
                        "target": {"memory": item.events},
                        "provenance": {
                            "source_sample_id": item.row["sample_id"],
                            "synthetic_none": False,
                            "retrieval_target_present": manager_recall,
                            "pure_delete_fallback": item.pure_delete_fallback,
                            "event_metadata": item.event_metadata,
                        },
                    }
                )
                counts["manager_update_rows"] += 1

                # The Patch source suppresses duplicates before the manager.
                # Add exact-duplicate cases so NONE is a supervised manager action.
                newly_active = [
                    detail["new_fact_version_id"]
                    for detail in item.event_metadata
                    if detail["event"] in {"ADD", "UPDATE"}
                    and detail.get("new_fact_version_id") in item.active_after
                ]
                for duplicate_index, fact_version_id in enumerate(newly_active):
                    fact = versions_by_id[fact_version_id]
                    query_embedding = embedding_by_id[fact_version_id]
                    retrieved_none = _rank_active(
                        query_embedding,
                        active_ids=item.active_after,
                        embedding_by_id=embedding_by_id,
                        versions_by_id=versions_by_id,
                        k=args.retrieval_k,
                    )
                    present = fact.record_id in {
                        value.record_id for value in retrieved_none
                    }
                    none_base = dict(base)
                    none_base["sample_id"] = (
                        f"{base['sample_id']}:manager-none:{duplicate_index:02d}"
                    )
                    none_base["train_eligible"] = (
                        bool(base["train_eligible"]) and present
                    )
                    writers["manager"].write(
                        {
                            "schema_version": MANAGER_SCHEMA_VERSION,
                            **none_base,
                            "input": {
                                "new_facts": [_extracted_fact(fact.fact)],
                                "retrieved_facts": _retrieved_payload(retrieved_none),
                            },
                            "target": {
                                "memory": [
                                    {"event": "NONE", "id": fact.record_id}
                                ]
                            },
                            "provenance": {
                                "source_sample_id": item.row["sample_id"],
                                "synthetic_none": True,
                                "retrieval_target_present": present,
                                "pure_delete_fallback": False,
                                "event_metadata": [],
                            },
                        }
                    )
                    counts["manager_none_rows"] += 1

    current_scenario: int | None = None
    rows: list[dict[str, Any]] = []
    try:
        with (source / "patch.jsonl").open(encoding="utf-8") as handle:
            for raw in handle:
                row = json.loads(raw)
                scenario = int(row["scenario_index"])
                if current_scenario is None:
                    current_scenario = scenario
                if scenario != current_scenario:
                    process(rows)
                    rows = []
                    current_scenario = scenario
                rows.append(row)
        process(rows)
        file_metadata = {name: writer.close() for name, writer in writers.items()}
        source_manifest = json.loads(
            (source / "manifest.json").read_text(encoding="utf-8")
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": str(source),
            "source_schema_version": source_manifest.get("schema_version"),
            "source_patch_sha256": source_manifest.get("files", {})
            .get("patch", {})
            .get("sha256"),
            "split_policy": {
                "train": "S21-S80 + T1-T10",
                "validation": "S81-S85 + T11",
                "test": "S86-S100 + T12-T20",
                "excluded": "S1-S20",
            },
            "retrieval": {
                "model": args.embedding_model,
                "revision": embedder.revision,
                "pooling": "attention-mask mean pooling + L2 normalization",
                "similarity": "cosine",
                "top_k": args.retrieval_k,
                "oracle_target_injection": False,
            },
            "files": file_metadata,
            "counts": dict(sorted(counts.items())),
            "split_counts": dict(sorted(split_counts.items())),
            "retrieval_counts": dict(sorted(retrieval_counts.items())),
            "quiz_sft": {
                "path": str(source / "quiz_sft.jsonl"),
                "vehicle_tools_path": str(source / "vehicle_tools.json"),
            },
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_readme(
            temporary / "README.md", source=source, retrieval_k=args.retrieval_k
        )
        os.replace(temporary, output)
        return manifest
    except BaseException:
        for writer in writers.values():
            if not writer.handle.closed:
                writer.handle.close()
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--retrieval-k", type=int, default=20)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-device", default="auto")
    parser.add_argument("--embedding-batch-size", type=int, default=512)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.retrieval_k < 1:
        raise ValueError("retrieval-k must be positive")
    if args.embedding_batch_size < 1:
        raise ValueError("embedding-batch-size must be positive")
    print(json.dumps(prepare(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
