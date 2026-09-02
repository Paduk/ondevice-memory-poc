"""Indexed access to aligned Summary, Patch, and Delta JSONL training views."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Literal

from .config import DEFAULT_DATA_ROOT, DEFAULT_WORKSPACE_ROOT, split_for_scenario

CATALOG_SCHEMA_VERSION = "palmclaw-memory-dataset-catalog-v1"
MEMORY_VIEWS = ("summary", "patch", "delta")
MemoryView = Literal["summary", "patch", "delta"]


@dataclass(frozen=True)
class SampleRecord:
    row_id: int
    scenario_index: int
    split: str
    global_turn_index: int
    turn_id: str
    decision: str


def default_catalog_path(workspace: Path = DEFAULT_WORKSPACE_ROOT) -> Path:
    return workspace / "cache" / "dataset" / "catalog.sqlite"


def _source_fingerprint(data_root: Path) -> str:
    digest = hashlib.sha256()
    for name in ("manifest.json", "quiz_manifest.json"):
        path = data_root / name
        digest.update(name.encode())
        digest.update(path.read_bytes())
    for view in MEMORY_VIEWS:
        path = data_root / f"{view}.jsonl"
        stat = path.stat()
        digest.update(f"{view}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def _decode_row(raw: bytes, *, view: str, line_number: int) -> dict[str, Any]:
    try:
        row = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {view}.jsonl:{line_number}") from exc
    if not isinstance(row, dict):
        raise TypeError(f"Expected object in {view}.jsonl:{line_number}")
    return row


def _identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("scenario_index"),
        row.get("global_turn_index"),
        row.get("turn_id"),
        row.get("split"),
        row.get("target", {}).get("decision"),
        row.get("train_eligible", True),
    )


def build_catalog(data_root: Path, catalog_path: Path) -> dict[str, Any]:
    """Build an atomic byte-offset catalog without copying source JSONL files."""
    data_root = data_root.resolve()
    catalog_path = catalog_path.resolve()
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = catalog_path.with_suffix(".sqlite.tmp")
    temporary.unlink(missing_ok=True)

    connection = sqlite3.connect(temporary)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode = OFF;
            PRAGMA synchronous = OFF;
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE samples (
                row_id INTEGER PRIMARY KEY,
                scenario_index INTEGER NOT NULL,
                split TEXT NOT NULL,
                global_turn_index INTEGER NOT NULL,
                turn_id TEXT NOT NULL,
                decision TEXT NOT NULL,
                train_eligible INTEGER NOT NULL,
                summary_offset INTEGER NOT NULL,
                summary_length INTEGER NOT NULL,
                patch_offset INTEGER NOT NULL,
                patch_length INTEGER NOT NULL,
                delta_offset INTEGER NOT NULL,
                delta_length INTEGER NOT NULL,
                UNIQUE(scenario_index, global_turn_index)
            );
            CREATE INDEX samples_split_decision
                ON samples(split, decision, row_id);
            CREATE INDEX samples_scenario_turn
                ON samples(scenario_index, global_turn_index);
            """
        )
        paths = [data_root / f"{view}.jsonl" for view in MEMORY_VIEWS]
        handles = [path.open("rb") for path in paths]
        try:
            batch = []
            row_id = 0
            split_counts: dict[str, int] = {}
            decision_counts: dict[str, int] = {}
            while True:
                offsets = [handle.tell() for handle in handles]
                lines = [handle.readline() for handle in handles]
                if not any(lines):
                    break
                if not all(lines):
                    raise ValueError("Summary/Patch/Delta line counts differ")
                rows = [
                    _decode_row(line, view=view, line_number=row_id + 1)
                    for view, line in zip(MEMORY_VIEWS, lines, strict=True)
                ]
                identity = _identity(rows[0])
                if any(_identity(row) != identity for row in rows[1:]):
                    raise ValueError(f"Cross-view mismatch at row {row_id}")
                scenario, global_turn, turn_id, split, decision, train_eligible = identity
                if not isinstance(scenario, int) or split != split_for_scenario(
                    scenario
                ):
                    raise ValueError(f"Invalid scenario split at row {row_id}")
                if not isinstance(global_turn, int) or not isinstance(turn_id, str):
                    raise TypeError(f"Invalid turn identity at row {row_id}")
                if decision not in {"NO_OP", "UPDATE"}:
                    raise ValueError(f"Invalid decision at row {row_id}: {decision}")
                if not isinstance(train_eligible, bool):
                    raise TypeError(f"Invalid train_eligible at row {row_id}")
                batch.append(
                    (
                        row_id,
                        scenario,
                        split,
                        global_turn,
                        turn_id,
                        decision,
                        int(train_eligible),
                        offsets[0],
                        len(lines[0]),
                        offsets[1],
                        len(lines[1]),
                        offsets[2],
                        len(lines[2]),
                    )
                )
                split_counts[split] = split_counts.get(split, 0) + 1
                decision_counts[decision] = decision_counts.get(decision, 0) + 1
                row_id += 1
                if len(batch) >= 4096:
                    connection.executemany(
                        "INSERT INTO samples VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        batch,
                    )
                    batch.clear()
            if batch:
                connection.executemany(
                    "INSERT INTO samples VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    batch,
                )
        finally:
            for handle in handles:
                handle.close()

        metadata = {
            "schema_version": CATALOG_SCHEMA_VERSION,
            "data_root": str(data_root),
            "source_fingerprint": _source_fingerprint(data_root),
            "row_count": str(row_id),
            "split_counts": json.dumps(split_counts, sort_keys=True),
            "decision_counts": json.dumps(decision_counts, sort_keys=True),
        }
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)", metadata.items()
        )
        connection.commit()
    except BaseException:
        connection.close()
        temporary.unlink(missing_ok=True)
        raise
    else:
        connection.close()
    os.replace(temporary, catalog_path)
    return {
        "catalog_path": str(catalog_path),
        "row_count": row_id,
        "split_counts": split_counts,
        "decision_counts": decision_counts,
        "source_fingerprint": metadata["source_fingerprint"],
    }


class DatasetCatalog:
    """Read-only metadata queries over the aligned memory views."""

    def __init__(self, path: Path, data_root: Path = DEFAULT_DATA_ROOT) -> None:
        self.path = path.resolve()
        self.data_root = data_root.resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self._validate_source()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        return connection

    def metadata(self) -> dict[str, str]:
        with self._connect() as connection:
            return {
                row["key"]: row["value"]
                for row in connection.execute("SELECT key, value FROM metadata")
            }

    def _validate_source(self) -> None:
        metadata = self.metadata()
        if metadata.get("schema_version") != CATALOG_SCHEMA_VERSION:
            raise ValueError("Unsupported or stale dataset catalog schema")
        if Path(metadata.get("data_root", "")).resolve() != self.data_root:
            raise ValueError("Dataset catalog points to a different data root")
        if metadata.get("source_fingerprint") != _source_fingerprint(self.data_root):
            raise ValueError("Dataset source changed; rebuild the offset catalog")

    def count(self, *, split: str | None = None, decision: str | None = None) -> int:
        where, parameters = _where_clause(split=split, decision=decision)
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT COUNT(*) AS count FROM samples{where}", parameters
            ).fetchone()
        return int(row["count"])

    def row_ids(
        self, *, split: str | None = None, decision: str | None = None
    ) -> list[int]:
        where, parameters = _where_clause(split=split, decision=decision)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT row_id FROM samples{where} ORDER BY row_id", parameters
            )
            return [int(row["row_id"]) for row in rows]

    def eligible_row_ids(
        self, *, split: str | None = None, decision: str | None = None
    ) -> list[int]:
        """Return train-eligible rows; legacy catalogs default every row to eligible."""
        if not self._has_sample_column("train_eligible"):
            return self.row_ids(split=split, decision=decision)
        where, parameters = _where_clause(split=split, decision=decision)
        clause = f"{where} AND train_eligible = 1" if where else " WHERE train_eligible = 1"
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT row_id FROM samples{clause} ORDER BY row_id", parameters
            )
            return [int(row["row_id"]) for row in rows]

    def scenario_eligible_segments(self, scenario_index: int) -> list[list[int]]:
        """Split one scenario at ineligible rows and missing turn positions."""
        if not self._has_sample_column("train_eligible"):
            rows = self.scenario_row_ids(scenario_index)
            return [rows] if rows else []
        with self._connect() as connection:
            rows = list(
                connection.execute(
                    "SELECT row_id, global_turn_index, train_eligible FROM samples "
                    "WHERE scenario_index = ? ORDER BY global_turn_index",
                    (scenario_index,),
                )
            )
        segments: list[list[int]] = []
        current: list[int] = []
        previous_turn: int | None = None
        for row in rows:
            turn = int(row["global_turn_index"])
            eligible = bool(row["train_eligible"])
            if not eligible or (
                previous_turn is not None and turn != previous_turn + 1
            ):
                if current:
                    segments.append(current)
                current = []
            if eligible:
                current.append(int(row["row_id"]))
                previous_turn = turn
            else:
                previous_turn = None
        if current:
            segments.append(current)
        return segments

    def _has_sample_column(self, name: str) -> bool:
        with self._connect() as connection:
            return any(
                str(row["name"]) == name
                for row in connection.execute("PRAGMA table_info(samples)")
            )

    def scenarios(self, *, split: str | None = None) -> list[int]:
        where, parameters = _where_clause(split=split)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT DISTINCT scenario_index FROM samples{where} "
                "ORDER BY scenario_index",
                parameters,
            )
            return [int(row["scenario_index"]) for row in rows]

    def scenario_row_ids(self, scenario_index: int) -> list[int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT row_id FROM samples WHERE scenario_index = ? "
                "ORDER BY global_turn_index",
                (scenario_index,),
            )
            return [int(row["row_id"]) for row in rows]

    def adjacent_row_ids(
        self,
        *,
        split: str,
        anchor_decision: str = "UPDATE",
        target_decision: str = "NO_OP",
        max_distance: int = 2,
        eligible_only: bool = False,
    ) -> dict[int, list[int]]:
        """Return target rows grouped by nearest distance from an anchor row."""
        if max_distance < 1:
            raise ValueError("max_distance must be positive")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT row_id, scenario_index, global_turn_index, decision "
                "FROM samples WHERE split = ? ORDER BY scenario_index, global_turn_index",
                (split,),
            )
            materialized = list(rows)
        if eligible_only and self._has_sample_column("train_eligible"):
            eligible = set(self.eligible_row_ids(split=split))
            materialized = [row for row in materialized if int(row["row_id"]) in eligible]
        anchors = {
            (int(row["scenario_index"]), int(row["global_turn_index"]))
            for row in materialized
            if row["decision"] == anchor_decision
        }
        grouped = {distance: [] for distance in range(1, max_distance + 1)}
        for row in materialized:
            if row["decision"] != target_decision:
                continue
            scenario = int(row["scenario_index"])
            turn = int(row["global_turn_index"])
            nearest = next(
                (
                    distance
                    for distance in range(1, max_distance + 1)
                    if (scenario, turn - distance) in anchors
                    or (scenario, turn + distance) in anchors
                ),
                None,
            )
            if nearest is not None:
                grouped[nearest].append(int(row["row_id"]))
        return grouped

    def record(self, row_id: int) -> SampleRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT row_id, scenario_index, split, global_turn_index, "
                "turn_id, decision FROM samples WHERE row_id = ?",
                (row_id,),
            ).fetchone()
        if row is None:
            raise IndexError(row_id)
        return SampleRecord(**dict(row))

    def row_id_for_turn(self, scenario_index: int, global_turn_index: int) -> int:
        """Resolve one canonical trajectory position to its aligned row id."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT row_id FROM samples WHERE scenario_index = ? "
                "AND global_turn_index = ?",
                (scenario_index, global_turn_index),
            ).fetchone()
        if row is None:
            raise KeyError((scenario_index, global_turn_index))
        return int(row["row_id"])

    def location(self, row_id: int, view: MemoryView) -> tuple[int, int]:
        if view not in MEMORY_VIEWS:
            raise ValueError(f"Unknown memory view: {view}")
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT {view}_offset AS offset, {view}_length AS length "
                "FROM samples WHERE row_id = ?",
                (row_id,),
            ).fetchone()
        if row is None:
            raise IndexError(row_id)
        return int(row["offset"]), int(row["length"])

    def locations(
        self,
        view: MemoryView,
        *,
        row_ids: Sequence[int] | None = None,
        split: str | None = None,
    ) -> list[tuple[int, int, int, int]]:
        """Return row_id, offset, length, scenario in source order."""
        if view not in MEMORY_VIEWS:
            raise ValueError(f"Unknown memory view: {view}")
        if row_ids is not None and split is not None:
            raise ValueError("Provide row_ids or split, not both")
        where, parameters = _where_clause(split=split)
        selected = set(row_ids) if row_ids is not None else None
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT row_id, {view}_offset AS offset, {view}_length AS length, "
                f"scenario_index FROM samples{where} ORDER BY row_id",
                parameters,
            )
            locations = [
                (
                    int(row["row_id"]),
                    int(row["offset"]),
                    int(row["length"]),
                    int(row["scenario_index"]),
                )
                for row in rows
                if selected is None or int(row["row_id"]) in selected
            ]
        if selected is not None and len(locations) != len(selected):
            found = {location[0] for location in locations}
            raise IndexError(f"Unknown row_ids: {sorted(selected - found)[:10]}")
        return locations


def _where_clause(
    *, split: str | None = None, decision: str | None = None
) -> tuple[str, tuple[str, ...]]:
    clauses = []
    parameters = []
    if split is not None:
        clauses.append("split = ?")
        parameters.append(split)
    if decision is not None:
        clauses.append("decision = ?")
        parameters.append(decision)
    return (" WHERE " + " AND ".join(clauses) if clauses else "", tuple(parameters))


class IndexedMemoryDataset(Sequence[dict[str, Any]]):
    """Worker-safe lazy JSONL reader backed by the byte-offset catalog."""

    def __init__(
        self,
        catalog: DatasetCatalog,
        view: MemoryView,
        *,
        row_ids: Sequence[int] | None = None,
        split: str | None = None,
    ) -> None:
        if view not in MEMORY_VIEWS:
            raise ValueError(f"Unknown memory view: {view}")
        if row_ids is not None and split is not None:
            raise ValueError("Provide row_ids or split, not both")
        self.catalog = catalog
        self.view = view
        requested = list(row_ids) if row_ids is not None else None
        by_row_id = {
            row_id: (offset, length, scenario)
            for row_id, offset, length, scenario in catalog.locations(
                view, row_ids=requested, split=split
            )
        }
        ordered_ids = requested if requested is not None else sorted(by_row_id)
        self._entries = [(row_id, *by_row_id[row_id]) for row_id in ordered_ids]
        self._position_by_row_id = {
            row_id: position for position, (row_id, *_rest) in enumerate(self._entries)
        }
        self._handle: BinaryIO | None = None
        self._handle_pid: int | None = None

    def __len__(self) -> int:
        return len(self._entries)

    def position_for_row_id(self, row_id: int) -> int:
        """Return the dataset position used by a PyTorch batch sampler."""
        try:
            return self._position_by_row_id[row_id]
        except KeyError as exc:
            raise IndexError(row_id) from exc

    def row_id_at_position(self, position: int) -> int:
        return int(self._entries[position][0])

    def contains_row_id(self, row_id: int) -> bool:
        return row_id in self._position_by_row_id

    def _file(self) -> BinaryIO:
        pid = os.getpid()
        if self._handle is None or self._handle_pid != pid:
            self.close()
            self._handle = (self.catalog.data_root / f"{self.view}.jsonl").open("rb")
            self._handle_pid = pid
        return self._handle

    def __getitem__(self, index: int | slice) -> dict[str, Any] | list[dict[str, Any]]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        row_id, offset, length, scenario_index = self._entries[index]
        handle = self._file()
        handle.seek(offset)
        raw = handle.read(length)
        row = _decode_row(raw, view=self.view, line_number=row_id + 1)
        if row.get("scenario_index") != scenario_index:
            raise ValueError(f"Catalog/source mismatch at row {row_id}")
        return row

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
        self._handle = None
        self._handle_pid = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handle"] = None
        state["_handle_pid"] = None
        return state

    def __del__(self) -> None:
        self.close()


def ensure_catalog(data_root: Path, catalog_path: Path) -> DatasetCatalog:
    try:
        return DatasetCatalog(catalog_path, data_root)
    except (FileNotFoundError, ValueError):
        build_catalog(data_root, catalog_path)
        return DatasetCatalog(catalog_path, data_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "inspect"))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE_ROOT)
    args = parser.parse_args()
    catalog_path = default_catalog_path(args.workspace)
    if args.command == "build":
        result = build_catalog(args.data_root, catalog_path)
    else:
        catalog = ensure_catalog(args.data_root, catalog_path)
        result = {
            "catalog_path": str(catalog.path),
            "metadata": catalog.metadata(),
            "counts": {
                split: {
                    decision: catalog.count(split=split, decision=decision)
                    for decision in ("NO_OP", "UPDATE")
                }
                for split in ("train", "validation", "test")
            },
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
