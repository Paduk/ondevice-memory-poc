from __future__ import annotations

import json
from pathlib import Path

from memory_training.dataset import (
    DatasetCatalog,
    IndexedMemoryDataset,
    build_catalog,
)
from memory_training.sampling import EpochSampler, SamplingConfig
from memory_training.validation import stratified_teacher_forced_row_ids


def _write_views(root: Path) -> None:
    handles = {
        view: (root / f"{view}.jsonl").open("w", encoding="utf-8")
        for view in ("summary", "patch", "delta")
    }
    try:
        for scenario in (1, 81, 91):
            split = (
                "train" if scenario == 1 else "validation" if scenario == 81 else "test"
            )
            for turn in range(12):
                decision = "UPDATE" if turn in {2, 7} else "NO_OP"
                for view, handle in handles.items():
                    row = {
                        "sample_id": f"s{scenario:03d}:{view}:{turn:05d}",
                        "scenario_index": scenario,
                        "split": split,
                        "global_turn_index": turn,
                        "turn_id": f"s{scenario}-turn-{turn}",
                        "current_turn": {"text": f"turn {turn}"},
                        "input": {},
                        "target": {
                            "decision": decision,
                            **(
                                {"next_memory": "memory"}
                                if view == "summary"
                                else {"operations": []}
                            ),
                        },
                    }
                    handle.write(json.dumps(row) + "\n")
    finally:
        for handle in handles.values():
            handle.close()
    (root / "manifest.json").write_text("{}", encoding="utf-8")
    (root / "quiz_manifest.json").write_text("{}", encoding="utf-8")


def test_catalog_lazy_read_and_split(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_views(data_root)
    catalog_path = tmp_path / "catalog.sqlite"

    result = build_catalog(data_root, catalog_path)
    catalog = DatasetCatalog(catalog_path, data_root)
    dataset = IndexedMemoryDataset(catalog, "patch", split="validation")

    assert result["row_count"] == 36
    assert catalog.count(split="train", decision="UPDATE") == 2
    assert len(dataset) == 12
    assert dataset[0]["scenario_index"] == 81
    assert dataset[-1]["global_turn_index"] == 11


def test_epoch_sampler_preserves_updates_and_is_deterministic(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_views(data_root)
    catalog_path = tmp_path / "catalog.sqlite"
    build_catalog(data_root, catalog_path)
    catalog = DatasetCatalog(catalog_path, data_root)
    config = SamplingConfig(
        noop_per_update=2,
        adjacent_noop_fraction=0.5,
        adjacent_max_distance=1,
        trajectory_fraction=0.25,
        trajectory_min_turns=2,
        trajectory_max_turns=4,
        seed=7,
    )

    first = EpochSampler(catalog, config).build(0)
    repeated = EpochSampler(catalog, config).build(0)
    next_epoch = EpochSampler(catalog, config).build(1)
    update_ids = set(catalog.row_ids(split="train", decision="UPDATE"))

    assert first == repeated
    assert update_ids.issubset(first.independent_row_ids)
    assert first.sampled_noop_count == 4
    assert first.sampled_adjacent_noop_count == 2
    assert first.sampled_random_noop_count == 2
    assert first.trajectory_windows
    assert first != next_epoch
    for window in first.trajectory_windows:
        assert 2 <= len(window.row_ids) <= 4
        assert (
            len({catalog.record(row_id).scenario_index for row_id in window.row_ids})
            == 1
        )


def test_teacher_forced_validation_subset_is_fixed_and_update_preserving(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_views(data_root)
    catalog_path = tmp_path / "catalog.sqlite"
    build_catalog(data_root, catalog_path)
    catalog = DatasetCatalog(catalog_path, data_root)

    first = stratified_teacher_forced_row_ids(catalog, max_rows=8, seed=9)
    repeated = stratified_teacher_forced_row_ids(catalog, max_rows=8, seed=9)
    updates = set(catalog.row_ids(split="validation", decision="UPDATE"))

    assert first == repeated
    assert len(first) == 8
    assert updates.issubset(first)
    assert all(catalog.record(row_id).split == "validation" for row_id in first)


def test_teacher_forced_validation_subset_rejects_dropped_updates(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_views(data_root)
    catalog_path = tmp_path / "catalog.sqlite"
    build_catalog(data_root, catalog_path)
    catalog = DatasetCatalog(catalog_path, data_root)

    try:
        stratified_teacher_forced_row_ids(catalog, max_rows=1)
    except ValueError as exc:
        assert "cannot retain all" in str(exc)
    else:
        raise AssertionError("Expected UPDATE-preserving size validation")
