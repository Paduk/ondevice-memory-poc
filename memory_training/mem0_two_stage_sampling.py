"""Task-balanced epoch sampling for the shared-model Mem0 two-stage view."""

from __future__ import annotations

import random
from typing import Any

from .sampling import EpochPlan, EpochSampler, SamplingConfig, TrainingUnit


class _ExtractionOnlyCatalog:
    def __init__(self, catalog: Any, boundary: int) -> None:
        self.catalog = catalog
        self.boundary = boundary

    def __getattr__(self, name: str) -> Any:
        return getattr(self.catalog, name)

    def eligible_row_ids(self, **kwargs: Any) -> list[int]:
        return [
            row_id
            for row_id in self.catalog.eligible_row_ids(**kwargs)
            if row_id < self.boundary
        ]

    def adjacent_row_ids(self, **kwargs: Any) -> dict[int, list[int]]:
        return {
            distance: [row_id for row_id in rows if row_id < self.boundary]
            for distance, rows in self.catalog.adjacent_row_ids(**kwargs).items()
        }

    def scenario_eligible_segments(self, scenario_index: int) -> list[list[int]]:
        return [
            selected
            for segment in self.catalog.scenario_eligible_segments(scenario_index)
            if (selected := [row_id for row_id in segment if row_id < self.boundary])
        ]


class Mem0TwoStageEpochSampler:
    """Sample EXTRACT as 1:N while retaining every MANAGE CRUD/NONE row."""

    def __init__(
        self, catalog: Any, config: SamplingConfig, *, extraction_rows: int
    ) -> None:
        self.catalog = catalog
        self.config = config
        self.extraction_rows = extraction_rows
        self.extraction_sampler = EpochSampler(
            _ExtractionOnlyCatalog(catalog, extraction_rows), config
        )
        allowed = set(config.train_scenarios or range(config.train_scenario_min, config.train_scenario_max + 1))
        with catalog._connect() as connection:
            rows = connection.execute(
                "SELECT row_id, scenario_index, decision FROM samples "
                "WHERE split = 'train' AND train_eligible = 1 AND row_id >= ? "
                "ORDER BY row_id",
                (extraction_rows,),
            )
            selected = [row for row in rows if int(row["scenario_index"]) in allowed]
        self.manager_ids = tuple(int(row["row_id"]) for row in selected)
        self.manager_update_count = sum(row["decision"] == "UPDATE" for row in selected)
        self.manager_noop_count = len(selected) - self.manager_update_count

    def build(self, epoch: int) -> EpochPlan:
        extraction = self.extraction_sampler.build(epoch)
        generator = random.Random(extraction.seed + 7919)
        manager_ids = list(self.manager_ids)
        generator.shuffle(manager_ids)
        manager_units = [TrainingUnit("independent", (row_id,)) for row_id in manager_ids]
        units = list(extraction.units) + manager_units
        generator.shuffle(units)
        independent = list(extraction.independent_row_ids) + manager_ids
        return EpochPlan(
            epoch=extraction.epoch,
            seed=extraction.seed,
            independent_row_ids=tuple(independent),
            trajectory_windows=extraction.trajectory_windows,
            units=tuple(units),
            update_count=extraction.update_count + self.manager_update_count,
            sampled_noop_count=extraction.sampled_noop_count + self.manager_noop_count,
            sampled_adjacent_noop_count=extraction.sampled_adjacent_noop_count,
            sampled_random_noop_count=extraction.sampled_random_noop_count + self.manager_noop_count,
            adjacent_noop_by_distance=extraction.adjacent_noop_by_distance,
        )
