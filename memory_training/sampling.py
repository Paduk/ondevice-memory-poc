"""Deterministic UPDATE-preserving NO_OP and trajectory sampling."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .config import DEFAULT_DATA_ROOT, DEFAULT_WORKSPACE_ROOT
from .dataset import DatasetCatalog, default_catalog_path, ensure_catalog


@dataclass(frozen=True)
class SamplingConfig:
    noop_per_update: int = 10
    adjacent_noop_fraction: float = 0.30
    adjacent_max_distance: int = 2
    trajectory_fraction: float = 0.20
    trajectory_min_turns: int = 32
    trajectory_max_turns: int = 128
    seed: int = 42

    def __post_init__(self) -> None:
        if self.noop_per_update < 0:
            raise ValueError("noop_per_update must be non-negative")
        if not 0 <= self.adjacent_noop_fraction <= 1:
            raise ValueError("adjacent_noop_fraction must be in [0, 1]")
        if self.adjacent_max_distance < 1:
            raise ValueError("adjacent_max_distance must be positive")
        if not 0 <= self.trajectory_fraction < 1:
            raise ValueError("trajectory_fraction must be in [0, 1)")
        if self.trajectory_min_turns < 1:
            raise ValueError("trajectory_min_turns must be positive")
        if self.trajectory_max_turns < self.trajectory_min_turns:
            raise ValueError("trajectory_max_turns must be >= trajectory_min_turns")


@dataclass(frozen=True)
class TrainingUnit:
    kind: Literal["independent", "trajectory"]
    row_ids: tuple[int, ...]
    scenario_index: int | None = None


@dataclass(frozen=True)
class EpochPlan:
    epoch: int
    seed: int
    independent_row_ids: tuple[int, ...]
    trajectory_windows: tuple[TrainingUnit, ...]
    units: tuple[TrainingUnit, ...]
    update_count: int
    sampled_noop_count: int
    sampled_adjacent_noop_count: int
    sampled_random_noop_count: int
    adjacent_noop_by_distance: tuple[tuple[int, int], ...]

    @property
    def trajectory_turn_count(self) -> int:
        return sum(len(window.row_ids) for window in self.trajectory_windows)

    def summary(self) -> dict[str, int | float]:
        total_exposures = len(self.independent_row_ids) + self.trajectory_turn_count
        return {
            "epoch": self.epoch,
            "seed": self.seed,
            "update_count": self.update_count,
            "sampled_noop_count": self.sampled_noop_count,
            "sampled_adjacent_noop_count": self.sampled_adjacent_noop_count,
            "sampled_random_noop_count": self.sampled_random_noop_count,
            "adjacent_noop_fraction": (
                self.sampled_adjacent_noop_count / self.sampled_noop_count
                if self.sampled_noop_count
                else 0.0
            ),
            "adjacent_distance_1_count": dict(self.adjacent_noop_by_distance).get(1, 0),
            "adjacent_distance_2_count": dict(self.adjacent_noop_by_distance).get(2, 0),
            "independent_turn_count": len(self.independent_row_ids),
            "trajectory_window_count": len(self.trajectory_windows),
            "trajectory_turn_count": self.trajectory_turn_count,
            "total_turn_exposures": total_exposures,
            "effective_trajectory_fraction": (
                self.trajectory_turn_count / total_exposures if total_exposures else 0.0
            ),
        }


class EpochSampler:
    def __init__(self, catalog: DatasetCatalog, config: SamplingConfig) -> None:
        self.catalog = catalog
        self.config = config

    def build(self, epoch: int) -> EpochPlan:
        seed = self.config.seed + epoch * 1_000_003
        generator = random.Random(seed)
        update_ids = self.catalog.row_ids(split="train", decision="UPDATE")
        noop_ids = self.catalog.row_ids(split="train", decision="NO_OP")
        noop_count = min(len(noop_ids), len(update_ids) * self.config.noop_per_update)
        adjacent_target = round(noop_count * self.config.adjacent_noop_fraction)
        adjacent_tiers = self.catalog.adjacent_row_ids(
            split="train",
            max_distance=self.config.adjacent_max_distance,
        )
        sampled_adjacent = []
        adjacent_by_distance = []
        adjacent_candidates = set()
        for distance in range(1, self.config.adjacent_max_distance + 1):
            candidates = adjacent_tiers.get(distance, [])
            adjacent_candidates.update(candidates)
            remaining = adjacent_target - len(sampled_adjacent)
            if remaining <= 0:
                adjacent_by_distance.append((distance, 0))
                continue
            selected = (
                candidates
                if len(candidates) <= remaining
                else generator.sample(candidates, remaining)
            )
            sampled_adjacent.extend(selected)
            adjacent_by_distance.append((distance, len(selected)))

        random_target = noop_count - len(sampled_adjacent)
        random_candidates = [
            row_id for row_id in noop_ids if row_id not in adjacent_candidates
        ]
        if len(random_candidates) < random_target:
            selected_set = set(sampled_adjacent)
            random_candidates.extend(
                row_id for row_id in adjacent_candidates if row_id not in selected_set
            )
        sampled_random = generator.sample(random_candidates, random_target)
        sampled_noops = sampled_adjacent + sampled_random

        independent = update_ids + sampled_noops
        generator.shuffle(independent)
        trajectory_windows = self._trajectory_windows(
            independent_count=len(independent), generator=generator
        )
        units = [TrainingUnit("independent", (row_id,)) for row_id in independent]
        units.extend(trajectory_windows)
        generator.shuffle(units)
        return EpochPlan(
            epoch=epoch,
            seed=seed,
            independent_row_ids=tuple(independent),
            trajectory_windows=tuple(trajectory_windows),
            units=tuple(units),
            update_count=len(update_ids),
            sampled_noop_count=noop_count,
            sampled_adjacent_noop_count=len(sampled_adjacent),
            sampled_random_noop_count=len(sampled_random),
            adjacent_noop_by_distance=tuple(adjacent_by_distance),
        )

    def _trajectory_windows(
        self, *, independent_count: int, generator: random.Random
    ) -> list[TrainingUnit]:
        fraction = self.config.trajectory_fraction
        if not independent_count or fraction == 0:
            return []
        target_turns = round(independent_count * fraction / (1 - fraction))
        scenarios = self.catalog.scenarios(split="train")
        scenario_rows = {
            scenario: self.catalog.scenario_row_ids(scenario) for scenario in scenarios
        }
        eligible = [
            scenario
            for scenario, rows in scenario_rows.items()
            if len(rows) >= self.config.trajectory_min_turns
        ]
        if not eligible:
            raise ValueError(
                "No training scenario is long enough for trajectory sampling"
            )

        windows = []
        sampled_turns = 0
        used: set[int] = set()
        attempts = 0
        maximum_attempts = max(1000, target_turns * 4)
        while sampled_turns < target_turns and attempts < maximum_attempts:
            attempts += 1
            scenario = generator.choice(eligible)
            rows = scenario_rows[scenario]
            desired = generator.randint(
                self.config.trajectory_min_turns,
                min(self.config.trajectory_max_turns, len(rows)),
            )
            start = generator.randint(0, len(rows) - desired)
            window = tuple(rows[start : start + desired])
            if any(row_id in used for row_id in window):
                continue
            used.update(window)
            windows.append(TrainingUnit("trajectory", window, scenario))
            sampled_turns += len(window)
        if sampled_turns < target_turns:
            raise RuntimeError(
                f"Could only sample {sampled_turns}/{target_turns} trajectory turns"
            )
        return windows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE_ROOT)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noop-per-update", type=int, default=10)
    parser.add_argument("--adjacent-noop-fraction", type=float, default=0.30)
    parser.add_argument("--adjacent-max-distance", type=int, default=2)
    parser.add_argument("--trajectory-fraction", type=float, default=0.20)
    args = parser.parse_args()
    catalog = ensure_catalog(args.data_root, default_catalog_path(args.workspace))
    plan = EpochSampler(
        catalog,
        SamplingConfig(
            noop_per_update=args.noop_per_update,
            adjacent_noop_fraction=args.adjacent_noop_fraction,
            adjacent_max_distance=args.adjacent_max_distance,
            trajectory_fraction=args.trajectory_fraction,
            seed=args.seed,
        ),
    ).build(args.epoch)
    print(json.dumps(plan.summary(), indent=2))


if __name__ == "__main__":
    main()
