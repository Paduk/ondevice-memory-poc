from __future__ import annotations

from memory_training.scripts.analyze_patch_delta_error_dynamics import (
    TurnState,
    classify_fact_episodes,
    summarize_turns,
)


Fact = tuple[str, str, str]


def _facts(*values: Fact) -> frozenset[Fact]:
    return frozenset(values)


def test_fact_episode_separates_acquisition_from_noop_loss() -> None:
    first = ("Alex", "2026-01-01", "volume=20")
    second = ("Alex", "2026-01-02", "volume=25")
    turns = [
        TurnState(
            1, "UPDATE", "UPDATE", _facts(), _facts(first), _facts(), _facts(first)
        ),
        TurnState(
            2,
            "NO_OP",
            "UPDATE",
            _facts(first),
            _facts(first),
            _facts(first),
            _facts(),
        ),
        TurnState(
            3,
            "NO_OP",
            "UPDATE",
            _facts(first),
            _facts(first),
            _facts(),
            _facts(first),
        ),
        TurnState(
            4,
            "UPDATE",
            "UPDATE",
            _facts(first),
            _facts(second),
            _facts(first),
            _facts(second),
        ),
    ]

    counts = classify_fact_episodes(turns)

    assert counts["introduced_facts"] == 2
    assert counts["immediately_acquired_facts"] == 2
    assert counts["first_loss_on_noop"] == 1
    assert counts["recovered_after_first_loss"] == 1
    assert counts["retained_until_retirement_or_end"] == 1
    assert counts["retired_immediately_removed"] == 1


def test_noop_summary_counts_only_new_damage_as_loss() -> None:
    gold = ("Alex", "2026-01-01", "volume=20")
    extra = ("Alex", "2026-01-01", "volume=99")
    turns = [
        TurnState(
            1,
            "NO_OP",
            "UPDATE",
            _facts(gold),
            _facts(gold),
            _facts(gold),
            _facts(gold, extra),
        ),
        TurnState(
            2,
            "NO_OP",
            "NO_OP",
            _facts(gold),
            _facts(gold),
            _facts(gold, extra),
            _facts(gold, extra),
        ),
        TurnState(
            3,
            "NO_OP",
            "UPDATE",
            _facts(gold),
            _facts(gold),
            _facts(gold, extra),
            _facts(extra),
        ),
    ]

    summary = summarize_turns(turns)
    counts = summary["counts"]

    assert counts["noop_new_extra_turns"] == 1
    assert counts["noop_new_loss_turns"] == 1
    assert counts["noop_error_persistence_turns"] == 1
    assert counts["noop_memory_mutations"] == 2
