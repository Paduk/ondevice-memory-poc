from __future__ import annotations

import re
from dataclasses import dataclass

from palmclaw_ubuntu.vehicle_bench.critic import (
    VehicleCriticContextRequestPayload,
)
from palmclaw_ubuntu.vehicle_bench.critic_trace import CriticSourceTurn

_TOKEN = re.compile(r"[\w가-힣]+", re.UNICODE)


@dataclass(frozen=True)
class CriticRetrievalHit:
    turn: CriticSourceTurn
    score: int
    matched_hints: tuple[str, ...]
    primary: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "turn_index": self.turn.turn_index,
            "message_id": self.turn.message_id,
            "score": self.score,
            "matched_hints": list(self.matched_hints),
            "primary": self.primary,
        }


@dataclass(frozen=True)
class CriticRetrievalResult:
    current_turn_index: int
    current_message_id: int
    hits: tuple[CriticRetrievalHit, ...]
    primary_message_ids: tuple[int, ...]
    fallback_to_recent: bool
    candidate_count: int

    @property
    def turns(self) -> tuple[CriticSourceTurn, ...]:
        return tuple(hit.turn for hit in self.hits)

    def as_dict(self) -> dict[str, object]:
        return {
            "current_turn_index": self.current_turn_index,
            "current_message_id": self.current_message_id,
            "primary_message_ids": list(self.primary_message_ids),
            "fallback_to_recent": self.fallback_to_recent,
            "candidate_count": self.candidate_count,
            "hits": [hit.as_dict() for hit in self.hits],
        }


class CausalTurnRetriever:
    """Deterministic hint-based retrieval over one scenario's causal prefix."""

    def __init__(self, turns: tuple[CriticSourceTurn, ...]) -> None:
        if not turns:
            raise ValueError("Causal retriever requires at least one source turn")
        expected_method = turns[0].method
        expected_scenario = turns[0].scenario_index
        keys = [(turn.turn_index, turn.message_id) for turn in turns]
        if keys != sorted(keys) or len(set(keys)) != len(keys):
            raise ValueError("Causal retriever turns must be unique and chronological")
        if any(
            turn.method != expected_method
            or turn.scenario_index != expected_scenario
            for turn in turns
        ):
            raise ValueError("Causal retriever turns must share method and scenario")
        self._turns = turns
        self._positions = {
            (turn.turn_index, turn.message_id): index
            for index, turn in enumerate(turns)
        }
        self._message_positions = {
            turn.message_id: index for index, turn in enumerate(turns)
        }

    def turn_for_message_id(self, message_id: int) -> CriticSourceTurn:
        position = self._message_positions.get(message_id)
        if position is None:
            raise ValueError(f"Unknown critic source message ID: {message_id}")
        return self._turns[position]

    def retrieve(
        self,
        current_turn: CriticSourceTurn,
        request: VehicleCriticContextRequestPayload,
        *,
        top_k: int = 8,
        neighbor_window: int = 1,
    ) -> CriticRetrievalResult:
        if top_k < 1:
            raise ValueError("Critic retrieval top_k must be positive")
        if neighbor_window < 0:
            raise ValueError("Critic retrieval neighbor window cannot be negative")
        current_position = self._positions.get(
            (current_turn.turn_index, current_turn.message_id)
        )
        if current_position is None:
            raise ValueError("Current turn is outside the retriever source")
        prefix_positions = tuple(range(current_position))
        hints = _normalized_hints(request)
        ranked: list[tuple[int, int, tuple[str, ...]]] = []
        for position in prefix_positions:
            matched_hints, score = _score_turn(
                self._turns[position],
                hints=hints,
                recency_rank=position + 1,
            )
            if matched_hints:
                ranked.append((score, position, matched_hints))
        ranked.sort(key=lambda item: (-item[0], -item[1]))
        fallback = not ranked
        if fallback:
            selected_primary_positions = tuple(prefix_positions[-top_k:])
            scores = {position: position + 1 for position in selected_primary_positions}
            matches = {position: () for position in selected_primary_positions}
        else:
            primary = ranked[:top_k]
            selected_primary_positions = tuple(item[1] for item in primary)
            scores = {position: score for score, position, _ in primary}
            matches = {position: matched for _, position, matched in primary}

        selected_positions: set[int] = set()
        for position in selected_primary_positions:
            start = max(0, position - neighbor_window)
            end = min(current_position, position + neighbor_window + 1)
            selected_positions.update(range(start, end))
        primary_positions = set(selected_primary_positions)
        hits = tuple(
            CriticRetrievalHit(
                turn=self._turns[position],
                score=scores.get(position, 0),
                matched_hints=matches.get(position, ()),
                primary=position in primary_positions,
            )
            for position in sorted(selected_positions)
        )
        return CriticRetrievalResult(
            current_turn_index=current_turn.turn_index,
            current_message_id=current_turn.message_id,
            hits=hits,
            primary_message_ids=tuple(
                self._turns[position].message_id
                for position in selected_primary_positions
            ),
            fallback_to_recent=fallback,
            candidate_count=len(prefix_positions),
        )


def _normalized_hints(
    request: VehicleCriticContextRequestPayload,
) -> tuple[str, ...]:
    ordered = (
        *request.entity_hints,
        *request.setting_hints,
        *request.temporal_hints,
    )
    normalized: list[str] = []
    seen: set[str] = set()
    for hint in ordered:
        value = " ".join(hint.casefold().replace("_", " ").split())
        if value and value not in seen:
            normalized.append(value)
            seen.add(value)
    return tuple(normalized)


def _score_turn(
    turn: CriticSourceTurn,
    *,
    hints: tuple[str, ...],
    recency_rank: int,
) -> tuple[tuple[str, ...], int]:
    content = " ".join(turn.content.casefold().replace("_", " ").split())
    content_tokens = set(_TOKEN.findall(content))
    matched: list[str] = []
    exact_count = 0
    token_overlap = 0
    for hint in hints:
        hint_tokens = set(_TOKEN.findall(hint))
        exact = hint in content
        overlap = len(hint_tokens & content_tokens)
        if exact or overlap:
            matched.append(hint)
            exact_count += int(exact)
            token_overlap += overlap
    score = exact_count * 10_000 + token_overlap * 100 + recency_rank
    return tuple(matched), score
