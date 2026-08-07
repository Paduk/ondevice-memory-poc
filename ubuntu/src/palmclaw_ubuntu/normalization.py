from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from palmclaw_ubuntu.tool_memory_schema import normalize_identifier

RUNTIME_BOUND_FIELDS = frozenset(
    {
        "driver",
        "light",
        "occupant",
        "passenger",
        "person",
        "seat",
        "side",
        "window",
        "zone",
    }
)

_TRUE_TERMS = frozenset(
    {
        "activate",
        "activated",
        "enable",
        "enabled",
        "on",
        "start",
        "started",
        "켜",
        "켜기",
        "활성화",
    }
)
_FALSE_TERMS = frozenset(
    {
        "deactivate",
        "deactivated",
        "disable",
        "disabled",
        "off",
        "stop",
        "stopped",
        "꺼",
        "끄기",
        "비활성화",
    }
)
_UPDATE_TERMS = frozenset(
    {
        "back",
        "change",
        "changed",
        "correct",
        "instead",
        "lower",
        "new",
        "now",
        "prefer",
        "raise",
        "set",
        "turn",
        "update",
        "대신",
        "바꿔",
        "변경",
        "선호",
        "수정",
        "이제",
    }
)
_UPDATE_PREFIXES = (
    "chang",
    "correct",
    "lower",
    "prefer",
    "rais",
    "set",
    "turn",
    "updat",
)
_DELETE_TERMS = frozenset(
    {
        "delete",
        "forget",
        "invalidate",
        "remove",
        "revoke",
        "삭제",
        "잊어",
        "제거",
        "철회",
    }
)
_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
    "영": 0,
    "하나": 1,
    "한": 1,
    "둘": 2,
    "두": 2,
    "셋": 3,
    "세": 3,
    "넷": 4,
    "네": 4,
    "다섯": 5,
    "여섯": 6,
    "일곱": 7,
    "여덟": 8,
    "아홉": 9,
    "열": 10,
}
_ENUM_ALIASES = {
    "fresh_air": "outside",
    "outside_air": "outside",
    "external_air": "outside",
    "recirculation": "recirculate",
    "recirculated_air": "recirculate",
    "inside_air": "recirculate",
}
_TOKEN_PATTERN = re.compile(r"[\w가-힣]+", re.UNICODE)
_NUMBER_PATTERN = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")


@dataclass(frozen=True)
class ValueSupport:
    supported: bool
    checked_count: int
    unresolved: tuple[str, ...] = ()
    ignored_runtime_fields: tuple[str, ...] = ()


def normalize_memory_value(value: Any) -> tuple[Any, tuple[str, ...]]:
    ignored: list[str] = []

    def visit(item: Any, path: str = "") -> Any:
        if isinstance(item, Mapping):
            normalized: dict[str, Any] = {}
            for key, child in item.items():
                normalized_key = normalize_identifier(str(key))
                child_path = f"{path}.{normalized_key}" if path else normalized_key
                if normalized_key in RUNTIME_BOUND_FIELDS:
                    ignored.append(child_path)
                    continue
                normalized[normalized_key] = visit(child, child_path)
            return normalized
        if isinstance(item, (list, tuple)):
            return [
                visit(child, f"{path}[{index}]")
                for index, child in enumerate(item)
            ]
        if isinstance(item, str):
            stripped = item.strip()
            scalar = _normalized_scalar_from_text(stripped)
            return stripped if scalar is None else scalar
        return item

    return visit(value), tuple(ignored)


def strip_runtime_identity_fields(
    value: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    normalized, ignored = normalize_memory_value(value)
    assert isinstance(normalized, dict)
    return normalized, ignored


def normalized_value_support(value: Any, evidence_text: str) -> ValueSupport:
    evidence_tokens = tokens(evidence_text)
    evidence_numbers = numbers(evidence_text)
    unresolved: list[str] = []
    ignored: list[str] = []
    checked = 0
    for key, leaf in value_leaves(value):
        normalized_key = normalize_identifier(key)
        if normalized_key in RUNTIME_BOUND_FIELDS:
            ignored.append(key)
            continue
        checked += 1
        if isinstance(leaf, bool):
            expected = _TRUE_TERMS if leaf else _FALSE_TERMS
            if not (expected & evidence_tokens):
                unresolved.append(f"{key}={str(leaf).lower()}")
            continue
        if isinstance(leaf, (int, float)) and not isinstance(leaf, bool):
            if float(leaf) not in evidence_numbers:
                unresolved.append(f"{key}={leaf}")
            continue
        normalized = normalize_identifier(str(leaf))
        canonical = _ENUM_ALIASES.get(normalized, normalized)
        alternatives = {
            alias
            for alias, target in _ENUM_ALIASES.items()
            if target == canonical
        } | {canonical}
        if any(set(item.split("_")) <= evidence_tokens for item in alternatives):
            continue
        unresolved.append(f"{key}={leaf}")
    return ValueSupport(
        supported=checked > 0 and not unresolved,
        checked_count=checked,
        unresolved=tuple(unresolved),
        ignored_runtime_fields=tuple(ignored),
    )


def contains_update_intent(text: str) -> bool:
    normalized_tokens = tokens(text)
    return bool(_UPDATE_TERMS & normalized_tokens) or any(
        token.startswith(prefix)
        for token in normalized_tokens
        for prefix in _UPDATE_PREFIXES
    )


def contains_delete_intent(text: str) -> bool:
    normalized_tokens = tokens(text)
    normalized_text = normalize_identifier(text)
    return bool(_DELETE_TERMS & normalized_tokens) or any(
        phrase in normalized_text
        for phrase in (
            "do_not_remember",
            "don_t_remember",
            "no_longer",
            "더_이상",
        )
    )


def numbers(text: str) -> set[float]:
    values = {float(match) for match in _NUMBER_PATTERN.findall(text)}
    normalized_tokens = [
        token.casefold()
        for token in _TOKEN_PATTERN.findall(text.replace("_", " "))
    ]
    for index, token in enumerate(normalized_tokens):
        if token not in _NUMBER_WORDS:
            continue
        value = _NUMBER_WORDS[token]
        if (
            value in {20, 30, 40, 50, 60, 70, 80, 90}
            and index + 1 < len(normalized_tokens)
            and normalized_tokens[index + 1] in _NUMBER_WORDS
        ):
            next_value = _NUMBER_WORDS[normalized_tokens[index + 1]]
            if 0 < next_value < 10:
                value += next_value
        values.add(float(value))
    return values


def tokens(text: str) -> set[str]:
    return {
        token.casefold()
        for token in _TOKEN_PATTERN.findall(text.replace("_", " "))
    }


def value_leaves(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            nested_prefix = f"{prefix}_{key}" if prefix else str(key)
            yield from value_leaves(nested, nested_prefix)
        return
    if isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            yield from value_leaves(nested, f"{prefix}_{index}")
        return
    if value is not None:
        yield prefix, value


def _normalized_scalar_from_text(value: str) -> Any | None:
    normalized = normalize_identifier(value)
    if normalized in _ENUM_ALIASES:
        return _ENUM_ALIASES[normalized]
    if normalized in _TRUE_TERMS:
        return True
    if normalized in _FALSE_TERMS:
        return False
    number_word_value = _number_words_to_value(normalized)
    if number_word_value is not None:
        return number_word_value
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)
    return None


def _number_words_to_value(normalized: str) -> int | None:
    parts = normalized.split("_")

    def under_hundred(items: list[str]) -> int | None:
        if len(items) == 1:
            value = _NUMBER_WORDS.get(items[0])
            return value if value is not None and value < 100 else None
        if len(items) == 2:
            tens = _NUMBER_WORDS.get(items[0])
            units = _NUMBER_WORDS.get(items[1])
            if tens in {20, 30, 40, 50, 60, 70, 80, 90} and units in {
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
            }:
                return tens + units
        return None

    if "hundred" not in parts:
        return under_hundred(parts)
    if len(parts) < 2 or parts[1] != "hundred":
        return None
    hundreds = _NUMBER_WORDS.get(parts[0])
    if hundreds not in {1, 2, 3, 4, 5, 6, 7, 8, 9}:
        return None
    if len(parts) == 2:
        return hundreds * 100
    remainder = under_hundred(parts[2:])
    return hundreds * 100 + remainder if remainder is not None else None
