from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

REDACTED = "[REDACTED]"
_SENSITIVE_KEYS = {
    "apikey",
    "accesskey",
    "accesstoken",
    "authorization",
    "authtoken",
    "cookie",
    "password",
    "refreshtoken",
    "secret",
    "secretkey",
    "setcookie",
    "token",
}

_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(
        r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/=-]{12,}",
    ),
    re.compile(
        r"(?i)\b(api[_-]?key|access[_-]?token|auth[_-]?token|token|password|cookie)"
        r"\b(\s*[:=]\s*)([^\s,;]+)"
    ),
)

_PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "email",
        re.compile(
            r"(?i)(?<![\w.+-])[A-Z0-9._%+-]+@"
            r"(?:[A-Z0-9-]+\.)+[A-Z]{2,63}(?![\w-])"
        ),
    ),
    (
        "government_id",
        re.compile(r"(?<!\d)(?:\d{6}[- ]?[1-4]\d{6}|\d{3}-\d{2}-\d{4})(?!\d)"),
    ),
    (
        "credit_card",
        re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)"),
    ),
    (
        "phone",
        re.compile(
            r"(?<![\w\d])(?:\+\d{1,3}[-.\s]?)?"
            r"(?:\(\d{2,4}\)|\d{2,4})[-.\s]"
            r"\d{3,4}[-.\s]\d{4}(?![\w\d])"
        ),
    ),
    (
        "address",
        re.compile(
            r"(?ix)(?<!\w)"
            r"(?:\d{1,6}\s+(?:[A-Z][\w.-]*\s+){1,4}"
            r"(?:street|st|road|rd|avenue|ave|boulevard|blvd|lane|ln|drive|dr)"
            r"|(?:[가-힣]+[ \t]+){0,4}"
            r"[가-힣A-Za-z0-9·.-]+(?:로|길)[ \t]+\d{1,5}(?:-\d{1,5})?)"
            r"(?!\w)"
        ),
    ),
)


@dataclass(frozen=True)
class PrivacySpan:
    category: str
    start: int
    end: int
    text: str = field(repr=False)


@dataclass(frozen=True)
class PrivacyReport:
    detected_count: int = 0
    redacted_count: int = 0
    category_counts: dict[str, int] = field(default_factory=dict)
    input_chars: int = 0
    output_chars: int = 0
    sensitive_chars_before: int = 0
    sensitive_chars_after: int = 0

    def as_dict(self, *, destination: str = "cloud") -> dict[str, Any]:
        return {
            "destination": destination,
            "detected_count": self.detected_count,
            "redacted_count": self.redacted_count,
            "category_counts": dict(self.category_counts),
            "input_chars": self.input_chars,
            "output_chars": self.output_chars,
            "sensitive_chars_before": self.sensitive_chars_before,
            "sensitive_chars_after": self.sensitive_chars_after,
        }


def detect_sensitive_spans(
    text: str,
    *,
    include_pii: bool = True,
    allowlist: tuple[str, ...] = (),
) -> tuple[PrivacySpan, ...]:
    candidates: list[PrivacySpan] = []
    for pattern_index, pattern in enumerate(_SECRET_PATTERNS):
        category = "secret"
        for match in pattern.finditer(text):
            if pattern_index == 1:
                start, end = match.span()
            elif pattern_index == 2:
                start, end = match.span(3)
            else:
                start, end = match.span()
            candidates.append(
                PrivacySpan(
                    category=category,
                    start=start,
                    end=end,
                    text=text[start:end],
                )
            )
    if include_pii:
        for category, pattern in _PII_PATTERNS:
            for match in pattern.finditer(text):
                matched = match.group(0)
                if category == "phone" and not _plausible_phone(matched):
                    continue
                if category == "credit_card" and not _luhn_valid(matched):
                    continue
                candidates.append(
                    PrivacySpan(
                        category=category,
                        start=match.start(),
                        end=match.end(),
                        text=matched,
                    )
                )

    allowed = set(allowlist)
    ordered = sorted(
        (span for span in candidates if span.text not in allowed),
        key=lambda span: (span.start, -(span.end - span.start), span.category),
    )
    selected: list[PrivacySpan] = []
    for span in ordered:
        if any(span.start < item.end and item.start < span.end for item in selected):
            continue
        selected.append(span)
    return tuple(sorted(selected, key=lambda span: span.start))


def redact_for_cloud(
    text: str,
    *,
    include_pii: bool = True,
    allowlist: tuple[str, ...] = (),
) -> tuple[str, PrivacyReport]:
    spans = detect_sensitive_spans(
        text,
        include_pii=include_pii,
        allowlist=allowlist,
    )
    redacted = text
    for span in reversed(spans):
        replacement = f"[REDACTED_{span.category.upper()}]"
        redacted = redacted[: span.start] + replacement + redacted[span.end :]
    remaining = detect_sensitive_spans(
        redacted,
        include_pii=include_pii,
        allowlist=allowlist,
    )
    categories = Counter(span.category for span in spans)
    return redacted, PrivacyReport(
        detected_count=len(spans),
        redacted_count=len(spans),
        category_counts=dict(sorted(categories.items())),
        input_chars=len(text),
        output_chars=len(redacted),
        sensitive_chars_before=sum(span.end - span.start for span in spans),
        sensitive_chars_after=sum(span.end - span.start for span in remaining),
    )


def redact_data_for_cloud(
    value: Any,
    *,
    include_pii: bool = True,
    allowlist: tuple[str, ...] = (),
    preserve_opaque_keys: tuple[str, ...] = ("encrypted_content",),
) -> tuple[Any, PrivacyReport]:
    reports: list[PrivacyReport] = []
    opaque_keys = {
        re.sub(r"[^a-z0-9]", "", key.lower()) for key in preserve_opaque_keys
    }

    def visit(item: Any, key: str | None = None) -> Any:
        normalized_key = re.sub(r"[^a-z0-9]", "", (key or "").lower())
        if normalized_key in opaque_keys:
            return item
        if normalized_key in _SENSITIVE_KEYS:
            text = str(item)
            reports.append(
                PrivacyReport(
                    detected_count=1,
                    redacted_count=1,
                    category_counts={"secret": 1},
                    input_chars=len(text),
                    output_chars=len(REDACTED),
                    sensitive_chars_before=len(text),
                )
            )
            return REDACTED
        if isinstance(item, str):
            redacted, report = redact_for_cloud(
                item,
                include_pii=include_pii,
                allowlist=allowlist,
            )
            reports.append(report)
            return redacted
        if isinstance(item, dict):
            return {str(k): visit(v, str(k)) for k, v in item.items()}
        if isinstance(item, list):
            return [visit(child) for child in item]
        if isinstance(item, tuple):
            return tuple(visit(child) for child in item)
        return item

    redacted = visit(value)
    return redacted, _merge_reports(reports)


def inspect_data_privacy(
    value: Any,
    *,
    include_pii: bool = True,
    allowlist: tuple[str, ...] = (),
) -> PrivacyReport:
    texts: list[str] = []

    def collect(item: Any, key: str | None = None) -> None:
        normalized_key = re.sub(r"[^a-z0-9]", "", (key or "").lower())
        if normalized_key == "encryptedcontent":
            return
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, dict):
            for child_key, child in item.items():
                collect(child, str(child_key))
        elif isinstance(item, (list, tuple)):
            for child in item:
                collect(child)

    collect(value)
    reports = []
    for text in texts:
        spans = detect_sensitive_spans(
            text,
            include_pii=include_pii,
            allowlist=allowlist,
        )
        categories = Counter(span.category for span in spans)
        sensitive_chars = sum(span.end - span.start for span in spans)
        reports.append(
            PrivacyReport(
                detected_count=len(spans),
                category_counts=dict(sorted(categories.items())),
                input_chars=len(text),
                output_chars=len(text),
                sensitive_chars_before=sensitive_chars,
                sensitive_chars_after=sensitive_chars,
            )
        )
    return _merge_reports(reports)


def redact_secrets(text: str) -> str:
    redacted = text
    for index, pattern in enumerate(_SECRET_PATTERNS):
        if index == 1:
            redacted = pattern.sub(r"\1 " + REDACTED, redacted)
        elif index == 2:
            redacted = pattern.sub(r"\1\2" + REDACTED, redacted)
        else:
            redacted = pattern.sub(REDACTED, redacted)
    return redacted


def redact_data(value: Any) -> Any:
    if isinstance(value, str):
        return redact_secrets(value)
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = re.sub(r"[^a-z0-9]", "", str(key).lower())
            redacted[str(key)] = (
                REDACTED if normalized_key in _SENSITIVE_KEYS else redact_data(item)
            )
        return redacted
    if isinstance(value, list):
        return [redact_data(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_data(item) for item in value)
    return value


def _merge_reports(reports: list[PrivacyReport]) -> PrivacyReport:
    categories: Counter[str] = Counter()
    for report in reports:
        categories.update(report.category_counts)
    return PrivacyReport(
        detected_count=sum(report.detected_count for report in reports),
        redacted_count=sum(report.redacted_count for report in reports),
        category_counts=dict(sorted(categories.items())),
        input_chars=sum(report.input_chars for report in reports),
        output_chars=sum(report.output_chars for report in reports),
        sensitive_chars_before=sum(report.sensitive_chars_before for report in reports),
        sensitive_chars_after=sum(report.sensitive_chars_after for report in reports),
    )


def _plausible_phone(value: str) -> bool:
    digits = re.sub(r"\D", "", value)
    if not 9 <= len(digits) <= 15:
        return False
    if re.fullmatch(r"\d{4}[-./]\d{1,2}[-./]\d{1,2}", value.strip()):
        return False
    return len(set(digits)) > 1


def _luhn_valid(value: str) -> bool:
    digits = [int(character) for character in value if character.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0
