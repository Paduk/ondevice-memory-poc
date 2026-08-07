from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    content: str
    always: bool
    source: str
    path: Path


class SkillsLoader:
    def __init__(
        self,
        builtin_root: Path,
        workspace_root: Path,
    ):
        self.builtin_root = builtin_root
        self.workspace_root = workspace_root

    def list_skills(self) -> list[Skill]:
        combined: dict[str, Skill] = {}
        for skill in self._load_root(self.builtin_root, "builtin"):
            combined[skill.name] = skill
        for skill in self._load_root(self.workspace_root, "workspace"):
            combined[skill.name] = skill
        return sorted(combined.values(), key=lambda skill: skill.name)

    def select(
        self,
        user_text: str,
        *,
        max_skills: int = 3,
    ) -> list[Skill]:
        input_tokens = self._tokens(user_text)
        scored: list[tuple[int, Skill]] = []
        for skill in self.list_skills():
            if skill.always:
                scored.append((10_000, skill))
                continue
            haystack = self._tokens(f"{skill.name} {skill.description}")
            score = sum(
                1
                for skill_token in haystack
                if any(
                    skill_token in input_token or input_token in skill_token
                    for input_token in input_tokens
                )
            )
            if score:
                scored.append((score, skill))
        scored.sort(key=lambda item: (-item[0], item[1].name))
        return [skill for _, skill in scored[:max_skills]]

    @staticmethod
    def render(skills: Iterable[Skill]) -> str:
        return "\n\n---\n\n".join(
            f"### Skill: {skill.name}\n\n{skill.content}" for skill in skills
        )

    def _load_root(self, root: Path, source: str) -> list[Skill]:
        if not root.exists():
            return []
        resolved_root = root.resolve()
        skills: list[Skill] = []
        for skill_file in sorted(root.glob("*/SKILL.md")):
            try:
                resolved_file = skill_file.resolve(strict=True)
                if not resolved_file.is_relative_to(resolved_root):
                    continue
                content = resolved_file.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            metadata, body = self._parse_frontmatter(content)
            name = metadata.get("name", skill_file.parent.name).strip()
            if not name:
                continue
            skills.append(
                Skill(
                    name=name,
                    description=metadata.get("description", name).strip(),
                    content=body.strip(),
                    always=metadata.get("always", "").lower() == "true",
                    source=source,
                    path=resolved_file,
                )
            )
        return skills

    @staticmethod
    def _parse_frontmatter(
        content: str,
    ) -> tuple[dict[str, str], str]:
        normalized = content.replace("\r\n", "\n")
        if not normalized.startswith("---\n"):
            return {}, normalized
        end = normalized.find("\n---\n", 4)
        if end < 0:
            return {}, normalized
        metadata: dict[str, str] = {}
        for line in normalized[4:end].splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip().strip("\"'")
        return metadata, normalized[end + 5 :]

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {
            token.lower() for token in re.findall(r"[\w가-힣]{2,}", text, re.UNICODE)
        }
