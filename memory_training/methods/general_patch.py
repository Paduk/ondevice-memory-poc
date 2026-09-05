"""Domain-neutral exact-block Patch method for external memory benchmarks."""

from __future__ import annotations

from .patch import PatchMethod

GENERAL_PATCH_SYSTEM_PROMPT = """You maintain durable user memory using exact-block patches.
Given previous_memory and the current turn, return exactly one JSON object.
Store durable personal facts, preferences, events, dates, and changed information.
If nothing should change, return {"decision":"NO_OP"}.
Otherwise return {"decision":"UPDATE","operations":[...]}.
Each operation has exactly op, target, content. op is add, replace, or delete.
Use replace when newer information supersedes a stored fact.
Targets must copy a unique complete block from previous_memory exactly.
Do not emit reasoning, markdown fences, an updated full memory, or extra fields."""


class GeneralPatchMethod(PatchMethod):
    """Patch semantics with a domain-neutral memory-selection prompt."""

    name = "general_patch"
    system_prompt = GENERAL_PATCH_SYSTEM_PROMPT

