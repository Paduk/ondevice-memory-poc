from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

from palmclaw_ubuntu.contracts import EmbeddingModel
from palmclaw_ubuntu.models import MemoryMessage, ModelUsage
from palmclaw_ubuntu.tool_memory_schema import build_tool_memory_ontology
from palmclaw_ubuntu.vehicle_bench.tools import vehicle_tool_definitions

VEHICLE_FACT_ONTOLOGY_VERSION = "vehicle-fact-ontology-v1"
VEHICLE_FACT_ONTOLOGY_RESOURCE = "data/vehicle_fact_ontology_v1.json"

_ARGUMENT_ROLES = frozenset({"content", "selector", "value"})
_FALLBACK_TARGETS = ("other", "unspecified")
_NODE_ID = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")
_TARGET_ID = re.compile(r"^[a-z][a-z0-9_]*$")


class VehicleFactOntologyError(ValueError):
    """Raised when the frozen ontology and source Tool interface diverge."""


@dataclass(frozen=True)
class VehicleFactArgumentBinding:
    name: str
    role: str
    required: bool
    schema: Mapping[str, Any]
    description_constraints: Mapping[str, Any]


@dataclass(frozen=True)
class VehicleFactToolBinding:
    tool_name: str
    description: str
    capability_id: str
    target: str
    operation: str
    arguments: tuple[VehicleFactArgumentBinding, ...]


@dataclass(frozen=True)
class VehicleFactCapability:
    id: str
    allowed_targets: tuple[str, ...]
    tool_names: tuple[str, ...]
    value_types: tuple[str, ...]


@dataclass(frozen=True)
class VehicleFactOntology:
    schema_version: str
    source_policy: str
    source_schema_sha256: str
    ontology_sha256: str
    capabilities: tuple[VehicleFactCapability, ...]
    bindings: tuple[VehicleFactToolBinding, ...]

    def capability(self, capability_id: str) -> VehicleFactCapability | None:
        return next(
            (item for item in self.capabilities if item.id == capability_id),
            None,
        )

    def binding(self, tool_name: str) -> VehicleFactToolBinding | None:
        return next(
            (item for item in self.bindings if item.tool_name == tool_name),
            None,
        )

    def resolve_target(
        self,
        capability_id: str,
        *,
        explicit_target: str | None = None,
        supported_targets: Sequence[str] = (),
    ) -> str:
        """Resolve only explicit or uniquely supported targets without guessing."""
        capability = self.capability(capability_id)
        if capability is None:
            raise VehicleFactOntologyError(
                f"Unknown canonical capability: {capability_id}"
            )
        known = set(capability.allowed_targets) - set(_FALLBACK_TARGETS)
        if explicit_target is not None:
            normalized = _normalize_target(explicit_target)
            if normalized in _FALLBACK_TARGETS:
                return normalized
            return normalized if normalized in known else "other"
        supported = {
            normalized
            for item in supported_targets
            if (normalized := _normalize_target(item)) in known
        }
        if len(supported) == 1:
            return supported.pop()
        return "unspecified"

    def capability_for_predicate(
        self,
        predicate: str,
    ) -> VehicleFactCapability | None:
        normalized = _normalize_target(predicate)
        return next(
            (
                item
                for item in self.capabilities
                if item.id.replace(".", "_") == normalized
            ),
            None,
        )


@dataclass(frozen=True)
class VehicleFactOntologyMatch:
    capability_id: str
    storage_predicate: str
    score: float
    lexical_score: float
    semantic_score: float | None
    matched_terms: tuple[str, ...]
    prompt_payload: Mapping[str, Any]


@dataclass(frozen=True)
class VehicleFactOntologyMatchResult:
    matches: tuple[VehicleFactOntologyMatch, ...]
    usage: ModelUsage
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class VehicleFactOntologyPerMessageMatchResult:
    matches: tuple[tuple[VehicleFactOntologyMatch, ...], ...]
    usage: ModelUsage
    metadata: Mapping[str, Any]


class VehicleFactOntologyMatcher:
    """Small hybrid retriever over frozen canonical capability nodes."""

    def __init__(
        self,
        ontology: VehicleFactOntology,
        *,
        embedding_model: EmbeddingModel | None = None,
        top_k_per_message: int = 2,
        max_candidates: int = 8,
        semantic_weight: float = 0.65,
    ):
        if top_k_per_message < 1:
            raise ValueError("top_k_per_message must be positive")
        if max_candidates < 1:
            raise ValueError("max_candidates must be positive")
        if not 0 <= semantic_weight <= 1:
            raise ValueError("semantic_weight must be between 0 and 1")
        self.ontology = ontology
        self.embedding_model = embedding_model
        self.top_k_per_message = top_k_per_message
        self.max_candidates = max_candidates
        self.semantic_weight = semantic_weight
        self._documents = {
            capability.id: self._capability_document(capability)
            for capability in ontology.capabilities
        }
        self._document_vectors: dict[str, tuple[float, ...]] = {}

    def match(
        self,
        messages: Sequence[MemoryMessage],
    ) -> VehicleFactOntologyMatchResult:
        queries = [
            message.content.strip()
            for message in messages
            if message.role == "user" and message.content.strip()
        ]
        if not queries:
            return VehicleFactOntologyMatchResult((), ModelUsage(), {})
        ranked_queries, usage, metadata = self._rank_queries(queries)

        aggregate: dict[str, dict[str, Any]] = {}
        for ranked in ranked_queries:
            for item in ranked[: self.top_k_per_message]:
                capability_id = str(item["capability_id"])
                previous = aggregate.get(capability_id)
                if previous is None or float(item["score"]) > float(
                    previous["score"]
                ):
                    aggregate[capability_id] = item
        selected = sorted(
            aggregate.values(),
            key=lambda item: (-float(item["score"]), str(item["capability_id"])),
        )[: self.max_candidates]
        matches = tuple(
            self._match_payload(item) for item in selected
        )
        return VehicleFactOntologyMatchResult(
            matches=matches,
            usage=usage,
            metadata={**metadata, "candidate_count": len(matches)},
        )

    def match_each(
        self,
        messages: Sequence[MemoryMessage],
    ) -> VehicleFactOntologyPerMessageMatchResult:
        """Return per-message rankings without a batch-level candidate cap."""
        queries = [
            message.content.strip()
            for message in messages
            if message.role == "user" and message.content.strip()
        ]
        if not queries:
            return VehicleFactOntologyPerMessageMatchResult(
                (),
                ModelUsage(),
                {},
            )
        ranked_queries, usage, metadata = self._rank_queries(queries)
        matches = tuple(
            tuple(
                self._match_payload(item)
                for item in ranked[: self.top_k_per_message]
            )
            for ranked in ranked_queries
        )
        return VehicleFactOntologyPerMessageMatchResult(
            matches=matches,
            usage=usage,
            metadata={
                **metadata,
                "candidate_count": sum(len(items) for items in matches),
                "per_message": True,
            },
        )

    def _rank_queries(
        self,
        queries: Sequence[str],
    ) -> tuple[list[list[dict[str, Any]]], ModelUsage, dict[str, Any]]:
        lexical = [self._lexical_scores(query) for query in queries]
        semantic: list[dict[str, float]] = [{} for _ in queries]
        usage = ModelUsage()
        semantic_error = None
        embedded_document_count = 0
        if self.embedding_model is not None:
            try:
                missing_ids = [
                    capability.id
                    for capability in self.ontology.capabilities
                    if capability.id not in self._document_vectors
                ]
                texts = [
                    *queries,
                    *(self._documents[item] for item in missing_ids),
                ]
                response = self.embedding_model.embed(texts)
                if len(response.vectors) != len(texts):
                    raise RuntimeError(
                        "Ontology EmbeddingModel returned an unexpected "
                        "vector count"
                    )
                query_vectors = response.vectors[: len(queries)]
                for capability_id, vector in zip(
                    missing_ids,
                    response.vectors[len(queries) :],
                    strict=True,
                ):
                    self._document_vectors[capability_id] = tuple(vector)
                embedded_document_count = len(missing_ids)
                usage = response.usage
                for query_index, query_vector in enumerate(query_vectors):
                    semantic[query_index] = {
                        capability.id: _cosine_similarity(
                            tuple(query_vector),
                            self._document_vectors[capability.id],
                        )
                        for capability in self.ontology.capabilities
                    }
            except Exception as exc:
                semantic_error = f"{type(exc).__name__}: {exc}"
        ranked_queries = [
            self._rank_query(
                lexical[query_index],
                semantic[query_index],
            )
            for query_index in range(len(queries))
        ]
        return (
            ranked_queries,
            usage,
            {
                "query_count": len(queries),
                "embedding_model_id": (
                    self.embedding_model.model_id
                    if self.embedding_model is not None
                    else None
                ),
                "embedded_document_count": embedded_document_count,
                "cached_document_count": (
                    len(self._document_vectors) - embedded_document_count
                ),
                "semantic_error": semantic_error,
            },
        )

    def _rank_query(
        self,
        lexical: Mapping[str, tuple[float, tuple[str, ...]]],
        semantic: Mapping[str, float],
    ) -> list[dict[str, Any]]:
        max_lexical = max((item[0] for item in lexical.values()), default=0.0)
        positive_semantic = {
            key: max(0.0, float(value)) for key, value in semantic.items()
        }
        max_semantic = max(positive_semantic.values(), default=0.0)
        ranked = []
        for capability in self.ontology.capabilities:
            lexical_score, matched_terms = lexical.get(
                capability.id,
                (0.0, ()),
            )
            semantic_score = semantic.get(capability.id)
            if lexical_score <= 0 and semantic_score is None:
                continue
            lexical_normalized = (
                lexical_score / max_lexical if max_lexical else 0.0
            )
            semantic_normalized = (
                max(0.0, float(semantic_score)) / max_semantic
                if semantic_score is not None and max_semantic
                else 0.0
            )
            score = (
                (1 - self.semantic_weight) * lexical_normalized
                + self.semantic_weight * semantic_normalized
                if semantic
                else lexical_normalized
            )
            ranked.append(
                {
                    "capability_id": capability.id,
                    "score": score,
                    "lexical_score": lexical_score,
                    "semantic_score": semantic_score,
                    "matched_terms": matched_terms,
                }
            )
        ranked.sort(
            key=lambda item: (-float(item["score"]), str(item["capability_id"]))
        )
        return ranked

    def _lexical_scores(
        self,
        query: str,
    ) -> dict[str, tuple[float, tuple[str, ...]]]:
        query_terms = set(_tokens(query))
        scores = {}
        for capability in self.ontology.capabilities:
            strong_terms = {
                *_tokens(capability.id),
                *(
                    term
                    for target in capability.allowed_targets
                    for term in _tokens(target)
                ),
            }
            document_terms = set(_tokens(self._documents[capability.id]))
            strong_overlap = query_terms & strong_terms
            general_overlap = query_terms & document_terms
            score = len(strong_overlap) * 4 + len(general_overlap)
            if score:
                scores[capability.id] = (
                    float(score),
                    tuple(sorted(general_overlap)),
                )
        return scores

    def _capability_document(
        self,
        capability: VehicleFactCapability,
    ) -> str:
        bindings = [
            binding
            for binding in self.ontology.bindings
            if binding.capability_id == capability.id
        ]
        return " ".join(
            [
                capability.id,
                *capability.allowed_targets,
                *(
                    f"{binding.target} {binding.description} "
                    + " ".join(item.name for item in binding.arguments)
                    for binding in bindings
                ),
            ]
        )

    def _match_payload(
        self,
        item: Mapping[str, Any],
    ) -> VehicleFactOntologyMatch:
        capability_id = str(item["capability_id"])
        capability = self.ontology.capability(capability_id)
        assert capability is not None
        bindings = [
            binding
            for binding in self.ontology.bindings
            if binding.capability_id == capability_id
        ]
        prompt_payload = {
            "capability_id": capability.id,
            "storage_predicate": capability.id.replace(".", "_"),
            "allowed_targets": list(capability.allowed_targets),
            "value_types": list(capability.value_types),
            "bindings": [
                {
                    "target": binding.target,
                    "operation": binding.operation,
                    "arguments": [
                        {
                            "name": argument.name,
                            "role": argument.role,
                            "required": argument.required,
                            "schema": dict(argument.schema),
                            "description_constraints": dict(
                                argument.description_constraints
                            ),
                        }
                        for argument in binding.arguments
                    ],
                }
                for binding in bindings
            ],
        }
        semantic_score = item.get("semantic_score")
        return VehicleFactOntologyMatch(
            capability_id=capability_id,
            storage_predicate=capability_id.replace(".", "_"),
            score=float(item["score"]),
            lexical_score=float(item["lexical_score"]),
            semantic_score=(
                float(semantic_score) if semantic_score is not None else None
            ),
            matched_terms=tuple(item["matched_terms"]),
            prompt_payload=prompt_payload,
        )


def load_vehicle_fact_ontology_v1(
    tool_schemas: Sequence[Mapping[str, Any]],
) -> VehicleFactOntology:
    resource = files("palmclaw_ubuntu.vehicle_bench").joinpath(
        VEHICLE_FACT_ONTOLOGY_RESOURCE
    )
    definition = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(definition, Mapping):
        raise VehicleFactOntologyError("Ontology resource must contain an object")
    return build_vehicle_fact_ontology(tool_schemas, definition)


def build_vehicle_fact_ontology(
    tool_schemas: Sequence[Mapping[str, Any]],
    definition: Mapping[str, Any],
) -> VehicleFactOntology:
    schema_version = str(definition.get("schema_version", ""))
    if schema_version != VEHICLE_FACT_ONTOLOGY_VERSION:
        raise VehicleFactOntologyError(
            f"Unsupported ontology version: {schema_version!r}"
        )
    source_policy = str(definition.get("source_policy", "")).strip()
    if not source_policy:
        raise VehicleFactOntologyError("Ontology source_policy cannot be empty")
    expected_source_hash = str(definition.get("source_schema_sha256", ""))
    actual_source_hash = vehicle_tool_schema_sha256(tool_schemas)
    if expected_source_hash != actual_source_hash:
        raise VehicleFactOntologyError(
            "Vehicle Tool schema hash does not match ontology-v1: "
            f"expected {expected_source_hash}, got {actual_source_hash}"
        )

    raw_tools = definition.get("tools")
    if not isinstance(raw_tools, Mapping) or not raw_tools:
        raise VehicleFactOntologyError("Ontology tools must be a non-empty object")
    schemas_by_name = _schemas_by_name(tool_schemas)
    mapped_names = {str(name) for name in raw_tools}
    source_names = set(schemas_by_name)
    if mapped_names != source_names:
        missing = sorted(source_names - mapped_names)
        extra = sorted(mapped_names - source_names)
        raise VehicleFactOntologyError(
            f"Ontology Tool coverage mismatch; missing={missing}, extra={extra}"
        )

    operations = {
        item.tool_name: item.action
        for item in build_tool_memory_ontology(
            vehicle_tool_definitions(tool_schemas)
        ).tools
    }
    bindings = tuple(
        _build_binding(
            tool_name,
            schemas_by_name[tool_name],
            raw_tools[tool_name],
            operation=operations[tool_name],
        )
        for tool_name in sorted(source_names)
    )
    _validate_reverse_bindings(bindings)
    capabilities = _build_capabilities(bindings)
    return VehicleFactOntology(
        schema_version=schema_version,
        source_policy=source_policy,
        source_schema_sha256=actual_source_hash,
        ontology_sha256=_sha256_json(definition),
        capabilities=capabilities,
        bindings=bindings,
    )


def vehicle_tool_schema_sha256(
    tool_schemas: Sequence[Mapping[str, Any]],
) -> str:
    normalized = sorted(
        (dict(schema) for schema in tool_schemas),
        key=lambda item: str(item.get("name", "")),
    )
    return _sha256_json(normalized)


def _schemas_by_name(
    tool_schemas: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for index, schema in enumerate(tool_schemas):
        name = schema.get("name")
        if not isinstance(name, str) or not name:
            raise VehicleFactOntologyError(
                f"Tool schema at index {index} has no valid name"
            )
        if name in result:
            raise VehicleFactOntologyError(f"Duplicate Tool schema: {name}")
        parameters = schema.get("parameters")
        if not isinstance(parameters, Mapping):
            raise VehicleFactOntologyError(
                f"Tool parameters must be an object: {name}"
            )
        result[name] = schema
    if not result:
        raise VehicleFactOntologyError("Tool schema collection cannot be empty")
    return result


def _build_binding(
    tool_name: str,
    schema: Mapping[str, Any],
    raw_binding: Any,
    *,
    operation: str,
) -> VehicleFactToolBinding:
    if not isinstance(raw_binding, Mapping):
        raise VehicleFactOntologyError(
            f"Ontology binding must be an object: {tool_name}"
        )
    capability_id = str(raw_binding.get("capability", ""))
    target = str(raw_binding.get("target", ""))
    if not _NODE_ID.fullmatch(capability_id):
        raise VehicleFactOntologyError(
            f"Invalid capability id for {tool_name}: {capability_id!r}"
        )
    if not _TARGET_ID.fullmatch(target) or target in _FALLBACK_TARGETS:
        raise VehicleFactOntologyError(
            f"Invalid concrete target for {tool_name}: {target!r}"
        )
    raw_arguments = raw_binding.get("arguments")
    if not isinstance(raw_arguments, Mapping):
        raise VehicleFactOntologyError(
            f"Ontology arguments must be an object: {tool_name}"
        )
    parameters = schema["parameters"]
    properties = parameters.get("properties", {})
    if not isinstance(properties, Mapping):
        raise VehicleFactOntologyError(
            f"Tool properties must be an object: {tool_name}"
        )
    mapped_arguments = {str(name) for name in raw_arguments}
    source_arguments = {str(name) for name in properties}
    if mapped_arguments != source_arguments:
        raise VehicleFactOntologyError(
            f"Argument coverage mismatch for {tool_name}; "
            f"source={sorted(source_arguments)}, "
            f"mapped={sorted(mapped_arguments)}"
        )
    required = {
        str(name)
        for name in parameters.get("required", [])
        if isinstance(name, str)
    }
    arguments = []
    value_argument_count = sum(
        str(raw_arguments[name]) == "value" for name in source_arguments
    )
    for name in sorted(source_arguments):
        role = str(raw_arguments[name])
        if role not in _ARGUMENT_ROLES:
            raise VehicleFactOntologyError(
                f"Invalid argument role for {tool_name}.{name}: {role!r}"
            )
        argument_schema = properties[name]
        if not isinstance(argument_schema, Mapping):
            raise VehicleFactOntologyError(
                f"Argument schema must be an object: {tool_name}.{name}"
            )
        arguments.append(
            VehicleFactArgumentBinding(
                name=name,
                role=role,
                required=name in required,
                schema=dict(argument_schema),
                description_constraints=_description_constraints(
                    str(schema.get("description", "")),
                    argument_schema,
                    enabled=role == "value" and value_argument_count == 1,
                ),
            )
        )
    if arguments and not any(
        item.role in {"content", "value"} for item in arguments
    ):
        raise VehicleFactOntologyError(
            f"Tool binding has selectors but no value/content: {tool_name}"
        )
    return VehicleFactToolBinding(
        tool_name=tool_name,
        description=str(schema.get("description", "")).strip(),
        capability_id=capability_id,
        target=target,
        operation=operation,
        arguments=tuple(arguments),
    )


def _validate_reverse_bindings(
    bindings: Sequence[VehicleFactToolBinding],
) -> None:
    seen: set[tuple[str, str, str, tuple[tuple[str, str], ...]]] = set()
    for binding in bindings:
        key = (
            binding.capability_id,
            binding.target,
            binding.operation,
            tuple((item.name, item.role) for item in binding.arguments),
        )
        if key in seen:
            raise VehicleFactOntologyError(
                "Ambiguous reverse Tool binding: "
                f"{binding.capability_id}/{binding.target}/{binding.operation}"
            )
        seen.add(key)


def _build_capabilities(
    bindings: Sequence[VehicleFactToolBinding],
) -> tuple[VehicleFactCapability, ...]:
    grouped: dict[str, list[VehicleFactToolBinding]] = {}
    for binding in bindings:
        grouped.setdefault(binding.capability_id, []).append(binding)
    capabilities = []
    for capability_id, items in sorted(grouped.items()):
        value_types = {
            str(argument.schema.get("type", "any"))
            for item in items
            for argument in item.arguments
            if argument.role == "value"
        }
        capabilities.append(
            VehicleFactCapability(
                id=capability_id,
                allowed_targets=tuple(
                    sorted({item.target for item in items} | set(_FALLBACK_TARGETS))
                ),
                tool_names=tuple(sorted(item.tool_name for item in items)),
                value_types=tuple(sorted(value_types)),
            )
        )
    return tuple(capabilities)


def _normalize_target(value: str) -> str:
    return re.sub(
        r"[^a-z0-9]+", "_", value.strip().casefold()
    ).strip("_")


def _description_constraints(
    description: str,
    argument_schema: Mapping[str, Any],
    *,
    enabled: bool,
) -> dict[str, Any]:
    """Conservatively recover explicit constraints omitted from JSON Schema."""
    if not enabled:
        return {}
    groups = re.findall(r"\(([^()]*)\)", description)
    argument_type = argument_schema.get("type")
    for group in groups:
        range_match = re.fullmatch(r"\s*(-?\d+)\s*-\s*(-?\d+)\s*", group)
        if range_match and argument_type in {"integer", "number"}:
            return {
                "maximum": int(range_match.group(2)),
                "minimum": int(range_match.group(1)),
            }
        lower_match = re.fullmatch(r"\s*>\s*(-?\d+)\s*", group)
        if lower_match and argument_type in {"integer", "number"}:
            return {"exclusiveMinimum": int(lower_match.group(1))}
        quoted = re.findall(r"'([^']+)'", group)
        if len(quoted) >= 2 and argument_type == "string":
            return {"enum": quoted}
        stripped = group.strip()
        if (
            argument_type == "string"
            and re.fullmatch(r"[a-zA-Z0-9_]+(?:/[a-zA-Z0-9_]+)+", stripped)
        ):
            return {"enum": stripped.split("/")}
    return {}


def _sha256_json(value: Any) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(
        item
        for item in re.findall(
            r"[\w가-힣]+",
            value.casefold().replace("_", " ").replace(".", " "),
            re.UNICODE,
        )
        if item
    )


def _cosine_similarity(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    if len(first) != len(second) or not first:
        return 0.0
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if not first_norm or not second_norm:
        return 0.0
    return sum(left * right for left, right in zip(first, second, strict=True)) / (
        first_norm * second_norm
    )
