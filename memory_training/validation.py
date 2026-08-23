"""Teacher-forced, one-step, and sequential closed-loop validation."""

from __future__ import annotations

import copy
import json
import math
import random
import shlex
import subprocess
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .dataset import DatasetCatalog, IndexedMemoryDataset
from .methods import DeltaMethod, DeltaState, MemoryMethod
from .methods.delta import DeltaBatch
from .methods.operations import normalize_memory, validate_operations
from .training_data import ChatExampleEncoder


def evaluate_teacher_forced_loss(
    model: Any,
    loader: DataLoader,
    *,
    accelerator: Any,
    max_batches: int | None = None,
) -> dict[str, float | int]:
    was_training = model.training
    model.eval()
    loss_sum = torch.tensor(0.0, device=accelerator.device)
    token_count = torch.tensor(0, device=accelerator.device)
    batch_count = 0
    with torch.no_grad():
        for batch in loader:
            if max_batches is not None and batch_count >= max_batches:
                break
            metadata = batch.pop("metadata", None)
            del metadata
            batch = {key: value.to(accelerator.device) for key, value in batch.items()}
            output = model(**batch)
            tokens = (batch["labels"] != -100).sum()
            loss_sum += output.loss.detach() * tokens
            token_count += tokens
            batch_count += 1
    gathered_loss = accelerator.reduce(loss_sum, reduction="sum")
    gathered_tokens = accelerator.reduce(token_count, reduction="sum")
    mean_loss = float((gathered_loss / gathered_tokens.clamp_min(1)).cpu())
    if was_training:
        model.train()
    return {
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20)),
        "target_tokens": int(gathered_tokens.cpu()),
        "batches": batch_count,
    }


def evaluate_one_step(
    model: Any,
    tokenizer: Any,
    method: MemoryMethod[Any],
    source: IndexedMemoryDataset,
    gold_summary: IndexedMemoryDataset,
    row_ids: Sequence[int],
    *,
    accelerator: Any,
    max_length: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    if len(source) != len(gold_summary):
        raise ValueError("Source and gold Summary views are not aligned")
    encoder = ChatExampleEncoder(tokenizer, max_length=max_length)
    counts = _DecisionCounts()
    state_scores = []
    parse_success = 0
    apply_success = 0
    prefill_tokens = 0
    generated_tokens = 0
    latency = 0.0
    reason_correct = 0
    reason_total = 0
    failures = []
    for row_id in row_ids:
        position = source.position_for_row_id(row_id)
        row = _single(source, position)
        gold_row = _single(gold_summary, position)
        gold_decision = str(row["target"]["decision"])
        state = state_from_input(method, row["input"])
        generated, prompt_count, token_count, elapsed = generate_output(
            model,
            tokenizer,
            method,
            row,
            encoder=encoder,
            accelerator=accelerator,
            max_new_tokens=max_new_tokens,
        )
        prefill_tokens += prompt_count
        generated_tokens += token_count
        latency += elapsed
        try:
            parsed = method.parse_output(generated)
            parse_success += 1
            counts.add(gold_decision, parsed.decision)
            if parsed.reason_code is not None:
                reason_total += 1
                reason_correct += int(
                    parsed.reason_code == row["target"].get("reason_code")
                )
            next_state = method.apply_output(
                state, parsed, turn_id=str(row.get("turn_id", ""))
            )
            apply_success += 1
            predicted_memory = method.materialize_memory(next_state)
        except Exception as exc:  # noqa: BLE001 - metric captures model failures.
            counts.add(gold_decision, "INVALID")
            predicted_memory = method.materialize_memory(state)
            if len(failures) < 20:
                failures.append(
                    {"row_id": row_id, "error": str(exc), "output": generated}
                )
        gold_memory = str(gold_row["target"].get("next_memory", ""))
        state_scores.append(memory_scores(predicted_memory, gold_memory))
    return {
        **counts.metrics(),
        "rows": len(row_ids),
        "schema_success_rate": parse_success / len(row_ids) if row_ids else 0.0,
        "apply_success_rate": apply_success / len(row_ids) if row_ids else 0.0,
        "state_exact": _mean(score["exact"] for score in state_scores),
        "state_f1": _mean(score["f1"] for score in state_scores),
        "reason_code_accuracy": reason_correct / reason_total if reason_total else None,
        "prefill_tokens": prefill_tokens,
        "decode_tokens": generated_tokens,
        "total_tokens": prefill_tokens + generated_tokens,
        "latency_seconds": latency,
        "tokens_per_second": generated_tokens / latency if latency else 0.0,
        "failures": failures,
    }


def evaluate_closed_loop(
    model: Any,
    tokenizer: Any,
    method: MemoryMethod[Any],
    catalog: DatasetCatalog,
    *,
    accelerator: Any,
    split: str = "validation",
    scenario_limit: int | None = None,
    turn_limit_per_scenario: int | None = None,
    max_length: int = 4096,
    max_new_tokens: int = 768,
) -> dict[str, Any]:
    scenarios = catalog.scenarios(split=split)
    if scenario_limit is not None:
        scenarios = scenarios[:scenario_limit]
    source = IndexedMemoryDataset(catalog, method.source_view, split=split)
    gold_summary = IndexedMemoryDataset(catalog, "summary", split=split)
    encoder = ChatExampleEncoder(tokenizer, max_length=max_length)
    counts = _DecisionCounts()
    all_scores = []
    scenario_reports = []
    total_tokens = 0
    total_prefill_tokens = 0
    total_latency = 0.0
    for scenario in scenarios:
        row_ids = catalog.scenario_row_ids(scenario)
        row_ids = [row_id for row_id in row_ids if source.contains_row_id(row_id)]
        if turn_limit_per_scenario is not None:
            row_ids = row_ids[:turn_limit_per_scenario]
        state = method.initial_state()
        first_error = None
        recovery_count = 0
        was_wrong = False
        scenario_scores = []
        failures = 0
        for sequence_index, row_id in enumerate(row_ids):
            position = source.position_for_row_id(row_id)
            canonical = _single(source, position)
            row = row_with_runtime_state(canonical, method, state)
            gold = _single(gold_summary, position)
            gold_decision = str(canonical["target"]["decision"])
            generated, prompt_tokens, generated_tokens, elapsed = generate_output(
                model,
                tokenizer,
                method,
                row,
                encoder=encoder,
                accelerator=accelerator,
                max_new_tokens=max_new_tokens,
            )
            total_prefill_tokens += prompt_tokens
            total_tokens += generated_tokens
            total_latency += elapsed
            try:
                parsed = method.parse_output(generated)
                counts.add(gold_decision, parsed.decision)
                state = method.apply_output(
                    state, parsed, turn_id=str(row.get("turn_id", ""))
                )
            except Exception:  # noqa: BLE001 - invalid generation keeps prior state.
                counts.add(gold_decision, "INVALID")
                failures += 1
            score = memory_scores(
                method.materialize_memory(state),
                str(gold["target"].get("next_memory", "")),
            )
            scenario_scores.append(score)
            all_scores.append(score)
            wrong = not bool(score["exact"])
            if wrong and first_error is None:
                first_error = sequence_index
            if was_wrong and not wrong:
                recovery_count += 1
            was_wrong = wrong
        final = scenario_scores[-1] if scenario_scores else {"exact": 0.0, "f1": 0.0}
        scenario_reports.append(
            {
                "scenario_index": scenario,
                "turns": len(row_ids),
                "state_exact": _mean(score["exact"] for score in scenario_scores),
                "state_f1": _mean(score["f1"] for score in scenario_scores),
                "final_state_exact": final["exact"],
                "final_state_f1": final["f1"],
                "first_error_turn": first_error,
                "recovery_count": recovery_count,
                "invalid_outputs": failures,
            }
        )
    return {
        **counts.metrics(),
        "split": split,
        "scenarios": len(scenario_reports),
        "turns": len(all_scores),
        "state_exact": _mean(score["exact"] for score in all_scores),
        "state_f1": _mean(score["f1"] for score in all_scores),
        "final_state_exact": _mean(
            report["final_state_exact"] for report in scenario_reports
        ),
        "final_state_f1": _mean(
            report["final_state_f1"] for report in scenario_reports
        ),
        "prefill_tokens": total_prefill_tokens,
        "decode_tokens": total_tokens,
        "total_tokens": total_prefill_tokens + total_tokens,
        "latency_seconds": total_latency,
        "scenario_reports": scenario_reports,
    }


def stratified_one_step_row_ids(
    catalog: DatasetCatalog,
    *,
    split: str = "validation",
    noop_ratio: float = 1.0,
    seed: int = 42,
    max_rows: int | None = None,
) -> list[int]:
    updates = catalog.row_ids(split=split, decision="UPDATE")
    noops = catalog.row_ids(split=split, decision="NO_OP")
    target_noops = min(len(noops), round(len(updates) * noop_ratio))
    selected_noops = random.Random(seed).sample(noops, target_noops)
    selected = sorted([*updates, *selected_noops])
    if max_rows is not None and len(selected) > max_rows:
        selected = random.Random(seed).sample(selected, max_rows)
        selected.sort()
    return selected


def stratified_teacher_forced_row_ids(
    catalog: DatasetCatalog,
    *,
    split: str = "validation",
    max_rows: int = 512,
    adjacent_noop_fraction: float = 0.5,
    adjacent_max_distance: int = 2,
    seed: int = 42,
) -> list[int]:
    """Build a fixed, UPDATE-preserving teacher-forced Validation subset.

    UPDATE rows are always retained. The remaining budget is split between NO_OP
    rows nearest to an UPDATE and deterministic random NO_OP rows. Keeping the row
    IDs fixed across checkpoints makes intermediate loss directly comparable while
    avoiding a prefix-biased slice of the Validation JSONL.
    """
    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    if not 0 <= adjacent_noop_fraction <= 1:
        raise ValueError("adjacent_noop_fraction must be in [0, 1]")
    if adjacent_max_distance < 1:
        raise ValueError("adjacent_max_distance must be positive")

    updates = catalog.row_ids(split=split, decision="UPDATE")
    if len(updates) > max_rows:
        raise ValueError(
            f"Validation subset size {max_rows} cannot retain all "
            f"{len(updates)} UPDATE rows"
        )
    noops = catalog.row_ids(split=split, decision="NO_OP")
    noop_budget = min(len(noops), max_rows - len(updates))
    adjacent_target = round(noop_budget * adjacent_noop_fraction)
    generator = random.Random(seed)

    tiers = catalog.adjacent_row_ids(
        split=split,
        max_distance=adjacent_max_distance,
    )
    adjacent_candidates = {
        row_id for candidates in tiers.values() for row_id in candidates
    }
    selected_adjacent: list[int] = []
    for distance in range(1, adjacent_max_distance + 1):
        remaining = adjacent_target - len(selected_adjacent)
        if remaining <= 0:
            break
        candidates = tiers.get(distance, [])
        selected_adjacent.extend(
            candidates
            if len(candidates) <= remaining
            else generator.sample(candidates, remaining)
        )

    random_target = noop_budget - len(selected_adjacent)
    selected_set = set(selected_adjacent)
    random_candidates = [
        row_id for row_id in noops if row_id not in adjacent_candidates
    ]
    if len(random_candidates) < random_target:
        random_candidates.extend(
            row_id
            for row_id in noops
            if row_id in adjacent_candidates and row_id not in selected_set
        )
    selected_random = generator.sample(random_candidates, random_target)
    return sorted([*updates, *selected_adjacent, *selected_random])


def generate_output(
    model: Any,
    tokenizer: Any,
    method: MemoryMethod[Any],
    row: dict[str, Any],
    *,
    encoder: ChatExampleEncoder,
    accelerator: Any,
    max_new_tokens: int,
) -> tuple[str, int, int, float]:
    inputs = {
        key: value.to(accelerator.device)
        for key, value in encoder.generation_inputs(row, method).items()
    }
    was_training = model.training
    model.eval()
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    if accelerator.device.type == "cuda":
        torch.cuda.synchronize(accelerator.device)
    elapsed = time.perf_counter() - start
    prompt_length = inputs["input_ids"].shape[-1]
    generated = output[0, prompt_length:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    if was_training:
        model.train()
    return text, int(prompt_length), int(generated.numel()), elapsed


def state_from_input(method: MemoryMethod[Any], value: Mapping[str, Any]) -> Any:
    if isinstance(method, DeltaMethod):
        pending = tuple(
            DeltaBatch(
                str(batch["turn_id"]),
                validate_operations(batch["operations"]),
            )
            for batch in value["pending_deltas"]
        )
        return DeltaState(
            base_summary=normalize_memory(str(value["base_summary"])),
            pending_deltas=pending,
            updates_since_compaction=int(value["updates_since_compaction"]),
        )
    return normalize_memory(str(value["previous_memory"]))


def row_with_runtime_state(
    canonical: dict[str, Any], method: MemoryMethod[Any], state: Any
) -> dict[str, Any]:
    row = copy.deepcopy(canonical)
    if isinstance(method, DeltaMethod):
        row["input"] = {
            "base_summary": state.base_summary,
            "pending_deltas": [asdict(batch) for batch in state.pending_deltas],
            "updates_since_compaction": state.updates_since_compaction,
        }
    else:
        row["input"] = {"previous_memory": method.materialize_memory(state)}
    return row


def memory_scores(predicted: str, gold: str) -> dict[str, float]:
    predicted = normalize_memory(predicted)
    gold = normalize_memory(gold)
    predicted_lines = {line.strip() for line in predicted.splitlines() if line.strip()}
    gold_lines = {line.strip() for line in gold.splitlines() if line.strip()}
    overlap = len(predicted_lines & gold_lines)
    precision = (
        overlap / len(predicted_lines) if predicted_lines else float(not gold_lines)
    )
    recall = overlap / len(gold_lines) if gold_lines else float(not predicted_lines)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "exact": float(predicted == gold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def run_quiz_hook(
    command: str, context: dict[str, Any], output_path: Path
) -> dict[str, Any]:
    """Run an optional adapter-to-Quiz bridge; it must write one JSON result."""
    context_path = output_path.with_suffix(".context.json")
    context_path.write_text(json.dumps(context, indent=2) + "\n", encoding="utf-8")
    rendered = command.format(
        context=shlex.quote(str(context_path)), output=shlex.quote(str(output_path))
    )
    subprocess.run(shlex.split(rendered), check=True)
    return json.loads(output_path.read_text(encoding="utf-8"))


class _DecisionCounts:
    def __init__(self) -> None:
        self.tp = self.fp = self.fn = self.tn = self.invalid = 0
        self.invalid_noop = 0

    def add(self, gold: str, predicted: str) -> None:
        if predicted == "INVALID":
            self.invalid += 1
            if gold == "UPDATE":
                self.fn += 1
            else:
                self.invalid_noop += 1
            return
        if gold == "UPDATE" and predicted == "UPDATE":
            self.tp += 1
        elif gold == "NO_OP" and predicted == "UPDATE":
            self.fp += 1
        elif gold == "UPDATE":
            self.fn += 1
        else:
            self.tn += 1

    def metrics(self) -> dict[str, float | int]:
        precision = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        recall = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        return {
            "update_precision": precision,
            "update_recall": recall,
            "update_f1": f1,
            "noop_specificity": self.tn / (self.tn + self.fp + self.invalid_noop)
            if self.tn + self.fp + self.invalid_noop
            else 0.0,
            "false_update_rate": self.fp / (self.tn + self.fp + self.invalid_noop)
            if self.tn + self.fp + self.invalid_noop
            else 0.0,
            "invalid_count": self.invalid,
        }


def _single(dataset: IndexedMemoryDataset, position: int) -> dict[str, Any]:
    row = dataset[position]
    if not isinstance(row, dict):
        raise TypeError("Expected one row")
    return row


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else 0.0
