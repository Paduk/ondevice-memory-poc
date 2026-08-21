# PalmClaw Engineering Documentation

This directory is the public engineering knowledge base for PalmClaw. It records how the app is structured, which reusable improvements are planned, how changes are verified, and why major engineering changes were made.

## Documents

- [Vehicle Memory methodology progress sheet](memory-methodology-progress-sheet.md): current research candidate, method-by-method status, pipeline bottlenecks, next work queue, and experiment log.
- [Vehicle Memory P0 fixed-cache Agent repeats](memory-methodology-p0-agent-repeat-results.md): three-run stability comparison of Recursive-assisted and Post-normalized Ontology-assisted memory.
- [Vehicle Memory P1 stable-failure analysis](memory-methodology-p1-stable-failure-analysis.md): generic selector, late-binding, and condition-matching causes separated from Fact absence and under-specified cases.
- [Vehicle Memory Scenario 1–5 comparison](memory-methodology-s1-s5-comparison-results.md): Recursive Summary versus Post-normalized Recursive-assisted Ours, including cross-scenario selector, recall, execution, and efficiency evidence.
- [Vehicle Memory Scenario 1–10 structural comparison](memory-methodology-s1-s10-structural-bottleneck-comparison.md): methodology-driven bottleneck taxonomy and the proposed Memory Bundle, desired-state, and schema-constrained planning architecture.
- [Joint Memory–Tool Planning implementation plan](joint-memory-tool-planning-implementation-plan.md): staged implementation of joint Tool selection, Memory Bundles, Desired State, schema-valid multi-Tool planning, fallback, and evaluation gates.
- [Current best Ours summary](current-best-ours-summary.md): concise overview of the current Fact Memory plus Joint Memory–Tool Planning method, S1–10 results, and remaining validation boundary.
- [Joint Tool Selection Scenario 6–10 results](joint-tool-selection-s6-s10-results.md): same-cache paired E2E comparison, selector misses, discovery efficiency, taskwise churn, and adoption decision.
- [Joint Tool Selection Scenario 1–5 results](joint-tool-selection-s1-s5-results.md): same-cache paired E2E comparison showing selector-recall gains, candidate expansion regressions, and the current hold decision.
- [Architecture](architecture.md): application layers, runtime ownership, agent-turn execution, persistence, and extension points.
- [Engineering roadmap](roadmap.md): source-verified completed work and the current reusable improvement backlog.
- [Ubuntu Agent Runtime implementation plan](ubuntu-agent-runtime-implementation-plan.md): compact phase-by-phase implementation sequence and acceptance criteria.
- [Ubuntu Agent Runtime PoC detailed plan](ubuntu-agent-runtime-poc-plan.md): detailed runtime port, memory research, security, and evaluation design.
- [VehicleMemBench integration plan](vehiclemembench-integration-plan.md): benchmark adapter phases, evaluation isolation rules, and acceptance criteria.
- [VehicleMemBench V4 method examples](vehiclemembench-v4-method-examples.md): concise examples of the four memory profiles and State/Tool F1 interpretation.
- [Fact-first incremental memory methodology](tool-schema-on-device-memory-methodology.md): high-recall Fact extraction, bounded patch operations, and late Tool binding.
- [Fact-first incremental memory implementation plan](tool-schema-on-device-memory-implementation-plan.md): Cloud PoC redesign followed by Local SLM and Android validation.
- [Fact-first R5 evaluation](tool-schema-on-device-memory-r5-results.md): 50-task comparison, Fact acceptance, cost, gate decision, and failure analysis.
- [Fact-first holdout evaluation](tool-schema-on-device-memory-holdout-s6-s10-results.md): frozen-setting Scenario 6–10 comparison against Structured Hybrid.
- [Fact-first Oracle evaluation plan](fact-first-oracle-evaluation-plan.md): stage-isolated Gold interventions for extraction, structure, gate, routing, retrieval, and late binding.
- [Schema-informed Fact ontology v1 plan](schema-informed-fact-ontology-v1-plan.md): build and validate a compact canonical Fact ontology from the public Vehicle Tool interface without QA, History, or Gold data.
- [Schema-informed Fact ontology v1 audit](schema-informed-fact-ontology-v1.md): frozen 111-Tool mapping, coverage checks, target policy, and remaining extractor-integration scope.
- [Schema-informed Fact Scenario 6 smoke](schema-informed-fact-s6-smoke-results.md): matcher, candidate generation, validation, and final active-Fact coverage without Quiz Agent calls.
- [Schema-informed Recursive-assisted Scenario 6](schema-informed-recursive-assisted-s6-results.md): combined profile safety guards, recovered Facts, E2E result, remaining binding failures, and call cost.
- [Schema-informed Recursive-assisted Scenario 6–10](schema-informed-recursive-assisted-s6-s10-results.md): 50-task comparison, taskwise churn, Fact coverage loss, token cost, and gate decision.
- [Post-normalized Recursive-assisted Scenario 6](post-normalized-recursive-assisted-s6-results.md): high-recall extraction with conservative post-normalization, candidate preservation, E2E smoke, and cost.
- [Post-normalized Recursive-assisted Scenario 6–10](post-normalized-recursive-assisted-s6-s10-results.md): 50-task E2E comparison, retrieval and structure gains, Agent-run variability, cost, and adoption gate.
- [Fact-first vs Hybrid E2E evaluation summary](fact-first-hybrid-e2e-evaluation-summary.md): reproducible evaluation flow, current results, E2E workflow, and Oracle-backed bottlenecks.
- [Recursive Summarization baseline](recursive-summarization-baseline-implementation-plan.md): methodology, implementation contract, cache/trace design, tests, and staged VehicleMemBench evaluation plan.
- [A-MEM VehicleMemBench baseline](amem-vehiclemembench-baseline-implementation-plan.md): controlled open-schema note/link/evolution methodology, similarity-gated `cloud_amem_style`, isolation rules, cache and staged Cloud evaluation gates.
- [Compact A-MEM-style implementation](compact-amem-style-implementation-plan.md): episode compaction, deterministic linking, correction-gated evolution, VehicleMemBench integration, and completed Scenario 6 validation.
- [Compact A-MEM-style Scenario 6 results](compact-amem-style-s6-results.md): full-cache compression, calls, token/latency accounting, 10-task E2E scores, retrieval failures, and A-MEM-style dry-run comparison.
- [LLM-Wiki-inspired Vehicle Memory](llm-wiki-inspired-vehicle-memory-implementation-plan.md): Summary-gated and Post-normalized Fact-Wiki profiles, bounded Agent-native traversal, staged cost accounting, and evaluation gates.
- [LLM-Wiki-inspired Scenario 6–10 results](llm-wiki-inspired-vehicle-memory-s6-s10-results.md): fixed-cache paired accuracy, traversal subsets, staged token/latency cost, Cloud regressions, and hold decision.
- [A-MEM baseline implementation record](amem-implementation-execution-plan.md): six completed implementation checkpoints, network-free verification, failure recovery, CLI, and remaining Cloud-smoke boundary.
- [Recursive Summarization bottleneck analysis](recursive-summarization-bottleneck-analysis.md): Scenario 6–10 final/intermediate summary coverage audit, recursive forgetting and distortion cases, downstream execution failures, and comparison with Fact-first.
- [Ours × Recursive combination experiment](ours-recursive-combination-experiment-plan.md): staged Hybrid and Recursive-assisted Fact-first implementation, smoke, holdout evaluation, and stop conditions.
- [Ubuntu PoC source-backed reference](palmclaw-ubuntu-source-reference.md): detailed current Runtime, Tool, Memory, privacy, persistence, evaluation behavior, source/test evidence, and explicit implementation limits.
- [Testing and QA](testing.md): automated checks, build verification, and manual regression checklists.
- [Engineering history](history/README.md): completed initiatives retained for context after they leave the active roadmap.

## Maintenance Rules

Update these documents in the same change as the related implementation when any of the following occurs:

- A runtime, storage, provider, tool, channel, or UI boundary changes.
- A roadmap item is started, completed, replaced, or found to be already implemented.
- A regression requires a new automated test or manual QA case.
- A large refactor changes the main owner of a workflow.

Use one of these status labels in the roadmap:

- `Planned`: agreed work with no implementation yet.
- `In progress`: implementation has started but acceptance checks are incomplete.
- `Source-verified`: confirmed in the current source; automated or manual checks are listed separately when available.
- `Deferred`: useful work that is intentionally not in the current development stage.

Do not mark an item `Source-verified` based only on a benchmark run, demo, generated trace, or local experiment.

## Public Documentation Boundary

This directory is intended to be safe to publish with the repository. Do not include:

- API keys, tokens, cookies, passwords, or private endpoints.
- Personal account details, private channel or calendar data, or contact records.
- Device serial numbers or machine-specific absolute paths.
- Raw benchmark traces containing user data.
- Temporary evaluation scripts presented as product capabilities.

Machine-specific setup can be described using placeholders when it is useful to contributors. Security-sensitive findings should follow [SECURITY.md](../../SECURITY.md) instead of being recorded here.

## Writing Style

Keep engineering documents concise and source-grounded. State the current behavior, the owner in the codebase, the remaining problem, and a verifiable acceptance condition. Avoid release marketing and task-specific workarounds.
