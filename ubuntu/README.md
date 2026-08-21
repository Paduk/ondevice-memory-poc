# PalmClaw Ubuntu Runtime

Phase 5 implementation of the PalmClaw-inspired Ubuntu CLI agent runtime.

## Environment

```bash
source /mnt/data/miniconda3/bin/activate
conda activate /home/hj153lee/PalmClaw/.conda/ubuntu-agent
python -m pip install -e "/home/hj153lee/PalmClaw/ubuntu[dev]"
```

Cloud execution requires:

```bash
export OPENAI_API_KEY="..."
export PALMCLAW_AGENT_REASONING_EFFORT="low"
export PALMCLAW_MEMORY_REASONING_EFFORT="low"
```

The API key is read from the process environment and is not stored by the runtime.
The default models are `gpt-5.6-terra` for AgentModel, `gpt-5.6-luna` for
MemoryModel, and `text-embedding-3-small` for retrieval embeddings. Override
them with `PALMCLAW_AGENT_MODEL`, `PALMCLAW_MEMORY_MODEL`, and
`PALMCLAW_EMBEDDING_MODEL`.
Agent and memory outputs default to 2,048 and 1,024 tokens respectively. Override
them with `PALMCLAW_AGENT_MAX_OUTPUT_TOKENS` and
`PALMCLAW_MEMORY_MAX_OUTPUT_TOKENS`.

Runtime limits can be adjusted with:

- `PALMCLAW_MODEL_TIMEOUT_SECONDS`, `PALMCLAW_TOOL_TIMEOUT_SECONDS`
- `PALMCLAW_MAX_TOOL_ROUNDS`, `PALMCLAW_MAX_CONCURRENT_SESSIONS`
- `PALMCLAW_MAX_CONTEXT_TOKENS`, `PALMCLAW_MEMORY_CONTEXT_TOKENS`,
  `PALMCLAW_SKILL_CONTEXT_TOKENS`
- `PALMCLAW_MEMORY_STRATEGY=structured|summary`
- `PALMCLAW_JOINT_MEMORY_TOOL_PLANNING_ENABLED=1` enables bounded joint
  Query+Fact+Tool selection and multi-Fact Desired-state planning when
  Fact-memory retrieval is enabled. The default remains off.
- `PALMCLAW_MEMORY_BACKEND=openai|local|fake`
- `PALMCLAW_RETRIEVAL_MODE=none|full|bm25|embedding|hybrid`
- `PALMCLAW_MEMORY_TOP_K`, `PALMCLAW_EMBEDDING_DIMENSIONS`
- `PALMCLAW_MAX_FILE_BYTES`, `PALMCLAW_MAX_WEB_BYTES`,
  `PALMCLAW_MAX_WEB_REDIRECTS`
- `PALMCLAW_MEMORY_GATE_ENABLED`,
  `PALMCLAW_MEMORY_MIN_CONFIDENCE`,
  `PALMCLAW_MEMORY_GLOBAL_MIN_CONFIDENCE`
- `PALMCLAW_CLOUD_PII_REDACTION`,
  `PALMCLAW_LOCAL_PII_STORAGE=redacted|raw`
- `PALMCLAW_PATCH_MEMORY_ENABLED`,
  `PALMCLAW_PATCH_MEMORY_MODEL`,
  `PALMCLAW_PATCH_MEMORY_USER_ID`
- `PALMCLAW_PATCH_MEMORY_BATCH_SIZE` (default `32` turns),
  `PALMCLAW_PATCH_MEMORY_BATCH_TOKENS` (default `4096`),
  `PALMCLAW_PATCH_MEMORY_MAX_ATTEMPTS`,
  `PALMCLAW_PATCH_MEMORY_RETRY_DELAY_SECONDS`,
  `PALMCLAW_PATCH_MEMORY_LEASE_SECONDS`
- `PALMCLAW_FACT_MEMORY_ENABLED`,
  `PALMCLAW_FACT_MEMORY_BACKEND=openai|fake`,
  `PALMCLAW_FACT_MEMORY_MODEL`,
  `PALMCLAW_FACT_MEMORY_USER_ID`
- `PALMCLAW_FACT_MEMORY_LINKING_CONTEXT_LIMIT` (default `24`)
- `PALMCLAW_TOOL_MEMORY_RETRIEVAL_ENABLED`
- `PALMCLAW_TOOL_MEMORY_RETRIEVAL_MODE=none|full|bm25|embedding|hybrid`
- `PALMCLAW_TOOL_MEMORY_TOP_K`,
  `PALMCLAW_TOOL_MEMORY_CONTEXT_TOKENS`
- `PALMCLAW_TOOL_MEMORY_MAX_ROUTES`,
  `PALMCLAW_TOOL_MEMORY_SEMANTIC_ROUTING_ENABLED`,
  `PALMCLAW_TOOL_MEMORY_VEHICLE_ID`

Patch Memory batching uses whichever limit is reached first. VehicleMemBench
can override these with `--patch-batch-turns` and `--patch-batch-tokens`.

## Local MemoryModel

The AgentModel can remain on OpenAI while only consolidation runs on a local
SLM:

```bash
export PALMCLAW_BACKEND=openai
export PALMCLAW_MEMORY_BACKEND=local
export PALMCLAW_LOCAL_MEMORY_MODEL="<local-model-name>"
export PALMCLAW_LOCAL_MEMORY_BASE_URL="http://127.0.0.1:11434/v1"
export PALMCLAW_LOCAL_MEMORY_API_KEY="local"

palmclaw doctor
palmclaw ask "기억할 사용자 선호를 포함한 메시지"
```

The local endpoint must be an OpenAI-compatible Chat Completions server bound
to `localhost`, `127.0.0.1`, or `::1`. Remote URLs and URLs containing
credentials are rejected. llama.cpp-compatible and Ollama-compatible local
servers can be used when they expose `/v1/chat/completions`.

Local inference errors do not fail the completed Agent turn. The consolidation
run is stored as `retryable` and its uncompleted message window is eligible for
a later retry.

## Quick start

```bash
palmclaw doctor
palmclaw session new --title "PoC"
palmclaw ask "현재 workspace에 hello.txt를 작성해줘"
palmclaw history
palmclaw memory show
palmclaw memory search --mode hybrid "현재 질문"
palmclaw memory compare "현재 질문"
palmclaw memory retrievals
palmclaw memory review
palmclaw memory backend-report
```

Use a network-free deterministic backend while developing:

```bash
PALMCLAW_BACKEND=fake palmclaw ask "hello"
```

Runtime data defaults to `ubuntu/.data`. Override it with `PALMCLAW_DATA_DIR`.

## Fact-first Memory R1

R1 adds an isolated, Tool-independent storage and Patch foundation for the
revised PoC. A Fact is identified by `user_id`, `entity_id`, `predicate`,
`identity_conditions`, and `applicability`; its value and runtime Tool
arguments are deliberately excluded from the identity.

- `ADD` creates the first active version.
- `UPDATE` supersedes the active version and appends an immutable version.
- `MERGE` preserves every source record and links them to one result.
- `DELETE` creates a tombstone instead of physically deleting history.
- Evidence messages, exact quotes, spans, status changes, and idempotent Patch
  events are stored in the separate `fact_memory_*` tables introduced by
  migration `014`.

R2 connects this layer to a Cloud high-recall extraction worker and the
VehicleMem cache builder. R3/R3.1 add normalized and batched semantic
validation. R4 adds Memory-first retrieval and late Tool binding.

## Fact-first Memory R2

Enable the revised Cloud PoC instead of the older Tool-schema patch worker:

```bash
export PALMCLAW_PATCH_MEMORY_ENABLED=0
export PALMCLAW_FACT_MEMORY_ENABLED=1
export PALMCLAW_FACT_MEMORY_BACKEND=openai
export PALMCLAW_FACT_MEMORY_MODEL="${PALMCLAW_MEMORY_MODEL:-gpt-5.6-luna}"
export PALMCLAW_FACT_MEMORY_USER_ID="default_user"

palmclaw ask "Patricia는 headrest 높이 44를 선호해"
palmclaw ask "이제 Patricia는 48을 선호해"
palmclaw memory patch run
palmclaw memory fact show --all-versions
palmclaw memory fact metrics
```

The Fact worker and the older schema-patch worker are mutually exclusive
because they share the durable background queue. The worker groups queued
turns using the existing 32-turn/4,096-token limits, then sends only the
transcript and a compact set of relevant active Fact records to the
MemoryModel. It never sends the Tool ontology.

Candidates are applied in source order. Exact identity and learned partial-name
aliases produce UPDATE when the value changes; equivalent values produce NOOP;
equivalent duplicate records can produce MERGE; explicit invalidation produces
DELETE. Ambiguous semantic links are persisted as review, while unsafe or
structurally invalid evidence is rejected per candidate without retrying the
same model batch.

`memory patch trace RUN_ID` includes `fact_candidates`, while
`memory fact metrics` reports decision counts, record versions, model calls,
tokens, and cost metadata. VehicleMem cache construction and query execution
use the internal `fact_patch` strategy.

## Fact-first Memory R3

R3 keeps evidence ownership, exact source quotes, JSON shape, PII/secrets,
transactions, versions, and idempotency strict. It normalizes durable values
before applying a candidate:

- number words and numeric strings become numbers (`three` → `3`);
- action booleans and enum aliases become canonical values;
- request-time fields such as `seat`, `driver`, `person`, and `zone` are
  removed from stored values and identity conditions;
- uncertain value entailment or UPDATE/DELETE intent becomes `review`;
- invalid evidence, batch/session boundaries, PII, and malformed structure
  become `rejected`.

Each candidate stores its validation code, span repair, ignored runtime fields,
and disposition in the Fact trace. `memory fact metrics` also aggregates these
codes. Migration `016` adds the validation trace without changing legacy
Summary, Structured Hybrid, or schema-patch tables.

## Fact-first Memory R3.1

R3.1 sends only deterministic `review` candidates to the same configured
Fact MemoryModel for one structured semantic review call per extraction batch.
The reviewer compares candidate meaning, exact evidence, proposed operation,
and related active records, then returns `ACCEPT`, `REVIEW`, or `REJECT` with
confidence and a reason.

Hard failures such as missing/cross-session evidence, PII, secrets, and invalid
structure never enter or get overridden by semantic review. ACCEPT/REJECT below
0.7 confidence remains review. Provider failure, timeout, duplicate indices, or
incomplete output also leaves the candidates in review while the queue job
completes, so successful extraction is not billed again.

Semantic review uses the existing maximum 32-turn/4,096-token idle-time worker
batch and requires no additional environment setting. Its call is stored with
the `fact_memory_semantic_validation` role; `memory fact metrics` reports its
call and token counts separately. No semantic call runs when every candidate is
deterministically accepted or hard-rejected.

## Fact-first Memory R4

When Fact Memory and retrieval are enabled, foreground requests use a separate
Memory-first retriever:

```bash
export PALMCLAW_FACT_MEMORY_ENABLED=1
export PALMCLAW_TOOL_MEMORY_RETRIEVAL_ENABLED=1
export PALMCLAW_TOOL_MEMORY_RETRIEVAL_MODE=hybrid
export PALMCLAW_TOOL_MEMORY_TOP_K=5
```

The retriever starts from every active Fact owned by the current user/session.
It interprets mentioned entities, requester, driver/passenger roles, time,
weather, and situation, then jointly ranks BM25, embedding, entity, condition,
and Tool-route signals. Tool routing is a score only: an empty or incorrect
route does not remove otherwise relevant Fact candidates.

After top-k selection, late binding combines each persistent Fact value with
request-local `seat`, `light`, `zone`, or `side` values. Only unambiguous calls
that pass JSON Schema and runtime enum validation become execution hints.
Migration `017` stores Fact embeddings and every candidate score, selection,
exclusion, token count, and latency.

The ordinary Ubuntu Agent receives selected Facts and the union of routed and
late-bound Tool schemas. VehicleMemBench exposes the same path as
`cloud_fact_patch`; `oracle_tool_fact_patch` replaces only Tool routing with
gold Tool names for diagnosis.

## Tool-schema memory patch queue

Enable the Cloud patch PoC before starting a conversation:

```bash
export PALMCLAW_PATCH_MEMORY_ENABLED=1
export PALMCLAW_PATCH_MEMORY_BACKEND=openai
export PALMCLAW_PATCH_MEMORY_MODEL="${PALMCLAW_MEMORY_MODEL:-gpt-5.6-luna}"
export PALMCLAW_PATCH_MEMORY_USER_ID="default_user"

palmclaw ask "앞으로 보고서는 notes.txt에 저장해줘"
palmclaw memory patch pending
palmclaw memory patch run
palmclaw memory patch trace PATCH_RUN_ID
```

A completed Agent turn only appends an immutable queue job; it does not wait for
the Cloud patch call. `memory patch run` executes one bounded idle-time worker
cycle. It sends the redacted source turn, Tool-derived ontology, and active
records to the patch model, then applies validated `ADD`, `UPDATE`, `MERGE`, or
`DELETE` operations transactionally. Exact evidence quotes are located by the
runtime; a uniquely occurring quote repairs incorrect model offsets and records
the correction in the validation trace. In a batch, each patch has its own
transaction so one rejected patch does not roll back valid patches.

Jobs retain their source turn and message cursor. Provider errors and timeouts
move a job to `retryable` until its attempt limit; startup recovery resumes
expired work and checkpoints a run that committed immediately before a process
exit. A validation rejection retries only the source job referenced by that
patch while unrelated jobs complete. Patch-model calls use the separate
`memory_patch` trace role, including tokens, latency, privacy counters, and
configured MemoryModel cost rates.

## Tool-schema memory retrieval

Tool-memory retrieval is enabled by default when the patch queue is enabled,
or can be controlled independently:

```bash
export PALMCLAW_TOOL_MEMORY_RETRIEVAL_ENABLED=1
export PALMCLAW_TOOL_MEMORY_RETRIEVAL_MODE=hybrid
export PALMCLAW_TOOL_MEMORY_TOP_K=5
export PALMCLAW_TOOL_MEMORY_CONTEXT_TOKENS=1000

palmclaw memory tool search "보고서를 파일로 저장해줘"
palmclaw memory tool retrievals
palmclaw memory tool trace RETRIEVAL_RUN_ID
```

The router prioritizes a directly quoted request over surrounding narrative,
then combines Tool-level lexical evidence with Tool description/schema
embeddings. Stopwords and generic slot-only matches cannot create broad routes.
At most two Tools per domain and four Tools overall contribute exact
domain/topic partitions. Tool schema vectors persist in SQLite across process
restarts. Set `PALMCLAW_TOOL_MEMORY_SEMANTIC_ROUTING_ENABLED=0` for the
lexical-only ablation.

SQLite filters active records by user and the selected partitions before
session, vehicle, global, and conditional scope checks. Ranking supports full,
BM25, embedding, and normalized hybrid modes; record embeddings are cached.

Only selected compact records are added to the Agent context under a fixed
top-k and token budget. Evidence, unrelated domains, and excluded records are
not injected. A missing route or retrieval/provider failure produces empty Tool
memory rather than an arbitrary fallback. Each run persists the routing focus,
lexical/semantic scores, selection or exclusion reasons, model usage, latency,
and selected context tokens.

Before Agent inference, each selected record is also mapped deterministically
to at most one Tool call. Only mappings that satisfy the original Tool JSON
schema become candidate execution hints; ambiguous mappings and missing
required arguments are traced and omitted. VehicleMemBench connects the
router-selected Tool schemas directly to the initial Agent Tool set while
retaining module discovery as a fallback. The Agent receives a multi-Tool
checklist rule, and the simulator blocks an exact repeat of an already
successful vehicle operation.

## Structured memory and retrieval

`PALMCLAW_MEMORY_STRATEGY=structured` is the Phase 3 default. The Cloud
MemoryModel returns typed candidates with source message IDs and evidence
quotes. The runtime validates source existence and exact spans before applying
the candidate; the model cannot write the memory tables directly.

- `working` memory is the in-process context retained across rounds of the
  current turn and is not persisted as a long-term record.
- `session` memory is visible only to its originating session.
- `global` memory is visible to all sessions in the same local runtime.
- Candidate lifecycle events persist `proposed`, `verified`, `rejected`,
  `review`, and `superseded` transitions.
- The validation gate classifies evidence and existing-memory relationships as
  `entailment`, `contradiction`, or `unknown`.
- Unsupported evidence and low-confidence candidates are rejected. Unresolved
  conflicts, PII candidates, and high-sensitivity candidates enter the review
  queue instead of replacing active memory.
- Explicit updates create immutable versions linked by `supersedes_id`;
  duplicate active values are retained as rejected candidate versions.
- Retrieval supports full-memory baseline, BM25, exact cosine embedding, and
  normalized hybrid ranking.
- Every retrieval stores its query, candidate scores, rank, top-k/token-budget
  decision, selected IDs, embedding model, and failure state.

Useful inspection commands:

```bash
palmclaw memory show --all-versions
palmclaw memory inspect MEMORY_ID
palmclaw memory versions FACT_KEY
palmclaw memory search --mode bm25 --top-k 5 "query"
palmclaw memory compare "query"
palmclaw memory retrieval-trace RETRIEVAL_RUN_ID
palmclaw memory review
palmclaw memory resolve MEMORY_ID --decision accept --reason user_confirmed
palmclaw memory backend-report
```

The original Cloud Summary baseline remains available:

```bash
PALMCLAW_MEMORY_STRATEGY=summary palmclaw ask "message"
```

## Phase 5 evaluation

Evaluation results, VehicleMemBench memory caches, and evaluation temporary
files default to `/mnt/data/hj153lee/PalmClaw/{evaluation,tmp}` so large runs do
not fill the system filesystem. Set `PALMCLAW_ARTIFACT_ROOT` to move the whole
artifact tree, or use `--output-dir`/`--memory-cache-dir` for one run.

List profiles and run the deterministic, network-free reference evaluation:

```bash
palmclaw eval profiles
palmclaw eval run --mode offline --name "Phase 5 Reference"
palmclaw eval list
palmclaw eval show RUN_ID
```

Run selected profiles against the configured OpenAI MemoryModel:

```bash
palmclaw eval run \
  --mode live \
  --profiles cloud_summary,cloud_structured_bm25,cloud_structured_embedding
```

The built-in dataset contains eight versioned synthetic cases for preferences,
decisions, unsupported candidates, conflicts, updates, retrieval distractors,
and synthetic PII. `--repetitions`, `--seed`, `--dataset`, `--case-limit`, and
`--output-dir` control reproducibility and experiment scope.

Each run is stored in SQLite with its case traces and produces:

- `metrics.json` and redacted `cases.jsonl`
- `results.tsv` and `results.md`
- SVG charts for task success, memory F1, Recall@k, and Cloud PII exposure

Metrics include task success, strict schema and value-level memory F1,
unsupported-memory rate, gate/conflict quality, Recall@k/MRR/nDCG, irrelevant
injection, PII detection, Cloud exposure, latency, tokens, and estimated cost.
Cost remains zero until `PALMCLAW_MEMORY_INPUT_COST_PER_MILLION` and
`PALMCLAW_MEMORY_OUTPUT_COST_PER_MILLION` are configured.

Offline mode uses deterministic fixtures for every logical `cloud` and `local`
profile. It validates the harness and expected ablation direction; it does not
measure model quality. Live mode makes actual provider calls. The current
reference runs are:

- [deterministic 10-profile result](evaluation/results/912e041a-2a1e-4b79-bb62-8d97e94a085f/results.md)
- [OpenAI live result](evaluation/results/c4c613ce-63df-42b8-af99-3919c781e814/results.md)

The OpenAI run completed 32/32 cases. Structured memory reached value-level F1
0.750 with BM25 and 0.667 with the deterministic comparison embedding;
Recall@k was 0.500 for both. PII exposure was 0.000 with redaction and 1.000 in
the explicit no-redaction ablation. These are small synthetic PoC results, not
statistically sufficient research conclusions. No local model is downloaded or
invoked by default.

## VehicleMemBench Phases V1-V3

VehicleMemBench remains an external, trusted checkout. Pin the upstream commit
and run the network-free gold smoke:

```bash
git clone https://github.com/MINE-USTC/VehicleMemBench /path/to/VehicleMemBench
git -C /path/to/VehicleMemBench checkout \
  5ef3c48a4dbb446e6bb84a91dcc3632e9b1d203b

palmclaw eval vehicle \
  --benchmark-root /path/to/VehicleMemBench \
  --scenario 1 \
  --task-limit 10
```

The loader validates all 50 history/QA pairs, 500 tasks, 111 JSON schemas, and
every gold argument before executing the selected scenario. Each task receives
fresh predicted and reference `VehicleWorld` instances. PalmClaw executes the
gold calls through its `ToolRegistry`, while the unmodified upstream scorer
computes Exact State Match, state/value F1, and Tool F1.

The external dataset and simulator are not copied into PalmClaw. The command
records their Git commit and SHA-256 fixture/schema digests in its JSON output.
`--allow-unpinned` exists only for local adapter development and should not be
used for reported experiments.

Run the Phase V2 OpenAI Agent profiles:

```bash
palmclaw eval vehicle \
  --benchmark-root /path/to/VehicleMemBench \
  --mode live \
  --profiles no_memory,gold_memory \
  --scenario 1 \
  --task-limit 10 \
  --max-tool-rounds 10
```

Live mode exposes only `list_module_tools` initially. Requested vehicle-module
Tools are registered dynamically for the next Agent round. File, web, shell,
session history, Skills, and ordinary runtime memory are not available in this
evaluation loop. `no_memory` receives only the task query; `gold_memory`
receives the benchmark gold memory and query.

Each live run writes `manifest.json`, `metrics.json`, and `cases.jsonl` under
`/mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench/RUN_ID/`. Cases include model/Tool traces, official
scorer inputs, final simulator states, token/latency usage, and explicit
isolation flags. The Phase V2 reference run completed all 20 tasks with
`gpt-5.6-terra`: No Memory ESM was 0.30 and Gold Memory ESM was 0.90.

Run the Phase V3 Cloud memory profiles:

```bash
export PALMCLAW_MEMORY_MAX_OUTPUT_TOKENS=2048
export PALMCLAW_MODEL_TIMEOUT_SECONDS=120

palmclaw eval vehicle \
  --benchmark-root /path/to/VehicleMemBench \
  --mode live \
  --profiles cloud_summary,cloud_structured_bm25,cloud_structured_embedding,cloud_structured_hybrid \
  --scenario 1 \
  --task-limit 10 \
  --memory-batch-tokens 10000
```

The history parser preserves dates, speakers, timestamps, and the original
evidence line. Daily groups are packed into token-bounded batches and
consolidated once per scenario. The cache key fixes the dataset/history hash,
model IDs, prompt/schema versions, embedding configuration, and memory policy.
Interrupted consolidation resumes at the pending batch without duplicate
ingestion.

All tasks in a scenario share one read-only memory snapshot. Task queries are
used only for retrieval and never enter the memory message stream. Summary,
BM25, embedding, and hybrid cases record the selected memory hash; structured
profiles additionally record version, evidence, candidate score, rank, and
exclusion reason.

Live evaluation checkpoints each completed profile/task in `cases.jsonl`.
Resume a compatible run without repeating completed Agent calls:

```bash
palmclaw eval vehicle \
  --benchmark-root /path/to/VehicleMemBench \
  --mode live \
  --profiles cloud_summary,cloud_structured_bm25,cloud_structured_embedding,cloud_structured_hybrid \
  --scenario 1 \
  --task-limit 10 \
  --memory-batch-tokens 10000 \
  --resume-run RUN_ID
```

The Phase V3 scenario-1 reference run completed 40/40 tasks with
`gpt-5.6-terra` as AgentModel and `gpt-5.6-luna` as MemoryModel. Exact State
Match was 0.40 for Cloud Summary and 0.70 for each Structured BM25, Embedding,
and Hybrid profile. This is a one-scenario integration smoke, not an overall
benchmark result.

Run the Phase V4 five-scenario diagnostic gate:

```bash
palmclaw eval vehicle \
  --benchmark-root /path/to/VehicleMemBench \
  --mode live \
  --profiles no_memory,gold_memory,cloud_summary,cloud_structured_hybrid \
  --scenario 1 \
  --scenario-limit 5 \
  --task-limit 10 \
  --memory-batch-tokens 10000
```

The suite checkpoints each scenario and task, stores all 200 profile-task cases
in SQLite, and produces `results.tsv`, `reasoning_types.tsv`, `results.md`,
redacted case/diagnostic JSONL, SVG charts, and `privacy_audit.json`. A completed
suite can be passed to `--resume-run` to regenerate reports without new model
calls.

The reference five-scenario result completed 200/200 runtime cases. ESM was
0.20 for No Memory, 0.90 for Gold Memory, 0.30 for Cloud Summary, and 0.58 for
Cloud Structured Hybrid. The report includes state precision/recall/F1,
value/tool F1, reasoning-type breakdowns, unnecessary calls, latency p95,
tokens, failure taxonomy, and Cloud transmission counts.

Memory generation detected and redacted two address spans before Cloud
transmission; exposed sensitive-character rate and residual artifact PII were
both zero. The full dataset audit identifies one gold-memory task whose answer
may lose an address under redaction. Cost remains zero and is explicitly marked
unconfigured until these rates are supplied:

```bash
export PALMCLAW_AGENT_INPUT_COST_PER_MILLION="..."
export PALMCLAW_AGENT_OUTPUT_COST_PER_MILLION="..."
export PALMCLAW_MEMORY_INPUT_COST_PER_MILLION="..."
export PALMCLAW_MEMORY_OUTPUT_COST_PER_MILLION="..."
export PALMCLAW_EMBEDDING_INPUT_COST_PER_MILLION="..."
```

## Fact-first and Tool-schema patch VehicleMemBench PoC

Run `cloud_fact_patch` together with the earlier baselines after the
network-free fixture tests:

```bash
export PALMCLAW_MEMORY_MAX_OUTPUT_TOKENS=2048
export PALMCLAW_MODEL_TIMEOUT_SECONDS=120

palmclaw eval vehicle \
  --benchmark-root /path/to/VehicleMemBench \
  --mode live \
  --profiles no_memory,gold_memory,cloud_structured_hybrid,cloud_schema_patch,cloud_fact_patch \
  --scenario 1 \
  --task-limit 10
```

For `cloud_schema_patch`, every chronological history line becomes one
immutable turn and one background patch job. The Cloud PatchMemoryModel may
return `ADD`, `UPDATE`, `MERGE`, `DELETE`, or no patch. Processing is cached and
resumable; all jobs reach `completed` or `failed` before the read-only task
snapshot is used.

At query time, the Vehicle Tool schemas route to candidate domain/topic
partitions. User, scope, and condition gates run before hybrid top-k retrieval.
Only compact selected records enter the Agent prompt; raw history, evidence,
and non-selected records do not. Task queries and Tool results never update the
snapshot.

Reports add argument exact match, retrieval Recall@k/hit rate, context tokens,
retrieval latency, patch operation/status counts, duplicate/conflict/evidence
diagnostics, and separate generation/retrieval provider usage. A history line
can cause one billable patch-model call even when it produces no patch, so
inspect scenario history size and configured model prices before a live run.
No paid P5 reference run is executed automatically.

For `cloud_fact_patch`, history is processed in bounded Fact batches without
the 111-Tool ontology in the extraction prompt. Query-time retrieval does not
hard-partition Facts by Tool route, and selected facts are late-bound to current
roles before the Tool is preloaded. Use `oracle_tool_fact_patch` only to
measure the remaining Tool-routing contribution.

`cloud_recursive_summary` is the query-independent summary baseline. It
processes History in calendar-day order and exposes only
`memory_update(new_memory)` to the Memory model. An update rewrites the complete
vehicle-preference summary; no Tool call advances the day as a no-op. The final
summary is capped at 8,192 characters and injected in full for every task, so
BM25, embedding retrieval, Tool routing, and execution hints are not used.
Reports render Retrieval Recall@k as `N/A`.

`cloud_recursive_summary_patch` keeps the same daily input, final Markdown
memory, and query-time Agent path, but replaces `memory_update(new_memory)`
with minimal exact-block `add`, `replace`, and `delete` operations. The runtime
validates every target and applies all operations locally as one deterministic
update; a missing, ambiguous, malformed, empty, or no-change patch is rejected
and retried. A missing Tool call remains a no-op. Patch counts, hashes, and
local apply latency are recorded without storing raw Patch text. Its prompt and
schema versions produce an independent cache. Run it separately from other
Recursive Summary consumers so the full-rewrite and Patch variants remain a
clean ablation.

## A-MEM VehicleMemBench baseline

`cloud_amem` is an independent open-schema baseline. Each chronological
VehicleMemBench history line becomes one immutable note. The Memory model adds
context, keywords, and tags, compares a new note with at most five embedding
candidates, and may create links or new metadata versions. Query-time retrieval
selects up to ten embedding seeds and expands their direct links within one
token budget. It does not use the Vehicle Tool ontology, selector preload,
execution hints, Gold Memory, or Gold calls.

Inspect the exact note count and generation-call bound without an API key or
model call:

```bash
export PALMCLAW_MEMORY_MAX_OUTPUT_TOKENS=2048
export PALMCLAW_MEMORY_OUTPUT_COST_PER_MILLION="<configured output rate>"

palmclaw eval vehicle \
  --benchmark-root /path/to/VehicleMemBench \
  --mode live \
  --profiles cloud_amem \
  --scenario 6 \
  --amem-note-limit 100 \
  --amem-dry-run
```

For `N` selected notes, the dry-run reports `2N-1` generation calls: one
construction call per note and one evolution call after every note except the
first. Thus the 100-note command reports 199 calls. It also reports the maximum
generation output-token allowance and its output-only cost when a rate is
configured. Input-token, latency, and full-cost estimates remain `null` until a
bounded Cloud smoke supplies empirical averages. Scenario 6 currently has
2,698 history lines, so its full call bound is 5,395 before retries.

Build a bounded debug cache only after reviewing that dry-run:

```bash
palmclaw eval vehicle \
  --benchmark-root /path/to/VehicleMemBench \
  --mode live \
  --profiles cloud_amem \
  --scenario 6 \
  --task-limit 10 \
  --amem-note-limit 100 \
  --amem-link-candidates 5 \
  --amem-retrieval-top-k 10 \
  --memory-token-budget 2000
```

`--amem-note-limit` is debug-only. Its manifest is marked
`partial_history=true` and `formal_aggregate_included=false`, so the suite does
not mix it into a formal A-MEM aggregate. No Cloud command is run automatically
by tests or by the dry-run.

The scenario cache is stored below the selected memory-cache directory under
the dataset digest, scenario number, and cache digest. `manifest.json` records
the model/prompt/schema and graph parameters; `memory.db` stores notes,
versions, embeddings, links, provider calls, and retrieval traces. Re-running
the same build command resumes the first missing source and makes zero new
construction/evolution calls when the cache is ready. If only Agent evaluation
was interrupted, add `--resume-run RUN_ID` to the same command; do not change
profile, scenario, model, or cache settings.

Formal result artifacts include the immutable graph fingerprint, generation
and embedding roles, token/latency/cost accounting, retrieval selections, and
the standard residual-PII audit. Actual Scenario 6 Cloud ingestion and ESM
measurement are intentionally separate experiments and have not been run by
this implementation work.

### Similarity-gated A-MEM-style profile

`cloud_amem_style` keeps A-MEM note construction and linked retrieval, but only
calls evolution when the highest-scoring candidate reaches a configurable
cosine-similarity threshold. It uses an independent session and cache profile,
so it can be compared with `cloud_amem` without overwriting the faithful
baseline.

```bash
palmclaw eval vehicle \
  --benchmark-root /path/to/VehicleMemBench \
  --mode live \
  --profiles cloud_amem_style \
  --scenario 6 \
  --amem-style-evolution-threshold 0.75 \
  --amem-dry-run
```

For `N` notes, this profile performs `N` construction calls and between `0`
and `N-1` evolution calls. The dry-run reports `N` as the minimum and `2N-1`
as the conservative maximum. The manifest records the threshold plus the
actual evolution and skipped counts. Raising the threshold reduces calls but
may remove useful graph links, so threshold changes define different
experimental configurations.

For this efficiency profile only, a malformed/incomplete evolution response or
an ID outside the supplied candidates is recorded as a failed provider call and
recovered as `no evolution`. The immutable source note still commits, while no
links or metadata updates from the invalid response are applied. Authentication,
network, and other unclassified provider errors still stop the build.

### R0 offline patch and routing diagnostics

Replay an existing `cloud_schema_patch` run without invoking a Memory, Agent,
or embedding model:

```bash
palmclaw eval vehicle-r0 \
  --run-dir /mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench/RUN_ID \
  --memory-cache-dir /mnt/data/hj153lee/PalmClaw/evaluation/vehiclemembench-memory
```

The command opens each source `memory.db` in SQLite read-only mode and writes a
derived `r0-offline` directory. It reports raw and retry-deduplicated proposal
counts, current rejection codes, a deterministic relaxed-validation preview,
zero-memory tasks, Tool mis-routing, condition mismatch, ambiguous routing,
and recorded Recall@k. The preview never applies a patch and does not estimate
Agent E2E accuracy.

## Runtime safety

- Startup recovery marks abandoned turns, Tool calls, and consolidations as
  `interrupted`.
- One runtime process owns a data directory at a time. Inside that process,
  turns in the same session are serialized and different sessions run with a
  bounded concurrency limit.
- Context assembly applies token budgets and only includes complete Tool
  call/result chains.
- Tool traces record side-effect, retry-safety, fingerprint, state, duration,
  and structured errors. Repeated unsafe writes in one turn are denied.
- File Tools enforce canonical session/shared workspace boundaries and reject
  path traversal, escaping symlinks, and hard links.
- `web_fetch` permits public HTTPS destinations only, pins the validated
  connection IP, revalidates redirects, and bounds redirects, time, and bytes.
- Messages, Tool arguments/results, model metadata, and traces redact common
  secret forms and sensitive-key values before persistence.
- Cloud Agent, Memory, and embedding adapters apply a final outbound filter for
  secrets, email addresses, phone numbers, government IDs, payment-card
  numbers, and common street-address forms.
- The outbound filter records only category/count/exposure metrics, never the
  detected value. Local candidate storage defaults to redacted PII while the
  local conversation history remains available for the runtime.

## Verification

```bash
ruff check src tests
pytest --cov=palmclaw_ubuntu --cov-report=term-missing
python -m pip check
```

The suite includes one optional official-checkout integration smoke. It covers
the local Chat Completions contract, loopback-only endpoint enforcement, local
failure isolation, evidence/conflict gate decisions, review resolution, PII
redaction, evaluation reproducibility, ablation metrics/artifacts, and all
Phase 1-5 regressions. It also covers VehicleMemBench fixture validation,
dynamic Tool discovery, raw-history isolation, memory batching/cache/retrieval,
checkpoint resume, and scorer artifacts. The Phase 5 OpenAI live reference
uses only the built-in synthetic dataset. A real local-model quality run
requires explicitly choosing and loading a model; no model weights are
downloaded automatically.
