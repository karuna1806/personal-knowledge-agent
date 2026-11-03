# ADR 0010: Deviations from the architecture document

- **Status:** accepted
- **Context:** the architecture document was written before implementation. Building it
  surfaced places where it was internally inconsistent, didn't match the actual models and
  hardware, or left something unspecified. Each deviation is listed here with its reason, so
  nothing differs silently.

## 1. Queue framework: aio-pika directly, not arq (amends ADR 0005)

arq is a Redis-only job queue; it cannot use RabbitMQ as its broker. ADR 0005 chose RabbitMQ
*for* dead-letter exchanges, per-message ack with `prefetch_count=1`, and topic routing, and chose
arq *for* being asyncio-native. The two choices are incompatible as stated.

**Decision:** keep RabbitMQ (its reasons stand) and use **aio-pika**, the asyncio AMQP client,
with a small consumer loop (`src/pka/workers/runner.py`). This preserves both rationales: AMQP
failure semantics, and one async mental model shared with the gateway. The retry/DLQ topology
arq would not have provided is explicit in `src/pka/queue/broker.py`:

- main queues dead-letter into `pka.dlx` → `pka.dlq`
- transient failure → republished (confirmed) to `pka.retry`, a TTL queue that dead-letters
  back to the main exchange; after `QUEUE_MAX_ATTEMPTS` (3) the message is rejected → DLQ
- permanent failure (unsupported/missing file, empty document) → job marked `failed`, acked,
  not retried; retrying cannot help and would only fill the DLQ with noise

## 2. Embedding dimension: 1024, configurable (amends §1.2 / §4)

The schema said `VECTOR(768)`; the embedding model already in use (`Qwen/Qwen3-Embedding-0.6B`)
outputs 1024 dimensions. The migration takes the dimension from `EMBED_DIM`, and the embedder
refuses to start if the model's real output size differs, rather than failing on first insert.

## 3. Package layout: `src/pka/…` (amends §7)

A top-level `src/mcp/` package would shadow the `mcp` SDK it needs to import. Everything lives
under one package, `pka`, with the same sub-structure as §7 (`api`, `agent`, `memory`, `mcp`,
`ingest`, `llm`, `db`, `workers`, `core`) plus `queue`.

## 4. `ingest_jobs` table (adds to §4)

Async ingestion needs somewhere to report status (`queued → running → done | skipped | failed |
dead`) and a way to tie a DLQ entry back to its source. Migration 0002 adds it, plus indexes on
`chunks.embed_model`, `documents.source_uri` and active `memories(subject, predicate)`.
Migration 0001 is §4 verbatim.

## 5. Context window: 4096 by default (amends §3.9's "8k realistic")

Ollama runs models with a 4096-token context unless `OLLAMA_CONTEXT_LENGTH` is raised, and
through the OpenAI-compatible endpoint it **silently drops the start of an over-long prompt** —
i.e. the system prompt with the grounding rules. `LLM_CONTEXT_TOKENS` defaults to 4096 and the
prompt builder budgets to 85% of (context − answer tokens). Raise both together for 8k.

## 6. Thinking disabled (`LLM_REASONING_EFFORT=none`)

The available model (`gemma4:12b`) "thinks" by default through Ollama, which turned a one-word
routing decision into a 27-second call. `reasoning_effort: "none"` brings it to ~1 s. It's a
config value because vLLM and other servers may reject the parameter (the GPU override sets it
empty).

## 7. Own read-only MCP servers instead of third-party ones (implements ADR 0007)

The four servers in ADR 0007 (filesystem, git, postgres, fetch) are implemented in-repo with the
MCP Python SDK (`src/pka/mcp/servers/`) rather than pulled via `npx`/`uvx`. Reasons: the
"read-only, blast radius zero" rule is then enforced by code we own and test (path confinement
incl. symlinks; git read subcommands only, no shell; SQL via a SELECT-only role *and* a READ ONLY
transaction *and* a statement allowlist); no Node runtime in the image; and the official
Postgres server is archived. Any third-party MCP server can still be added in `mcp_servers.yaml`.

## 8. Router emits the standalone query (extends ADR 0007 rule 2)

§3.6 keeps the last N turns "for follow-up resolution" but doesn't say how. The router's single
constrained call returns `{route, standalone_query}` when there is history, so retrieval searches
for "What is RabbitMQ's DLQ for?" rather than "what about its DLQ?". With no history it emits only
the route (saves ~1 s of generation).

## 9. Background LLM work yields to interactive queries (extends §5)

On one local inference server, memory extraction competes with the next user query: measured
router latency went from ~1.6 s to ~10 s when an extraction was in flight. In-flight queries
register in a Redis sorted set (scored by expiry, so a crash can't wedge it); the memory worker
waits until no query has run for 3 s. This is the "queue protects the user experience" argument
made literal.

## 10. A changed file replaces its previous version

`UNIQUE (source_uri, content_hash)` makes re-ingesting identical content a no-op, but allows two
versions of the same file to coexist — and both would be retrievable. The ingest transaction
inserts the new version and deletes older versions of the same `source_uri` atomically.
Consequence: citations that pointed at the old version's chunks are removed with them
(`ON DELETE CASCADE`), so historical answers lose their provenance links after a file changes.
Keeping old versions behind an `is_current` flag is the fix if that history matters.

## 11. Memory supersession is keyed on (subject, predicate)

A new value for the same `(subject, predicate)` supersedes the old one. That is right for
single-valued facts (`prefers_queue_framework`) and wrong for multi-valued ones (`works_on`
project A *and* B). The extraction prompt steers the model toward reusable single-valued
predicates; a cardinality field per predicate is the proper fix. Restating an existing value
reinforces it (higher confidence wins) instead of creating a duplicate row.

## 12. Query-cache key includes corpus version and config

`q:{sha256(normalized_query)}` alone would serve stale answers after new documents are ingested.
The hash input also includes a corpus version counter (bumped on every ingest/delete) and a
fingerprint of model + retrieval config. Only first-turn, unfiltered, `retrieve`-route answers
are cached; follow-ups depend on history and tool answers on live data.
