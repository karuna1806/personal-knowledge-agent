# Personal Knowledge Agent

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)
![Inference: local](https://img.shields.io/badge/inference-100%25%20local-green.svg)

A self-hosted RAG agent grounded in my own notes, code and documents, with a persistent
memory layer. Fully local inference: no external LLM API anywhere in the request path.

It ingests a corpus (markdown, source code, PDFs, web pages), retrieves with hybrid search
(pgvector HNSW + Postgres full-text, fused with RRF, reranked by a cross-encoder), and streams
grounded answers whose citations attach to the specific claims they support. After each turn
a background worker extracts durable facts about me into a bitemporal memory store, which
later questions use as clearly-labelled, possibly-stale context.

It is **not** a "digital clone": retrieval gives the model my information, not my judgment.

Design rationale lives in [`docs/decisions/`](docs/decisions/) (one ADR per decision; ADR 0010
lists every place the implementation deviates from the original architecture and why).
Measured results are in [`docs/evals/`](docs/evals/).

**At a glance** (34-question golden set, default config — [details](#results)):

| Recall@5 | MRR | Refusal on out-of-corpus questions | False refusals | Answers with valid citations |
|---|---|---|---|---|
| **0.982** | **0.905** | **6/6** | **0/28** | **100%** |

Hybrid search + cross-encoder reranking lifts Recall@5 from 0.893 (vector-only) to 0.982 and
MRR from 0.767 to 0.905. Time to first token is ~11 s on a laptop with a 12B model — the
bottleneck is LLM prefill, not retrieval (see [Results](#results)).

```
            ┌──────────── FastAPI gateway (auth · validation · rate limit · SSE) ────────────┐
 POST /query│ router ─► retrieve: embed → HNSW ∪ tsvector → RRF → cross-encoder → top 5      │
            │   (1 constrained   tool:     ≤3 constrained MCP calls (read-only servers)      │
            │    LLM call)       direct:   no lookup                                         │
            │ → prompt (token budget; sources ≠ memory) → stream → span citations → persist  │
            └───────┬──────────────────────────────┬───────────────────────────┬────────────┘
                    │ sync                         │ publish                   │
        Postgres 16 + pgvector            RabbitMQ (topic) ──► workers (aio-pika, prefetch=1)
        documents · chunks · memories     ingest · reindex · memory   chunk · embed · extract
        citations · query_logs · evals    retry (TTL) · DLQ           supersede
                    ▲                                                        │
        Redis: emb cache · query cache · sessions · priority gate ◄─────────┘
        LLM: Ollama (dev) / vLLM (GPU) — same OpenAI-compatible API
```

## Quick start

Requirements: Docker, and an OpenAI-compatible LLM server. On a Mac the simplest is
[Ollama](https://ollama.com) on the host (containerised Ollama has no Metal acceleration):

```bash
git clone https://github.com/karuna1806/personal-knowledge-agent.git
cd personal-knowledge-agent
ollama pull gemma4:12b        # or any instruct model; set LLM_MODEL to match

cp -r ~/notes corpus/                         # whatever it should know (git-ignored)
docker compose up -d                          # postgres, redis, rabbitmq, migrate, api, worker
open http://localhost:8000                    # UI (API key: dev-key)
```

Then ingest and ask:

```bash
curl -X POST localhost:8000/ingest -H 'x-api-key: dev-key' \
     -H 'content-type: application/json' -d '{"path": "."}'           # everything in ./corpus

curl -N -X POST localhost:8000/query -H 'x-api-key: dev-key' \
     -H 'content-type: application/json' -d '{"query": "Why did I choose RabbitMQ?"}'
```

The first `up` downloads the embedding model (~1.2 GB) and reranker (~1.1 GB) into the
`hf-cache` volume. To reuse a host cache instead: `HF_CACHE=~/.cache/huggingface docker compose up -d`.

Variants:

| | |
|---|---|
| `docker compose --profile ollama up -d` | Ollama in a container too (Linux; then `LLM_BASE_URL=http://ollama:11434/v1`) |
| `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d` | vLLM on an NVIDIA GPU instead of Ollama |
| `docker compose --profile observability up -d` | + Prometheus `:9090`, Grafana `:3000` (dashboard provisioned), Jaeger `:16686` (set `OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4318`) |

### Running natively (faster on a Mac: the embedder and reranker use MPS)

```bash
uv sync
docker compose up -d postgres redis rabbitmq
uv run alembic upgrade head
uv run uvicorn pka.api.main:app --port 8000      # terminal 1
uv run python -m pka.workers                     # terminal 2
```

Defaults in `src/pka/core/config.py` already point at the compose services on localhost.

## Configuration

Everything is an environment variable (or `.env`). The ones most likely to need changing:

| Variable | Default | Notes |
|---|---|---|
| `API_KEY` | `dev-key` | sent as `X-API-Key` or `Authorization: Bearer`. Empty disables auth. |
| `CORPUS_ROOT` | `./corpus` (`/corpus` in compose, mounted from `CORPUS_DIR`) | `POST /ingest {"path"}` and the filesystem/git tools are confined to it |
| `LLM_BASE_URL` / `LLM_MODEL` | `http://localhost:11434/v1` / `gemma4:12b` | any OpenAI-compatible server |
| `LLM_REASONING_EFFORT` | `none` | turns off "thinking" (25 s → 1 s for a routing call). Set empty for servers that reject it |
| `LLM_CONTEXT_TOKENS` | `4096` | **must match the server.** Ollama defaults to 4096 and silently truncates the *start* of longer prompts. For 8k: `OLLAMA_CONTEXT_LENGTH=8192` on the server *and* `LLM_CONTEXT_TOKENS=8192` here |
| `EMBED_MODEL` / `EMBED_DIM` | `Qwen/Qwen3-Embedding-0.6B` / `1024` | dimension is checked at startup; changing it needs a fresh migration |
| `RERANK_ENABLED` / `RERANK_MODEL` | `true` / `BAAI/bge-reranker-base` | |
| `RETRIEVAL_MODE` | `hybrid` | `hybrid` \| `vector` \| `lexical` |
| `CHUNKER` | `structure` | `structure` \| `fixed` (baseline for experiments) |
| `FINAL_K` | `5` | chunks given to the LLM; the main latency/quality knob on small hardware |
| `MEMORY_ENABLED`, `MCP_ENABLED` | `true` | |

MCP tool servers are configured in [`mcp_servers.yaml`](mcp_servers.yaml).

## API

Interactive docs at `/docs`. Summary:

| Endpoint | |
|---|---|
| `POST /query` | `{query, conversation_id?, mode: auto\|retrieve\|tool\|direct, stream, filters?}`. SSE events: `meta` → `sources` → `token`… → `done` (citations with char spans, timings) or `error`. `stream=false` returns JSON |
| `POST /ingest` | `{path}` (file or directory under `CORPUS_ROOT`) or `{url}` → `202` with job ids |
| `POST /ingest/upload` | multipart file → `202` |
| `GET /ingest/jobs[/{id}]` | `queued → running → done \| skipped \| failed \| dead` |
| `GET /documents`, `GET /documents/{id}`, `DELETE /documents/{id}`, `GET /chunks/{id}` | corpus |
| `GET /conversations`, `GET /conversations/{id}/messages` | history with span citations |
| `GET /memories[?include_history=true]`, `GET /memories/{id}/history`, `POST /memories/{id}/retract` | what the agent believes about me, and why |
| `POST /admin/reindex`, `GET /admin/queues`, `GET /admin/dlq`, `POST /admin/dlq/replay`, `GET /admin/stats` | ops |
| `GET /health`, `GET /ready`, `GET /metrics` | liveness, dependency readiness, Prometheus |

## How it behaves

**Ingestion is always async.** The API records a job and publishes to RabbitMQ; a worker
chunks, embeds (Redis-cached by content hash) and writes the document and all its chunks in
one transaction. Re-ingesting unchanged content is a no-op (`UNIQUE (source_uri,
content_hash)`); a changed file atomically replaces its previous version. Transient failures
retry up to 3 times through a delay queue, then land in the DLQ (`GET /admin/dlq`) — never
silently dropped. Chunking is structure-aware: markdown by heading (heading path kept for
citations), code by function/class via tree-sitter (never mid-function; oversized classes split
into methods), PDFs by page.

**Queries are router-first.** One constrained-decoding call picks `retrieve | tool | direct`
from a fixed enum and, for follow-ups, rewrites the question to stand alone. `retrieve` runs
the hybrid → RRF → rerank funnel; `tool` runs at most 3 MCP calls, each tool choice and argument
object generated under its JSON schema and validated before the call; all shipped tools are
read-only (filesystem and git confined to `CORPUS_ROOT`; SQL through a SELECT-only role in a
READ ONLY transaction). The prompt keeps numbered `<sources>` (citable evidence) visibly separate
from `<user_memory>` (possibly stale; never cited), and tells the model to answer "I don't know
based on the available sources." when the sources don't contain the answer.

**Memory is append-only.** Extraction runs in the worker after each turn, waiting until no
interactive query has run for 3 s so it doesn't steal the GPU from the next question. A new value
for an existing `(subject, predicate)` supersedes the old row (`superseded_by`, `valid_until`)
instead of overwriting it, so `GET /memories/{id}/history` shows every belief and when it changed.

**Redis holds nothing whose loss matters.** Embedding cache, query cache (keyed on the corpus
version, so any ingest invalidates it), session history (falls back to Postgres), rate-limit
counters, the priority gate. Flush it and the system gets slower, not wrong.

## Results

Measured on an Apple M5 (24 GB), `gemma4:12b` Q4_K_M via Ollama, Qwen3-Embedding-0.6B,
bge-reranker-base; corpus = the architecture document + three early scripts (53 chunks);
golden set = 34 questions ([`evals/golden_set.yaml`](evals/golden_set.yaml)). The corpus
itself is not in the repo (`corpus/` is git-ignored). To evaluate on other documents, put
them in `corpus/` and write a golden set whose locators point at phrases in them.

| Configuration | R@5 | MRR | nDCG@5 | Identifier R@5 |
|---|---|---|---|---|
| **Default:** structure chunks · hybrid · rerank | **0.982** | **0.905** | **0.914** | **1.000** |
| hybrid, no rerank | 0.929 | 0.767 | 0.796 | 0.857 |
| vector-only, no rerank | 0.893 | 0.767 | 0.789 | 0.714 |
| fixed-size chunks · hybrid · rerank | 0.946 | 0.887 | 0.887 | 1.000 |
| fixed-size chunks · hybrid, no rerank | 0.875 | 0.815 | 0.810 | 0.714 |
| rerank input truncated to 256 tokens | 0.839 | 0.764 | 0.771 | 0.571 |

Generation (default config): refusal accuracy on out-of-corpus questions **6/6**, false
refusals **0/28**, every answer cited, **0** invalid citations. Faithfulness 1.000 is judged by
the same 12B model, so treat it as an upper bound (a manual spot-check agreed).
TTFT p50 **11.2 s**, p95 18.5 s: the pipeline before the LLM takes ~0.3 s natively; the rest is
12B prefill at ~300–500 tok/s on this laptop. Full write-up, hypotheses and caveats (small
corpus, small n) in [`docs/evals/README.md`](docs/evals/README.md).

## Development

```bash
uv sync
docker compose up -d postgres redis rabbitmq
uv run pytest                      # 63 tests: unit + integration (real pg/redis/rabbit, stub LLM)
uv run python -m evals.run --name my-change --generate
```

Integration tests use an isolated database (`pka_test`), Redis db 15 and RabbitMQ vhost
`pka_test`, and a deterministic OpenAI-compatible stub server, so they exercise the real HTTP,
streaming and structured-output paths without a GPU. They skip automatically if the
infrastructure isn't running.

## Known gaps

- Single user; one API key. No multi-tenancy (by design, §11 of the architecture).
- TTFT on laptop hardware is well above the 2 s target (see Results) — the bottleneck is LLM
  prefill, not the pipeline.
- Docker on macOS has no GPU access, so in containers the embedder and reranker run on CPU and
  rerank costs ~5 s per query (vs ~0.2 s natively on MPS). On a Mac, prefer running the API and
  worker natively; the containerised stack is the portable/Linux path.
- `docker-compose.gpu.yml` (vLLM + CUDA torch) is written but has not been run here — this
  machine has no NVIDIA GPU. Treat it as a starting point and verify on real hardware.
- Memory supersession treats every `(subject, predicate)` as single-valued.
- Faithfulness is judged by the same 12B model that answers; a larger judge would be more
  trustworthy.
- A changed file's old chunks are deleted, taking their historical citations with them.
- No OCR for scanned PDFs; no incremental reindex when the embedding *dimension* changes.
- The golden set is a 34-question starter on a small corpus. Grow both before trusting small
  differences between configurations.

## License

[MIT](LICENSE)
