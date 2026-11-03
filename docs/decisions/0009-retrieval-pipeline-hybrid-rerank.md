# ADR 0009: Retrieval pipeline: hybrid + rerank

- **Status:** accepted
- **Source:** architecture document §3.9

Plain vector search is where most portfolio RAG projects stop, and it's the most common source of bad answers. Three-stage pipeline instead:

```
query
  │
  ├──► vector search (pgvector, HNSW)     ── top 20 by cosine
  │
  ├──► lexical search (Postgres tsvector) ── top 20 by ts_rank
  │
  ├──► Reciprocal Rank Fusion             ── merge to top 20
  │
  ├──► cross-encoder rerank (bge-reranker) ── top 5
  │
  └──► prompt assembly with citation markers
```

**Why hybrid and not vector-only:** dense embeddings are semantically strong but lexically weak. A query for an exact identifier — `HNSWIndexError`, a specific function name, a config key — is precisely where embeddings underperform and BM25-style matching excels. For a corpus containing code, that's not an edge case, it's a large fraction of my queries. Vector-only retrieval on a code corpus is a known failure mode and skipping hybrid would be a real defect.

**Why RRF for fusion:** it merges rankings without needing the two scores to be on a comparable scale, which cosine similarity and `ts_rank` are not. Simple, robust, no tuning of score-normalization weights.

**Why a cross-encoder reranker:** bi-encoder retrieval embeds query and document independently, so it can't model their interaction. A cross-encoder scores the pair jointly and is substantially more accurate — but it's too slow to run over the whole corpus. Retrieve 20 cheaply, rerank 20 precisely, pass 5 to the LLM. Classic recall-then-precision funnel.

**Why this matters more with a local model:** context window is scarcer (8k realistic vs 128k) and small models degrade faster with irrelevant context. Precision in retrieval *is* answer quality here. With a frontier model you can be sloppy and stuff 50 chunks in; with a 7B model, garbage in the context window actively produces wrong answers. Reranking isn't a nice-to-have in this architecture, it's load-bearing.

**Chunking strategy — structure-aware, not fixed-size:**
- Markdown/notes: split on heading boundaries, keep the heading path in `metadata` so citations can say *which section*
- Code: split on function/class boundaries via tree-sitter, never mid-function, with file path and line range in metadata
- PDFs: split on layout blocks, preserve page number for citation

Fixed-size character chunking with overlap is the default everyone uses and it's the wrong default — it cuts functions in half and separates headings from their content. Being able to explain *why* structure-aware chunking improved my retrieval scores, with numbers, is a concrete depth signal.
