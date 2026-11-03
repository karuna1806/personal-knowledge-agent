# ADR 0003: Database + vector store: PostgreSQL 16 + pgvector

- **Status:** accepted
- **Source:** architecture document §3.3

**Chosen:** One Postgres instance, pgvector extension, doing both relational and vector duty

**Rejected:** Qdrant, Weaviate, Chroma, Pinecone, Milvus

**Why one database instead of two:** This is the decision I'd lead with. A dedicated vector DB adds: a second system to operate, a second backup strategy, a second failure mode, and — critically — **no transactional consistency between my documents and my embeddings.** With pgvector, ingesting a document and writing its chunks is one transaction. Either both land or neither does. With Postgres + Qdrant, I own a distributed consistency problem: what happens when the Postgres write commits and the Qdrant write fails? I'd be writing reconciliation logic to solve a problem I created by splitting the store.

**The honest tradeoff:** dedicated vector DBs win at scale. Qdrant will outperform pgvector on billion-vector workloads and offers richer filtering/quantization primitives out of the box. My corpus is on the order of 10⁴–10⁵ chunks. pgvector with an HNSW index handles that with millisecond latency. **I chose the simpler system for my actual scale, not the more impressive one for a hypothetical scale.** That reasoning — "right-sized, not resume-sized" — is itself the signal.

**What pgvector gives me that a standalone vector DB doesn't:**

1. **Hybrid search in a single query.** Postgres has full-text search (`tsvector`/`tsquery`) built in. I can combine BM25-style lexical scoring with vector similarity and fuse the rankings — no second system, no application-level join.
2. **Rich metadata filtering with real SQL.** `WHERE source_type = 'code' AND metadata->>'language' = 'python' AND created_at > ...` alongside the vector search, using indexes I already understand.
3. **Foreign keys.** `ON DELETE CASCADE` from documents to chunks means deleting a document can't orphan embeddings. In a split architecture that's manual cleanup, and manual cleanup rots.

**Index choice — HNSW over IVFFlat:**

```sql
CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);
```

HNSW gives better recall-latency tradeoff and, importantly, doesn't need to be rebuilt as data grows — IVFFlat's clusters degrade as the distribution shifts, requiring periodic reindexing. HNSW costs more to build and more memory; at my scale that's a fine trade. Query-time recall is tuned with `hnsw.ef_search`.

**Distance metric — cosine:** My embedding model produces normalized vectors, so cosine and inner product rank identically; cosine is the conventional choice and keeps the operator class explicit for anyone reading the schema.
