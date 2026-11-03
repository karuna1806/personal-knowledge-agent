# ADR 0001: Language: Python

- **Status:** accepted
- **Source:** architecture document §3.1

**Chosen:** Python 3.12+

**Rejected:** Go, Rust, TypeScript

**Why:** The entire ML/RAG ecosystem is Python-native — `sentence-transformers`, `pgvector`'s Python bindings, tokenizers, eval frameworks. Writing the retrieval and eval layer in Go would mean reimplementing or shelling out to Python anyway. Python's async story (asyncio) is adequate for an I/O-bound gateway, which is what this is: nearly every operation is a network wait on Postgres, Redis, or the inference server.

**What would change my mind:** If this became a high-throughput multi-tenant service where the gateway itself was CPU-bound (thousands of concurrent SSE connections), Go's goroutine model would win. At single-user scale, Python's GIL is not the bottleneck — the LLM is.

**The honest caveat to state in interview:** Python is the right call *here* specifically because the bottleneck is inference latency, not request handling. Choosing Python for a rate limiter or a proxy would be a mistake. Knowing which is which is the point.
