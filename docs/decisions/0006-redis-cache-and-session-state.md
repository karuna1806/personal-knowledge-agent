# ADR 0006: Redis: cache and session state

- **Status:** accepted
- **Source:** architecture document §3.6

**Three distinct jobs, deliberately separated by key namespace:**

| Purpose | Key pattern | TTL | Why Redis and not Postgres |
|---|---|---|---|
| Embedding cache | `emb:{model}:{sha256(text)}` | 30d | Re-embedding identical text is pure waste; hash-keyed lookup is what Redis is for |
| Query result cache | `q:{sha256(normalized_query)}` | 1h | Repeated questions during a work session return instantly |
| Session/conversation | `sess:{session_id}` | 24h | Last N turns for follow-up resolution — ephemeral by nature, high write frequency |

**The rule I'm enforcing:** *Redis holds nothing whose loss matters.* Every durable fact lives in Postgres. If Redis is flushed, the system gets slower, not wrong. This is a design invariant, not an accident, and it's the answer to "what happens if Redis goes down?" — nothing, degraded latency.

**Why not just Postgres for session state:** conversation turns are written on every message and read on every message. That's a hot-path read-modify-write on data with a 24-hour lifespan. Putting it in Postgres means table bloat, vacuum pressure, and index churn for data I'm going to throw away.
