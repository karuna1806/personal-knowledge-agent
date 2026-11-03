# ADR 0005: Message queue: RabbitMQ + arq

- **Status:** accepted
- **Source:** architecture document §3.5

**Chosen:** RabbitMQ as broker, `arq` as the task framework

**Rejected:** Celery, Redis Streams alone, Kafka, `BackgroundTasks`

**Why a broker at all:** Ingesting a large document is chunk → embed → store, and embedding is the slow part (seconds to minutes for a large corpus). This cannot live in a request. It also needs retries — if the embedding model OOMs on chunk 400 of 500, I want chunk 400 retried, not the whole job restarted.

**Why RabbitMQ over Redis Streams:** Redis Streams can do queuing, and since Redis is already in the stack it's the "one less component" choice. I chose RabbitMQ for three capabilities I actually need:

1. **Dead-letter exchanges.** A document that fails ingestion three times routes to a DLQ where I can inspect it, rather than silently vanishing or blocking the queue. Building this on Streams means writing it myself.
2. **Per-message acknowledgement with prefetch control.** A worker crashing mid-embed returns the message to the queue. `prefetch_count=1` prevents one worker from claiming 50 documents and then dying with all of them.
3. **Topic routing.** `ingest.pdf`, `ingest.code`, `memory.extract` route to different worker pools with different resource profiles — the embedding workers want GPU affinity, the extraction workers don't.

**The honest counter-argument to have ready:** for a single-user system, Redis Streams would genuinely be sufficient, and "reduce component count" is a legitimate engineering value. I chose RabbitMQ because durable messaging with proper failure semantics is the correct tool for work I cannot afford to silently lose, and I'd rather demonstrate that I know *why* AMQP semantics exist than save one container. I'll say that plainly rather than pretend Redis couldn't do it.

**Why not Kafka:** Kafka is a distributed commit log for high-throughput event streaming with replay. I have neither the throughput nor the replay requirement. Using it here would be the clearest possible signal that I pick tools by reputation rather than fit.

**Why arq over Celery:** Celery is the default and it's battle-tested, but it's fundamentally sync-first — running async tasks in Celery means fighting the framework. `arq` is asyncio-native, so my worker code shares the same async DB session and HTTP client patterns as my gateway. One mental model across the codebase.

**Why not `BackgroundTasks`:** FastAPI's `BackgroundTasks` runs in the same process as the web server. A long embed job would compete with request handling for the event loop, and a server restart loses the work entirely. It's fine for fire-and-forget logging, which is all I use it for.
