"""Prometheus metrics. Queue depth and DLQ depth are the two that matter most (§8)."""

from prometheus_client import Counter, Gauge, Histogram

QUERY_LATENCY = Histogram(
    "pka_query_latency_seconds",
    "End-to-end query latency",
    ["route"],
    buckets=(0.25, 0.5, 1, 2, 4, 8, 16, 32, 64),
)
TTFT = Histogram(
    "pka_ttft_seconds",
    "Time to first token",
    ["route"],
    buckets=(0.1, 0.25, 0.5, 1, 1.5, 2, 3, 5, 10),
)
TOKENS_PER_SECOND = Histogram(
    "pka_tokens_per_second", "Completion tokens per second", buckets=(1, 5, 10, 20, 40, 80, 160)
)
STAGE_LATENCY = Histogram(
    "pka_stage_latency_seconds",
    "Latency of pipeline stages",
    ["stage"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)
CACHE_REQUESTS = Counter("pka_cache_requests_total", "Cache lookups", ["namespace", "result"])
QUEUE_DEPTH = Gauge("pka_queue_depth", "Messages ready in a queue", ["queue"])
DLQ_DEPTH = Gauge("pka_dlq_depth", "Messages in the dead-letter queue")
INGEST_JOBS = Counter("pka_ingest_jobs_total", "Ingest jobs processed", ["status"])
TASKS = Counter("pka_worker_tasks_total", "Worker tasks", ["routing_key", "status"])
TASK_LATENCY = Histogram(
    "pka_worker_task_seconds",
    "Worker task duration",
    ["routing_key"],
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300),
)
TOOL_CALLS = Counter("pka_tool_calls_total", "MCP tool calls", ["tool", "status"])
ACCELERATOR_MEMORY = Gauge("pka_accelerator_memory_bytes", "GPU/MPS memory allocated by torch")


def update_accelerator_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            ACCELERATOR_MEMORY.set(torch.cuda.memory_allocated())
        elif torch.backends.mps.is_available():
            ACCELERATOR_MEMORY.set(torch.mps.current_allocated_memory())
    except Exception:  # metrics must never break a request
        pass
