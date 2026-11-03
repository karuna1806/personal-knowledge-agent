"""Task handlers, keyed by routing key. Each must be idempotent: a message can be delivered
more than once (worker crash after work, before ack)."""

import logging
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path

from sqlalchemy import update

from pka.core.metrics import INGEST_JOBS
from pka.db.models import IngestJob
from pka.ingest.loaders import UnsupportedSource, load_path, load_url
from pka.ingest.pipeline import ingest_document, reindex_stale
from pka.llm.priority import wait_for_idle
from pka.memory.extract import extract_and_store
from pka.queue.broker import INGEST_FILE, INGEST_URL, MEMORY_EXTRACT, REINDEX_MODEL
from pka.workers.context import PermanentError, WorkerContext

log = logging.getLogger(__name__)


async def set_job(ctx: WorkerContext, job_id: str | None, **values) -> None:
    if not job_id:
        return
    async with ctx.sessionmaker() as session, session.begin():
        await session.execute(update(IngestJob).where(IngestJob.id == uuid.UUID(job_id)).values(**values))


async def _ingest(ctx: WorkerContext, payload: dict, load) -> None:
    job_id = payload.get("job_id")
    attempt = payload.get("_attempt", 1)
    await set_job(ctx, job_id, status="running", attempts=attempt, error=None)
    try:
        doc = await load()
    except (UnsupportedSource, FileNotFoundError, IsADirectoryError) as exc:
        raise PermanentError(str(exc)) from exc
    try:
        outcome = await ingest_document(ctx.sessionmaker, ctx.embedder, doc, ctx.settings, ctx.redis)
    except ValueError as exc:  # no chunkable content
        raise PermanentError(str(exc)) from exc
    status = "skipped" if outcome.skipped else "done"
    INGEST_JOBS.labels(status).inc()
    await set_job(ctx, job_id, status=status, document_id=outcome.document_id, chunks=outcome.chunks)


async def handle_ingest_file(ctx: WorkerContext, payload: dict) -> None:
    path = Path(payload["path"])

    async def load():
        return load_path(path, source_uri=payload.get("source_uri"))

    await _ingest(ctx, payload, load)


async def handle_ingest_url(ctx: WorkerContext, payload: dict) -> None:
    url = payload["url"]

    async def load():
        return await load_url(url)

    await _ingest(ctx, payload, load)


async def handle_reindex(ctx: WorkerContext, payload: dict) -> None:
    job_id = payload.get("job_id")
    await set_job(ctx, job_id, status="running", attempts=payload.get("_attempt", 1))
    total = await reindex_stale(ctx.sessionmaker, ctx.embedder, ctx.redis)
    await set_job(ctx, job_id, status="done", chunks=total)


async def handle_memory_extract(ctx: WorkerContext, payload: dict) -> None:
    waited = await wait_for_idle(ctx.redis)
    if waited > 5:
        log.info("waited %.1fs for interactive queries to finish", waited)
    await extract_and_store(
        ctx.sessionmaker,
        ctx.llm,
        ctx.embedder,
        user_message_id=int(payload["user_message_id"]),
        assistant_message_id=int(payload["assistant_message_id"]),
        redis=ctx.redis,
    )


HANDLERS: dict[str, Callable[[WorkerContext, dict], Awaitable[None]]] = {
    INGEST_FILE: handle_ingest_file,
    INGEST_URL: handle_ingest_url,
    REINDEX_MODEL: handle_reindex,
    MEMORY_EXTRACT: handle_memory_extract,
}
