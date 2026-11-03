"""Queue failure semantics against a real RabbitMQ (vhost pka_test)."""

import asyncio
import uuid

import pytest
from sqlalchemy import insert

from pka.core.config import get_settings
from pka.db.models import IngestJob
from pka.llm.client import LLMClient
from pka.queue import broker as b
from pka.workers import tasks
from pka.workers.context import PermanentError, WorkerContext
from pka.workers.runner import Worker

pytestmark = pytest.mark.integration


@pytest.fixture
async def broker(rabbit_vhost):
    br = b.Broker(rabbit_vhost, retry_delay_ms=200)
    await br.connect()
    async with br.connection.channel() as ch:
        for name in [s.name for s in b.QUEUES.values()] + [b.RETRY_QUEUE, b.DLQ]:
            await (await ch.declare_queue(name, passive=True)).purge()
    yield br
    await br.close()


@pytest.fixture
async def worker_ctx(sessionmaker, redis, embedder):
    settings = get_settings().model_copy(update={"queue_max_attempts": 3, "queue_retry_delay_ms": 200})
    llm = LLMClient("http://127.0.0.1:9/v1", "none")
    yield WorkerContext(settings, sessionmaker, redis, embedder, llm)
    await llm.close()


async def _job(sessionmaker) -> str:
    async with sessionmaker() as s, s.begin():
        return str(await s.scalar(insert(IngestJob).values(kind="file", target="t").returning(IngestJob.id)))


async def _wait(predicate, timeout=10.0):  # noqa: ASYNC109
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if await predicate():
            return True
        await asyncio.sleep(0.1)
    return False


async def _job_status(sessionmaker, job_id):
    async with sessionmaker() as s:
        return await s.get(IngestJob, uuid.UUID(job_id))


async def test_transient_failure_retries_then_dead_letters(broker, worker_ctx, sessionmaker, monkeypatch):
    attempts = []

    async def flaky(ctx, payload):
        attempts.append(payload["_attempt"])
        raise ConnectionError("embedding server went away")

    monkeypatch.setitem(tasks.HANDLERS, b.INGEST_FILE, flaky)
    worker = Worker(worker_ctx, ["ingest"])
    await worker.start()
    try:
        job_id = await _job(sessionmaker)
        await broker.publish(b.INGEST_FILE, {"path": "/x", "job_id": job_id})

        async def dead():
            return (await broker.queue_depths())[b.DLQ] == 1

        assert await _wait(dead), f"never dead-lettered; attempts={attempts}"
        assert attempts == [1, 2, 3]
        job = await _job_status(sessionmaker, job_id)
        assert job.status == "dead" and "ConnectionError" in job.error and job.attempts == 3
        peeked = await broker.peek_dlq()
        assert peeked[0]["routing_key"] == b.INGEST_FILE and peeked[0]["payload"]["job_id"] == job_id
        assert peeked[0]["error"].startswith("ConnectionError")
        assert (await broker.queue_depths())[b.DLQ] == 1  # peeking does not consume
    finally:
        await worker.stop(timeout_s=5)


async def test_permanent_failure_is_recorded_not_retried(broker, worker_ctx, sessionmaker, monkeypatch):
    calls = []

    async def unsupported(ctx, payload):
        calls.append(1)
        raise PermanentError("unsupported file type: x.jpg")

    monkeypatch.setitem(tasks.HANDLERS, b.INGEST_FILE, unsupported)
    worker = Worker(worker_ctx, ["ingest"])
    await worker.start()
    try:
        job_id = await _job(sessionmaker)
        await broker.publish(b.INGEST_FILE, {"path": "/x.jpg", "job_id": job_id})

        async def failed():
            return (await _job_status(sessionmaker, job_id)).status == "failed"

        assert await _wait(failed)
        await asyncio.sleep(0.6)  # longer than the retry delay: make sure nothing comes back
        assert calls == [1]
        depths = await broker.queue_depths()
        assert depths[b.DLQ] == 0 and depths["ingest.queue"] == 0
    finally:
        await worker.stop(timeout_s=5)


async def test_real_ingest_handler_end_to_end(broker, worker_ctx, sessionmaker, tmp_path):
    f = tmp_path / "n.md"
    f.write_text("# Title\n\nSome body text about queues.\n")
    worker = Worker(worker_ctx, ["ingest"])
    await worker.start()
    try:
        job_id = await _job(sessionmaker)
        await broker.publish(b.INGEST_FILE, {"path": str(f), "source_uri": "n.md", "job_id": job_id})

        async def done():
            return (await _job_status(sessionmaker, job_id)).status == "done"

        assert await _wait(done)
        job = await _job_status(sessionmaker, job_id)
        assert job.chunks == 1 and job.document_id is not None

        # Same file again → idempotent skip
        job2 = await _job(sessionmaker)
        await broker.publish(b.INGEST_FILE, {"path": str(f), "source_uri": "n.md", "job_id": job2})

        async def skipped():
            return (await _job_status(sessionmaker, job2)).status == "skipped"

        assert await _wait(skipped)
    finally:
        await worker.stop(timeout_s=5)


async def test_malformed_and_unroutable_messages_dead_letter(broker, worker_ctx):
    import aio_pika

    worker = Worker(worker_ctx, ["ingest"])
    await worker.start()
    try:
        async with broker.connection.channel() as ch:
            ex = await ch.get_exchange(b.EXCHANGE)
            await ex.publish(aio_pika.Message(b"not json"), routing_key=b.INGEST_FILE)
            await ex.publish(aio_pika.Message(b"{}"), routing_key="ingest.unknown")

        async def both_dead():
            return (await broker.queue_depths())[b.DLQ] == 2

        assert await _wait(both_dead)
        assert await broker.replay_dlq() == 2
    finally:
        await worker.stop(timeout_s=5)
