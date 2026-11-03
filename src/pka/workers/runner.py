"""Consumer loop. One channel per queue with prefetch_count=1, so a worker never holds more
than one unacknowledged message per queue — if it dies, at most that one message is
redelivered (§3.5). Acks happen only after the handler succeeds."""

import asyncio
import json
import logging
import signal
import time

import aio_pika
from aio_pika.abc import AbstractIncomingMessage
from prometheus_client import start_http_server
from redis.asyncio import Redis

from pka.core.config import Settings
from pka.core.logging import configure_logging, new_request_id, request_id_var
from pka.core.metrics import TASK_LATENCY, TASKS, update_accelerator_memory
from pka.core.telemetry import configure_tracing, tracer
from pka.db.session import make_engine, make_sessionmaker
from pka.ingest.embed import Embedder, make_encoder
from pka.llm.client import LLMClient
from pka.queue.broker import (
    MAIN_QUEUE_ARGS,
    QUEUES,
    RETRY_EXCHANGE,
    build_message,
    declare_topology,
)
from pka.workers.context import PermanentError, WorkerContext
from pka.workers.tasks import HANDLERS, set_job

log = logging.getLogger("pka.worker")


class Worker:
    def __init__(self, ctx: WorkerContext, queue_names: list[str]):
        unknown = [q for q in queue_names if q not in QUEUES]
        if unknown:
            raise ValueError(f"unknown queues {unknown}; choose from {sorted(QUEUES)}")
        self.ctx = ctx
        self.queue_names = queue_names
        self._connection: aio_pika.abc.AbstractRobustConnection | None = None
        self._retry_exchange = None
        self._consumers: list[tuple] = []
        self._in_flight = 0
        self._idle = asyncio.Event()
        self._idle.set()

    async def start(self) -> None:
        settings = self.ctx.settings
        self._connection = await aio_pika.connect_robust(settings.amqp_url, timeout=10)
        setup = await self._connection.channel()
        await declare_topology(setup, settings.queue_retry_delay_ms)
        await setup.close()
        publish_channel = await self._connection.channel(publisher_confirms=True)
        self._retry_exchange = await publish_channel.get_exchange(RETRY_EXCHANGE)
        for name in self.queue_names:
            channel = await self._connection.channel()
            await channel.set_qos(prefetch_count=1)
            queue = await channel.declare_queue(QUEUES[name].name, durable=True, arguments=MAIN_QUEUE_ARGS)
            tag = await queue.consume(self.on_message)
            self._consumers.append((queue, tag))
            log.info("consuming", extra={"queue": QUEUES[name].name})

    async def on_message(self, message: AbstractIncomingMessage) -> None:
        headers = message.headers or {}
        attempt = int(headers.get("x-attempt") or 1)
        request_id_var.set(str(headers.get("x-request-id") or new_request_id()))
        routing_key = message.routing_key or ""
        self._in_flight += 1
        self._idle.clear()
        started = time.perf_counter()
        try:
            await self._process(message, routing_key, attempt)
        finally:
            TASK_LATENCY.labels(routing_key).observe(time.perf_counter() - started)
            update_accelerator_memory()
            self._in_flight -= 1
            if self._in_flight == 0:
                self._idle.set()

    async def _process(self, message: AbstractIncomingMessage, routing_key: str, attempt: int) -> None:
        handler = HANDLERS.get(routing_key)
        try:
            payload = json.loads(message.body)
            if not isinstance(payload, dict):
                raise ValueError("payload must be a JSON object")
        except ValueError as exc:
            log.error("malformed message, dead-lettering", extra={"routing_key": routing_key, "error": str(exc)})
            TASKS.labels(routing_key, "dead").inc()
            await message.reject(requeue=False)
            return
        if handler is None:
            log.error("no handler, dead-lettering", extra={"routing_key": routing_key})
            TASKS.labels(routing_key, "dead").inc()
            await message.reject(requeue=False)
            return

        payload["_attempt"] = attempt
        job_id = payload.get("job_id")
        try:
            with tracer.start_as_current_span(f"task {routing_key}") as span:
                span.set_attribute("task.attempt", attempt)
                await handler(self.ctx, payload)
        except PermanentError as exc:
            # Retrying cannot help; record why and drop. (Transient failures go to the DLQ.)
            log.warning("permanent failure", extra={"routing_key": routing_key, "error": str(exc)})
            TASKS.labels(routing_key, "failed").inc()
            await self._mark(job_id, status="failed", error=str(exc)[:2000], attempts=attempt)
            await message.ack()
            return
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"[:2000]
            payload.pop("_attempt", None)
            if attempt < self.ctx.settings.queue_max_attempts:
                log.warning(
                    "task failed, scheduling retry",
                    extra={"routing_key": routing_key, "attempt": attempt, "error": error},
                    exc_info=True,
                )
                TASKS.labels(routing_key, "retried").inc()
                retry = build_message(payload, attempt=attempt + 1)
                retry.headers["x-error"] = error
                # Publish (confirmed) before ack: a crash in between duplicates, never loses.
                await self._retry_exchange.publish(retry, routing_key=routing_key)
                await self._mark(job_id, status="queued", error=error, attempts=attempt)
                await message.ack()
            else:
                log.error(
                    "task failed permanently, dead-lettering",
                    extra={"routing_key": routing_key, "attempt": attempt, "error": error},
                )
                TASKS.labels(routing_key, "dead").inc()
                await self._mark(job_id, status="dead", error=error, attempts=attempt)
                await message.reject(requeue=False)
            return
        TASKS.labels(routing_key, "ok").inc()
        await message.ack()

    async def _mark(self, job_id: str | None, **values) -> None:
        try:
            await set_job(self.ctx, job_id, **values)
        except Exception:
            log.exception("could not update job status")

    async def stop(self, timeout_s: float = 60.0) -> None:
        if self._connection is None:
            return
        # Stop taking new work, let the in-flight message finish, then disconnect. Anything
        # still unacked when the connection closes is redelivered to another worker.
        for queue, tag in self._consumers:
            try:
                await queue.cancel(tag)
            except Exception:
                log.warning("could not cancel consumer %s", tag)
        try:
            await asyncio.wait_for(self._idle.wait(), timeout=timeout_s)
        except TimeoutError:
            log.warning("in-flight task did not finish before shutdown; it will be redelivered")
        await self._connection.close()


async def build_context(settings: Settings) -> WorkerContext:
    engine = make_engine(settings.database_url)
    redis = Redis.from_url(settings.redis_url)
    encoder = await asyncio.to_thread(
        make_encoder,
        settings.embed_model,
        settings.embed_dim,
        device=settings.embed_device,
        query_prompt=settings.embed_query_prompt,
        batch_size=settings.embed_batch_size,
    )
    llm = LLMClient(
        settings.llm_base_url,
        settings.llm_model,
        reasoning_effort=settings.llm_reasoning_effort,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout_s=settings.llm_timeout_s,
    )
    return WorkerContext(
        settings=settings,
        sessionmaker=make_sessionmaker(engine),
        redis=redis,
        embedder=Embedder(encoder, redis, settings.embed_cache_ttl_s),
        llm=llm,
    )


async def run(settings: Settings, queue_names: list[str]) -> None:
    configure_logging(settings.log_level, settings.log_json)
    configure_tracing("pka-worker", settings.otel_exporter_otlp_endpoint)
    if settings.worker_metrics_port:
        start_http_server(settings.worker_metrics_port)
    ctx = await build_context(settings)
    worker = Worker(ctx, queue_names)
    await worker.start()
    log.info("worker ready", extra={"queues": queue_names, "embed_model": ctx.embedder.model_name})

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    await stop.wait()
    log.info("shutting down")
    await worker.stop()
    await ctx.llm.close()
    if ctx.redis is not None:
        await ctx.redis.aclose()
