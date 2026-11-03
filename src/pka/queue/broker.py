"""RabbitMQ topology and publisher (§3.5).

    pka (topic) ──ingest.#──► ingest.queue ─┐
                ──reindex.#─► reindex.queue ├─ reject ─► pka.dlx (topic) ──#──► pka.dlq
                ──memory.#──► memory.queue ─┘
    pka.retry (topic) ──#──► pka.retry.queue  (TTL) ── expires ──► pka  (original routing key)

A failing message is re-published to pka.retry with attempt+1; after QUEUE_MAX_ATTEMPTS it is
rejected and dead-letters into pka.dlq for inspection. Nothing silently vanishes.
"""

import json
import logging
from dataclasses import dataclass

import aio_pika
from aio_pika import DeliveryMode, ExchangeType, Message
from aio_pika.abc import AbstractChannel, AbstractExchange, AbstractRobustConnection

from pka.core.logging import request_id_var

log = logging.getLogger(__name__)

EXCHANGE = "pka"
RETRY_EXCHANGE = "pka.retry"
DLX = "pka.dlx"
RETRY_QUEUE = "pka.retry.queue"
DLQ = "pka.dlq"


@dataclass(frozen=True)
class QueueSpec:
    name: str
    binding: str


QUEUES: dict[str, QueueSpec] = {
    "ingest": QueueSpec("ingest.queue", "ingest.#"),
    "reindex": QueueSpec("reindex.queue", "reindex.#"),
    "memory": QueueSpec("memory.queue", "memory.#"),
}

# Routing keys
INGEST_FILE = "ingest.file"
INGEST_URL = "ingest.url"
REINDEX_MODEL = "reindex.model"
MEMORY_EXTRACT = "memory.extract"

MAIN_QUEUE_ARGS = {"x-dead-letter-exchange": DLX}


async def declare_topology(channel: AbstractChannel, retry_delay_ms: int) -> AbstractExchange:
    exchange = await channel.declare_exchange(EXCHANGE, ExchangeType.TOPIC, durable=True)
    retry_exchange = await channel.declare_exchange(RETRY_EXCHANGE, ExchangeType.TOPIC, durable=True)
    dlx = await channel.declare_exchange(DLX, ExchangeType.TOPIC, durable=True)

    for spec in QUEUES.values():
        queue = await channel.declare_queue(spec.name, durable=True, arguments=MAIN_QUEUE_ARGS)
        await queue.bind(exchange, spec.binding)

    retry_queue = await channel.declare_queue(
        RETRY_QUEUE,
        durable=True,
        arguments={"x-message-ttl": retry_delay_ms, "x-dead-letter-exchange": EXCHANGE},
    )
    await retry_queue.bind(retry_exchange, "#")

    dlq = await channel.declare_queue(DLQ, durable=True)
    await dlq.bind(dlx, "#")
    return exchange


def build_message(payload: dict, *, attempt: int = 1, request_id: str | None = None) -> Message:
    return Message(
        json.dumps(payload).encode(),
        content_type="application/json",
        delivery_mode=DeliveryMode.PERSISTENT,
        headers={"x-attempt": attempt, "x-request-id": request_id or request_id_var.get()},
    )


class Broker:
    def __init__(self, url: str, retry_delay_ms: int = 5000):
        self._url = url
        self._retry_delay_ms = retry_delay_ms
        self.connection: AbstractRobustConnection | None = None
        self._channel: AbstractChannel | None = None
        self._exchange: AbstractExchange | None = None

    async def connect(self) -> None:
        self.connection = await aio_pika.connect_robust(self._url, timeout=10)
        self._channel = await self.connection.channel(publisher_confirms=True)
        self._exchange = await declare_topology(self._channel, self._retry_delay_ms)

    async def publish(self, routing_key: str, payload: dict, *, attempt: int = 1) -> None:
        if self._exchange is None:
            raise RuntimeError("broker not connected")
        await self._exchange.publish(build_message(payload, attempt=attempt), routing_key=routing_key)

    async def queue_depths(self) -> dict[str, int]:
        """Ready-message counts via passive declares (no side effects)."""
        if self.connection is None:
            raise RuntimeError("broker not connected")
        depths: dict[str, int] = {}
        async with self.connection.channel() as channel:
            for name in [s.name for s in QUEUES.values()] + [RETRY_QUEUE, DLQ]:
                queue = await channel.declare_queue(name, passive=True)
                depths[name] = queue.declaration_result.message_count
        return depths

    async def peek_dlq(self, limit: int = 20) -> list[dict]:
        """Read DLQ messages without removing them (get, then nack with requeue)."""
        if self.connection is None:
            raise RuntimeError("broker not connected")
        items: list[dict] = []
        async with self.connection.channel() as channel:
            queue = await channel.declare_queue(DLQ, passive=True)
            held = []
            for _ in range(limit):
                message = await queue.get(no_ack=False, fail=False)
                if message is None:
                    break
                held.append(message)
                deaths = (message.headers or {}).get("x-death") or []
                items.append(
                    {
                        "routing_key": message.routing_key,
                        "payload": _safe_json(message.body),
                        "attempt": (message.headers or {}).get("x-attempt"),
                        "error": (message.headers or {}).get("x-error"),
                        "death": [
                            {"queue": d.get("queue"), "reason": d.get("reason"), "count": d.get("count")}
                            for d in deaths
                            if isinstance(d, dict)
                        ],
                    }
                )
            for message in held:
                await message.nack(requeue=True)
        return items

    async def replay_dlq(self, limit: int = 100) -> int:
        """Move DLQ messages back onto the main exchange with a fresh attempt counter."""
        if self.connection is None or self._exchange is None:
            raise RuntimeError("broker not connected")
        moved = 0
        async with self.connection.channel() as channel:
            queue = await channel.declare_queue(DLQ, passive=True)
            # Only what is dead *now*: a replayed message that fails again lands back in the
            # DLQ during this loop and must not be picked up a second time.
            snapshot = queue.declaration_result.message_count
            for _ in range(min(limit, snapshot)):
                message = await queue.get(no_ack=False, fail=False)
                if message is None:
                    break
                await self._exchange.publish(
                    build_message(_safe_json(message.body) or {}, attempt=1),
                    routing_key=message.routing_key or "",
                )
                await message.ack()
                moved += 1
        return moved

    async def close(self) -> None:
        if self.connection is not None:
            await self.connection.close()
            self.connection = None


def _safe_json(body: bytes) -> dict | None:
    try:
        value = json.loads(body)
        return value if isinstance(value, dict) else {"value": value}
    except (ValueError, UnicodeDecodeError):
        return None
