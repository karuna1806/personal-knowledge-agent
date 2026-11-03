"""Redis-backed session history and query cache (§3.6). Invariant: Redis holds nothing whose
loss matters. Session history falls back to Postgres; the cache is just a cache. If Redis is
flushed or down, the system gets slower, not wrong."""

import hashlib
import json
import logging
import uuid

from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pka.core.metrics import CACHE_REQUESTS
from pka.core.versions import CORPUS_VERSION_KEY, MEMORY_VERSION_KEY
from pka.db.models import Message as MessageRow
from pka.llm.client import Message

log = logging.getLogger(__name__)


class SessionStore:
    def __init__(self, redis: Redis | None, sessionmaker: async_sessionmaker[AsyncSession], *, turns: int, ttl_s: int):
        self._redis = redis
        self._sessionmaker = sessionmaker
        self._max_messages = turns * 2
        self._ttl_s = ttl_s

    @staticmethod
    def _key(conversation_id: uuid.UUID) -> str:
        return f"sess:{conversation_id}"

    async def get(self, conversation_id: uuid.UUID) -> list[Message]:
        if self._redis is not None:
            try:
                raw = await self._redis.lrange(self._key(conversation_id), 0, -1)
                if raw:
                    CACHE_REQUESTS.labels("sess", "hit").inc()
                    return [json.loads(item) for item in raw]
            except Exception as exc:
                log.warning("session read failed, falling back to Postgres: %s", exc)
        CACHE_REQUESTS.labels("sess", "miss").inc()
        async with self._sessionmaker() as session:
            rows = (
                await session.scalars(
                    select(MessageRow)
                    .where(MessageRow.conversation_id == conversation_id, MessageRow.role.in_(("user", "assistant")))
                    .order_by(MessageRow.id.desc())
                    .limit(self._max_messages)
                )
            ).all()
        history = [{"role": r.role, "content": r.content} for r in reversed(rows)]
        if history:
            await self._write(conversation_id, history)
        return history

    async def append(self, conversation_id: uuid.UUID, user: str, assistant: str) -> None:
        if self._redis is None:
            return
        key = self._key(conversation_id)
        try:
            async with self._redis.pipeline(transaction=True) as pipe:
                pipe.rpush(key, json.dumps({"role": "user", "content": user}),
                           json.dumps({"role": "assistant", "content": assistant}))  # fmt: skip
                pipe.ltrim(key, -self._max_messages, -1)
                pipe.expire(key, self._ttl_s)
                await pipe.execute()
        except Exception as exc:
            log.warning("session write failed: %s", exc)

    async def _write(self, conversation_id: uuid.UUID, history: list[Message]) -> None:
        if self._redis is None:
            return
        key = self._key(conversation_id)
        try:
            async with self._redis.pipeline(transaction=True) as pipe:
                pipe.delete(key)
                pipe.rpush(key, *[json.dumps(m) for m in history])
                pipe.expire(key, self._ttl_s)
                await pipe.execute()
        except Exception as exc:
            log.warning("session backfill failed: %s", exc)


def normalize_query(query: str) -> str:
    return " ".join(query.lower().split()).rstrip("?!. ")


class QueryCache:
    """`q:{sha256(normalized_query + corpus version + config)}` → answer + sources, 1h TTL.
    Including the corpus version means any ingest/delete invalidates every cached answer."""

    def __init__(self, redis: Redis | None, ttl_s: int, config_fingerprint: str):
        self._redis = redis
        self._ttl_s = ttl_s
        self._fingerprint = config_fingerprint

    async def _key(self, query: str) -> str | None:
        if self._redis is None:
            return None
        corpus, memory = await self._redis.mget(CORPUS_VERSION_KEY, MEMORY_VERSION_KEY)
        material = "\x00".join(
            (normalize_query(query), (corpus or b"0").decode(), (memory or b"0").decode(), self._fingerprint)
        )
        return "q:" + hashlib.sha256(material.encode()).hexdigest()

    async def get(self, query: str) -> dict | None:
        try:
            key = await self._key(query)
            if key is None:
                return None
            raw = await self._redis.get(key)
        except Exception as exc:
            log.warning("query cache read failed: %s", exc)
            return None
        CACHE_REQUESTS.labels("q", "hit" if raw else "miss").inc()
        return json.loads(raw) if raw else None

    async def set(self, query: str, value: dict) -> None:
        try:
            key = await self._key(query)
            if key is not None:
                await self._redis.set(key, json.dumps(value), ex=self._ttl_s)
        except Exception as exc:
            log.warning("query cache write failed: %s", exc)
