"""Interactive traffic first. A single local inference server serves both user queries and
background work (memory extraction). Background LLM calls wait while any query is in flight,
so the queue protects the user's latency instead of competing with it.

Each in-flight query is a member of a sorted set scored by its expiry time, so a crashed API
process can't leave the gate stuck closed — stale entries simply age out.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from redis.asyncio import Redis

log = logging.getLogger(__name__)

KEY = "llm:interactive"
ENTRY_TTL_S = 90


@asynccontextmanager
async def interactive(redis: Redis | None, request_id: str):
    if redis is None:
        yield
        return
    try:
        await redis.zadd(KEY, {request_id: time.time() + ENTRY_TTL_S})
    except Exception as exc:
        log.debug("priority gate unavailable: %s", exc)
    try:
        yield
    finally:
        try:
            await redis.zrem(KEY, request_id)
        except BaseException:  # cancelled mid-cleanup: the entry expires on its own
            pass


async def wait_for_idle(
    redis: Redis | None, *, quiet_s: float = 3.0, max_wait_s: float = 60.0, poll_s: float = 0.5
) -> float:
    """Block until no interactive query has run for `quiet_s` (a user who just got an answer
    often asks a follow-up within seconds), or until max_wait_s. Returns seconds waited."""
    if redis is None:
        return 0.0
    started = time.monotonic()
    idle_since: float | None = None
    while time.monotonic() - started < max_wait_s:
        try:
            await redis.zremrangebyscore(KEY, "-inf", time.time())
            busy = bool(await redis.zcard(KEY))
        except Exception:
            break  # Redis down: don't block background work on a missing optimisation
        now = time.monotonic()
        if busy:
            idle_since = None
        else:
            idle_since = idle_since or now
            if now - idle_since >= quiet_s:
                break
        await asyncio.sleep(poll_s)
    return time.monotonic() - started
