"""Monotonic version counters in Redis. Cached answers are keyed on them, so anything that can
change an answer — new/deleted documents, new/superseded/retracted memories — invalidates the
query cache by bumping a counter instead of hunting down keys. Losing Redis resets them to 0,
which is safe: the cache is lost along with them."""

import logging

from redis.asyncio import Redis

log = logging.getLogger(__name__)

CORPUS_VERSION_KEY = "corpus:version"
MEMORY_VERSION_KEY = "memory:version"


async def bump(redis: Redis | None, key: str) -> None:
    if redis is None:
        return
    try:
        await redis.incr(key)
    except Exception as exc:
        log.warning("could not bump %s: %s", key, exc)
