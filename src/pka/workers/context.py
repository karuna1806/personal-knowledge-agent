from dataclasses import dataclass

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pka.core.config import Settings
from pka.ingest.embed import Embedder
from pka.llm.client import LLMClient


@dataclass
class WorkerContext:
    settings: Settings
    sessionmaker: async_sessionmaker[AsyncSession]
    redis: Redis | None
    embedder: Embedder
    llm: LLMClient


class PermanentError(Exception):
    """Retrying will not help (unsupported file, missing file, empty document)."""
