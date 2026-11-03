import pytest
from redis.asyncio import Redis
from sqlalchemy import text

from pka.core.config import get_settings
from pka.db.session import make_engine, make_sessionmaker
from pka.ingest.embed import Embedder, HashEncoder

pytestmark = pytest.mark.integration

TABLES = "citations, messages, conversations, memories, chunks, documents, query_logs, eval_results, eval_runs, ingest_jobs"


@pytest.fixture(scope="session")
async def engine(migrated_db):
    engine = make_engine(migrated_db)
    yield engine
    await engine.dispose()


@pytest.fixture(scope="session")
def sessionmaker(engine):
    return make_sessionmaker(engine)


@pytest.fixture
async def redis():
    client = Redis.from_url(get_settings().redis_url)
    await client.flushdb()
    yield client
    await client.aclose()


@pytest.fixture(autouse=True)
async def clean_db(engine):
    async with engine.begin() as conn:
        await conn.execute(text(f"TRUNCATE {TABLES} RESTART IDENTITY CASCADE"))
    yield


@pytest.fixture
def embedder(redis):
    return Embedder(HashEncoder(1024), redis, ttl_s=60)
