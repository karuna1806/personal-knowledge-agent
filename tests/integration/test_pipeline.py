import pytest
from sqlalchemy import func, select, update

from pka.agent.retriever import Retriever, SearchFilters
from pka.core.config import get_settings
from pka.core.versions import CORPUS_VERSION_KEY
from pka.db.models import Chunk, Document
from pka.ingest.loaders import LoadedDocument
from pka.ingest.pipeline import ingest_document, reindex_stale

pytestmark = pytest.mark.integration

NOTE = """# Queue design

## Why RabbitMQ

Dead-letter exchanges route documents that fail ingestion three times to a DLQ.

## Index choice

HNSW gives a better recall-latency tradeoff than IVFFlat and needs no rebuilds.
"""

CODE = '''def connect_broker(url):
    """Open the AMQP connection."""
    return url


class HNSWIndexError(Exception):
    """Raised when the vector index cannot be built."""
'''


def note(content=NOTE, uri="notes/queue.md"):
    return LoadedDocument("note", uri, "Queue design", content)


async def test_ingest_is_transactional_and_idempotent(sessionmaker, embedder, redis):
    settings = get_settings()
    first = await ingest_document(sessionmaker, embedder, note(), settings, redis)
    assert not first.skipped and first.chunks == 2
    again = await ingest_document(sessionmaker, embedder, note(), settings, redis)
    assert again.skipped and again.document_id == first.document_id
    async with sessionmaker() as s:
        assert await s.scalar(select(func.count()).select_from(Document)) == 1
        assert await s.scalar(select(func.count()).select_from(Chunk)) == 2
        rows = (await s.execute(select(Chunk.embed_model, Chunk.metadata_, Chunk.tsv))).all()
    assert {r.embed_model for r in rows} == {"hash"}
    assert all(r.metadata_["heading_path"][0] == "Queue design" for r in rows)
    assert all(r.tsv for r in rows)  # generated column populated
    assert int(await redis.get(CORPUS_VERSION_KEY)) == 1  # skip doesn't bump


async def test_changed_file_replaces_previous_version(sessionmaker, embedder, redis):
    settings = get_settings()
    v1 = await ingest_document(sessionmaker, embedder, note(), settings, redis)
    v2 = await ingest_document(sessionmaker, embedder, note(NOTE + "\n## New\n\nAdded section.\n"), settings, redis)
    assert v2.replaced == 1 and v2.document_id != v1.document_id
    async with sessionmaker() as s:
        assert (await s.scalars(select(Document.id))).all() == [v2.document_id]
        assert await s.scalar(select(func.count()).select_from(Chunk)) == 3


async def test_hybrid_search_finds_exact_identifier(sessionmaker, embedder, redis):
    settings = get_settings()
    await ingest_document(sessionmaker, embedder, note(), settings, redis)
    code = LoadedDocument("code", "src/broker.py", "broker.py", CODE, {"language": "python", "path": "src/broker.py"})
    await ingest_document(sessionmaker, embedder, code, settings, redis)
    retriever = Retriever(sessionmaker, embedder, settings)

    lexical = await retriever.search("HNSWIndexError", mode="lexical")
    assert lexical.chunks[0].metadata["symbol"] == "HNSWIndexError"
    hybrid = await retriever.search("what raises HNSWIndexError", mode="hybrid")
    assert hybrid.chunks[0].metadata["symbol"] == "HNSWIndexError"
    assert hybrid.chunks[0].lexical_rank == 1
    # OR semantics: a natural question with words absent from the doc still matches.
    loose = await retriever.search("why choose dead-letter exchanges zebra", mode="lexical")
    assert loose.chunks and "Dead-letter" in loose.chunks[0].content


async def test_filters(sessionmaker, embedder, redis):
    settings = get_settings()
    await ingest_document(sessionmaker, embedder, note(), settings, redis)
    code = LoadedDocument("code", "src/broker.py", "broker.py", CODE, {"language": "python", "path": "src/broker.py"})
    await ingest_document(sessionmaker, embedder, code, settings, redis)
    retriever = Retriever(sessionmaker, embedder, settings)
    only_code = await retriever.search("index", filters=SearchFilters(source_types=["code"]))
    assert only_code.chunks and {c.source_type for c in only_code.chunks} == {"code"}
    by_prefix = await retriever.search("index", filters=SearchFilters(source_prefix="notes/"))
    assert by_prefix.chunks and all(c.source_uri.startswith("notes/") for c in by_prefix.chunks)
    by_lang = await retriever.search("index", filters=SearchFilters(language="python"))
    assert by_lang.chunks and all(c.metadata["language"] == "python" for c in by_lang.chunks)
    # LIKE wildcards in a prefix are escaped
    none = await retriever.search("index", filters=SearchFilters(source_prefix="%"))
    assert none.chunks == []


async def test_reindex_only_touches_stale_rows(sessionmaker, embedder, redis):
    settings = get_settings()
    await ingest_document(sessionmaker, embedder, note(), settings, redis)
    async with sessionmaker() as s, s.begin():
        first_id = await s.scalar(select(func.min(Chunk.id)))
        await s.execute(update(Chunk).where(Chunk.id == first_id).values(embed_model="old-model"))
    assert await reindex_stale(sessionmaker, embedder, redis) == 1
    async with sessionmaker() as s:
        assert set((await s.scalars(select(Chunk.embed_model))).all()) == {"hash"}
    assert await reindex_stale(sessionmaker, embedder, redis) == 0


async def test_embedding_cache_survives_redis_loss(sessionmaker, embedder, redis):
    vec1 = await embedder.embed(["hello world"])
    assert await redis.dbsize() == 1
    await redis.flushdb()
    vec2 = await embedder.embed(["hello world"])
    assert (vec1 == vec2).all()
    q = await embedder.embed(["hello world"], is_query=True)
    assert await redis.dbsize() == 2  # query and document embeddings are cached separately
    assert q.shape == (1, 1024)
