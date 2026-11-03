"""chunk → embed → store. The slow part (embedding) happens outside the transaction; the
write — document row, all chunk rows, removal of the previous version — is one transaction,
so a document is either fully searchable or not present at all (§3.3)."""

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from redis.asyncio import Redis
from sqlalchemy import delete, insert, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pka.core.config import Settings
from pka.core.telemetry import tracer
from pka.core.versions import CORPUS_VERSION_KEY, bump
from pka.db.models import Chunk, Document, Memory
from pka.ingest.chunkers import chunk_document, embedding_text
from pka.ingest.embed import Embedder
from pka.ingest.loaders import LoadedDocument
from pka.ingest.tokens import get_token_counter

log = logging.getLogger(__name__)

@dataclass
class IngestOutcome:
    document_id: int | None
    chunks: int
    skipped: bool
    replaced: int = 0


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def bump_corpus_version(redis: Redis | None) -> None:
    """Query-cache keys include the corpus version, so new content invalidates them."""
    await bump(redis, CORPUS_VERSION_KEY)


async def ingest_document(
    sessionmaker: async_sessionmaker[AsyncSession],
    embedder: Embedder,
    doc: LoadedDocument,
    settings: Settings,
    redis: Redis | None = None,
) -> IngestOutcome:
    digest = content_hash(doc.content)
    async with sessionmaker() as session:
        existing = await session.scalar(
            select(Document.id).where(Document.source_uri == doc.source_uri, Document.content_hash == digest)
        )
    if existing is not None:
        log.info("unchanged, skipping", extra={"source_uri": doc.source_uri, "document_id": existing})
        return IngestOutcome(existing, 0, skipped=True)

    with tracer.start_as_current_span("ingest.chunk"):
        count = get_token_counter(settings.embed_model)
        drafts = await asyncio.to_thread(
            chunk_document,
            doc,
            count,
            strategy=settings.chunker,
            max_tokens=settings.chunk_max_tokens,
            fixed_chars=settings.chunk_fixed_chars,
            fixed_overlap=settings.chunk_fixed_overlap,
        )
    if not drafts:
        raise ValueError(f"no chunkable content in {doc.source_uri}")

    vectors = await embedder.embed([embedding_text(d.content, d.metadata, doc.title) for d in drafts])

    with tracer.start_as_current_span("ingest.store"):
        async with sessionmaker() as session, session.begin():
            now = datetime.now(UTC)
            document_id = await session.scalar(
                pg_insert(Document)
                .values(
                    source_type=doc.source_type,
                    source_uri=doc.source_uri,
                    title=doc.title,
                    content=doc.content,
                    content_hash=digest,
                    metadata_=doc.metadata,
                    ingested_at=now,
                )
                .on_conflict_do_nothing(index_elements=["source_uri", "content_hash"])
                .returning(Document.id)
            )
            if document_id is None:  # a concurrent worker ingested the identical content
                return IngestOutcome(None, 0, skipped=True)
            await session.execute(
                insert(Chunk),
                [
                    {
                        "document_id": document_id,
                        "chunk_index": i,
                        "content": d.content,
                        "token_count": d.token_count,
                        "embedding": vectors[i],
                        "embed_model": embedder.model_name,
                        "metadata_": d.metadata,
                    }
                    for i, d in enumerate(drafts)
                ],
            )
            # A changed file replaces its previous version; cascades remove the old chunks.
            replaced = await session.execute(
                delete(Document).where(Document.source_uri == doc.source_uri, Document.id != document_id)
            )
    await bump_corpus_version(redis)
    log.info(
        "ingested",
        extra={"source_uri": doc.source_uri, "document_id": document_id, "chunks": len(drafts),
               "replaced": replaced.rowcount},  # fmt: skip
    )
    return IngestOutcome(document_id, len(drafts), skipped=False, replaced=replaced.rowcount or 0)


async def reindex_stale(
    sessionmaker: async_sessionmaker[AsyncSession],
    embedder: Embedder,
    redis: Redis | None = None,
    batch_size: int = 64,
) -> int:
    """Re-embed chunks whose embed_model differs from the current model (§4: `embed_model`
    exists so a model switch reindexes only stale rows). Same-dimension models only — a new
    dimension needs a schema migration."""
    total = 0
    while True:
        async with sessionmaker() as session, session.begin():
            rows = (
                await session.execute(
                    select(Chunk.id, Chunk.content, Chunk.metadata_, Document.title)
                    .join(Document, Document.id == Chunk.document_id)
                    .where(Chunk.embed_model != embedder.model_name)
                    .order_by(Chunk.id)
                    .limit(batch_size)
                    .with_for_update(of=Chunk, skip_locked=True)
                )
            ).all()
            if not rows:
                break
            vectors = await embedder.embed([embedding_text(r.content, r.metadata_, r.title) for r in rows])
            for row, vector in zip(rows, vectors, strict=True):
                await session.execute(
                    update(Chunk)
                    .where(Chunk.id == row.id)
                    .values(embedding=vector, embed_model=embedder.model_name)
                )
            total += len(rows)
    # Memories have no embed_model column; re-embed all active ones when chunks were stale.
    if total:
        async with sessionmaker() as session, session.begin():
            memories = (
                await session.execute(
                    select(Memory.id, Memory.subject, Memory.predicate, Memory.object)
                )
            ).all()
            if memories:
                from pka.memory.resolve import memory_text

                vectors = await embedder.embed([memory_text(m.subject, m.predicate, m.object) for m in memories])
                for m, vector in zip(memories, vectors, strict=True):
                    await session.execute(update(Memory).where(Memory.id == m.id).values(embedding=vector))
        await bump_corpus_version(redis)
    log.info("reindex complete", extra={"chunks": total})
    return total
