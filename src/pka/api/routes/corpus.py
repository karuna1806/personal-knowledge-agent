"""Read/delete access to documents, chunks and conversations."""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import delete, func, select

from pka.api.context import AppContext
from pka.api.deps import get_ctx, require_api_key
from pka.api.schemas import ChunkOut, DocumentOut
from pka.db.models import Chunk, Citation, Conversation, Document, Message
from pka.ingest.pipeline import bump_corpus_version

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.get("/documents", response_model=list[DocumentOut], tags=["corpus"])
async def list_documents(
    ctx: AppContext = Depends(get_ctx),
    source_type: str | None = None,
    q: str | None = Query(None, description="substring match on source_uri or title"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> list[DocumentOut]:
    counts = select(Chunk.document_id, func.count().label("n")).group_by(Chunk.document_id).subquery()
    stmt = (
        select(Document, counts.c.n)
        .outerjoin(counts, counts.c.document_id == Document.id)
        .order_by(Document.ingested_at.desc().nullslast(), Document.id.desc())
        .limit(limit)
        .offset(offset)
    )
    if source_type:
        stmt = stmt.where(Document.source_type == source_type)
    if q:
        like = f"%{q}%"
        stmt = stmt.where(Document.source_uri.ilike(like) | Document.title.ilike(like))
    async with ctx.sessionmaker() as session:
        rows = (await session.execute(stmt)).all()
    return [
        DocumentOut(id=d.id, source_type=d.source_type, source_uri=d.source_uri, title=d.title,
                    content_hash=d.content_hash, metadata=d.metadata_ or {}, ingested_at=d.ingested_at,
                    chunk_count=n or 0)  # fmt: skip
        for d, n in rows
    ]


@router.get("/documents/{document_id}", tags=["corpus"])
async def get_document(document_id: int, ctx: AppContext = Depends(get_ctx)) -> dict:
    async with ctx.sessionmaker() as session:
        doc = await session.get(Document, document_id)
        if doc is None:
            raise HTTPException(404, "document not found")
        chunks = (
            await session.execute(
                select(Chunk.id, Chunk.chunk_index, Chunk.token_count, Chunk.metadata_)
                .where(Chunk.document_id == document_id)
                .order_by(Chunk.chunk_index)
            )
        ).all()
    return {
        "id": doc.id, "source_type": doc.source_type, "source_uri": doc.source_uri, "title": doc.title,
        "content": doc.content, "metadata": doc.metadata_, "ingested_at": doc.ingested_at,
        "chunks": [{"id": c.id, "chunk_index": c.chunk_index, "token_count": c.token_count,
                    "metadata": c.metadata_} for c in chunks],
    }  # fmt: skip


@router.delete("/documents/{document_id}", status_code=204, tags=["corpus"])
async def delete_document(document_id: int, ctx: AppContext = Depends(get_ctx)) -> None:
    async with ctx.sessionmaker() as session, session.begin():
        result = await session.execute(delete(Document).where(Document.id == document_id))
    if not result.rowcount:
        raise HTTPException(404, "document not found")
    await bump_corpus_version(ctx.redis)


@router.get("/chunks/{chunk_id}", response_model=ChunkOut, tags=["corpus"])
async def get_chunk(chunk_id: int, ctx: AppContext = Depends(get_ctx)) -> ChunkOut:
    async with ctx.sessionmaker() as session:
        row = (
            await session.execute(
                select(Chunk, Document.source_uri, Document.title, Document.source_type)
                .join(Document, Document.id == Chunk.document_id)
                .where(Chunk.id == chunk_id)
            )
        ).first()
    if row is None:
        raise HTTPException(404, "chunk not found")
    chunk, source_uri, title, source_type = row
    return ChunkOut(id=chunk.id, document_id=chunk.document_id, chunk_index=chunk.chunk_index,
                    content=chunk.content, token_count=chunk.token_count, embed_model=chunk.embed_model,
                    metadata=chunk.metadata_ or {}, source_uri=source_uri, title=title,
                    source_type=source_type)  # fmt: skip


@router.get("/conversations", tags=["conversations"])
async def list_conversations(ctx: AppContext = Depends(get_ctx), limit: int = Query(50, ge=1, le=500)) -> list[dict]:
    async with ctx.sessionmaker() as session:
        rows = (await session.scalars(select(Conversation).order_by(Conversation.created_at.desc()).limit(limit))).all()
    return [{"id": str(c.id), "title": c.title, "created_at": c.created_at} for c in rows]


@router.get("/conversations/{conversation_id}/messages", tags=["conversations"])
async def conversation_messages(conversation_id: uuid.UUID, ctx: AppContext = Depends(get_ctx)) -> list[dict]:
    async with ctx.sessionmaker() as session:
        if await session.get(Conversation, conversation_id) is None:
            raise HTTPException(404, "conversation not found")
        messages = (
            await session.scalars(
                select(Message).where(Message.conversation_id == conversation_id).order_by(Message.id)
            )
        ).all()
        ids = [m.id for m in messages]
        citations = (
            await session.execute(
                select(Citation, Document.source_uri)
                .join(Chunk, Chunk.id == Citation.chunk_id)
                .join(Document, Document.id == Chunk.document_id)
                .where(Citation.message_id.in_(ids))
                .order_by(Citation.message_id, Citation.span_start)
            )
        ).all() if ids else []  # fmt: skip
    by_message: dict[int, list[dict]] = {}
    for citation, source_uri in citations:
        by_message.setdefault(citation.message_id, []).append(
            {"n": citation.rank, "chunk_id": citation.chunk_id, "source_uri": source_uri,
             "span_start": citation.span_start, "span_end": citation.span_end, "score": citation.score}  # fmt: skip
        )
    return [
        {"id": m.id, "role": m.role, "content": m.content, "created_at": m.created_at,
         "citations": by_message.get(m.id, [])}  # fmt: skip
        for m in messages
    ]
