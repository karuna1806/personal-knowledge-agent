from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, insert, select, text

from pka.api.context import AppContext
from pka.api.deps import get_ctx, require_api_key
from pka.api.schemas import JobOut
from pka.db.models import Chunk, Document, IngestJob, Memory
from pka.queue.broker import REINDEX_MODEL

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_api_key)])


@router.post("/reindex", status_code=202, response_model=JobOut)
async def reindex(ctx: AppContext = Depends(get_ctx)) -> JobOut:
    """Re-embed chunks whose embed_model differs from the configured model."""
    if not ctx.broker_ready:
        raise HTTPException(503, "message broker unavailable")
    async with ctx.sessionmaker() as session, session.begin():
        job = await session.scalar(
            insert(IngestJob).values(kind="reindex", target=ctx.embedder.model_name).returning(IngestJob)
        )
    await ctx.publish(REINDEX_MODEL, {"job_id": str(job.id)})
    return JobOut.model_validate(job)


@router.get("/queues")
async def queues(ctx: AppContext = Depends(get_ctx)) -> dict:
    if not ctx.broker_ready:
        raise HTTPException(503, "message broker unavailable")
    return await ctx.broker.queue_depths()


@router.get("/dlq")
async def dead_letters(ctx: AppContext = Depends(get_ctx), limit: int = Query(20, ge=1, le=200)) -> list[dict]:
    """Inspect dead-lettered messages without consuming them."""
    if not ctx.broker_ready:
        raise HTTPException(503, "message broker unavailable")
    return await ctx.broker.peek_dlq(limit)


@router.post("/dlq/replay")
async def replay(ctx: AppContext = Depends(get_ctx), limit: int = Query(100, ge=1, le=10_000)) -> dict:
    """Re-queue dead-lettered messages (after fixing whatever made them fail)."""
    if not ctx.broker_ready:
        raise HTTPException(503, "message broker unavailable")
    return {"replayed": await ctx.broker.replay_dlq(limit)}


@router.get("/stats")
async def stats(ctx: AppContext = Depends(get_ctx)) -> dict:
    """Corpus size and query-latency percentiles over the last 24h (from query_logs)."""
    async with ctx.sessionmaker() as session:
        docs = await session.scalar(select(func.count()).select_from(Document))
        chunks = await session.scalar(select(func.count()).select_from(Chunk))
        stale = await session.scalar(
            select(func.count()).select_from(Chunk).where(Chunk.embed_model != ctx.embedder.model_name)
        )
        memories = await session.scalar(
            select(func.count()).select_from(Memory).where(Memory.superseded_by.is_(None), Memory.valid_until.is_(None))
        )
        by_type = dict(
            (await session.execute(select(Document.source_type, func.count()).group_by(Document.source_type))).all()
        )
        latency = (
            await session.execute(
                text(
                    "SELECT count(*) AS n,"
                    " percentile_cont(0.5) WITHIN GROUP (ORDER BY ttft_ms) AS ttft_p50,"
                    " percentile_cont(0.95) WITHIN GROUP (ORDER BY ttft_ms) AS ttft_p95,"
                    " percentile_cont(0.5) WITHIN GROUP (ORDER BY total_ms) AS total_p50,"
                    " percentile_cont(0.95) WITHIN GROUP (ORDER BY total_ms) AS total_p95,"
                    " avg(CASE WHEN cache_hit THEN 1 ELSE 0 END) AS cache_hit_rate"
                    " FROM query_logs WHERE created_at > now() - interval '24 hours'"
                )
            )
        ).mappings().one()
    return {
        "documents": docs, "documents_by_type": by_type, "chunks": chunks,
        "stale_chunks": stale, "active_memories": memories,
        "embed_model": ctx.embedder.model_name, "llm_model": ctx.llm.model,
        "queries_24h": dict(latency),
    }  # fmt: skip
