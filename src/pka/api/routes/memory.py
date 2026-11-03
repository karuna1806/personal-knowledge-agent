"""Inspect and correct what the agent believes about me. Nothing is ever deleted — a wrong
memory is retracted (validity ended), and its full history stays queryable."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select

from pka.api.context import AppContext
from pka.api.deps import get_ctx, require_api_key
from pka.api.schemas import MemoryOut
from pka.core.versions import MEMORY_VERSION_KEY, bump
from pka.db.models import Memory
from pka.memory.resolve import memory_history, retract_memory

router = APIRouter(prefix="/memories", tags=["memory"], dependencies=[Depends(require_api_key)])


def _out(m: Memory) -> MemoryOut:
    return MemoryOut(
        id=m.id, kind=m.kind, subject=m.subject, predicate=m.predicate, object=m.object,
        confidence=m.confidence, source_msg_id=m.source_msg_id, superseded_by=m.superseded_by,
        valid_from=m.valid_from, valid_until=m.valid_until, access_count=m.access_count,
        last_accessed=m.last_accessed, active=m.superseded_by is None and m.valid_until is None,
    )  # fmt: skip


@router.get("", response_model=list[MemoryOut])
async def list_memories(
    ctx: AppContext = Depends(get_ctx),
    include_history: bool = Query(False, description="include superseded/retracted rows"),
    limit: int = Query(200, ge=1, le=2000),
) -> list[MemoryOut]:
    stmt = select(Memory).order_by(Memory.valid_from.desc()).limit(limit)
    if not include_history:
        stmt = stmt.where(Memory.superseded_by.is_(None), Memory.valid_until.is_(None))
    async with ctx.sessionmaker() as session:
        return [_out(m) for m in (await session.scalars(stmt)).all()]


@router.get("/{memory_id}/history", response_model=list[MemoryOut])
async def history(memory_id: int, ctx: AppContext = Depends(get_ctx)) -> list[MemoryOut]:
    """Every value the agent has held for this memory's (subject, predicate), oldest first."""
    async with ctx.sessionmaker() as session:
        rows = await memory_history(session, memory_id)
    if not rows:
        raise HTTPException(404, "memory not found")
    return [_out(m) for m in rows]


@router.post("/{memory_id}/retract", response_model=MemoryOut)
async def retract(memory_id: int, ctx: AppContext = Depends(get_ctx)) -> MemoryOut:
    async with ctx.sessionmaker() as session, session.begin():
        if not await retract_memory(session, memory_id):
            raise HTTPException(409 if await session.get(Memory, memory_id) else 404, "memory not active or not found")
    await bump(ctx.redis, MEMORY_VERSION_KEY)
    async with ctx.sessionmaker() as session:
        return _out(await session.get(Memory, memory_id))
