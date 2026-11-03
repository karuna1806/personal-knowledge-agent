"""Conflict resolution by supersession with validity intervals (§5).

A new value for an existing (subject, predicate) never overwrites: the new row is inserted,
and the old row gets `superseded_by` and `valid_until`. Nothing is deleted, so "why does the
agent believe X?" — and "what did it believe last month?" — stay answerable.
"""

import re
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from pka.db.models import Memory

KINDS = ("preference", "fact", "project", "entity")


@dataclass
class CandidateMemory:
    kind: str
    subject: str
    predicate: str
    object: str
    confidence: float = 0.5


@dataclass
class StoreResult:
    action: str  # inserted | superseded | reinforced
    memory_id: int
    superseded_ids: list[int]


def normalize_subject(value: str) -> str:
    value = " ".join(value.strip().split())
    return "user" if value.lower() in {"user", "the user", "i", "me", "myself"} else value


def normalize_predicate(value: str) -> str:
    value = re.sub(r"[^0-9a-zA-Z]+", "_", value.strip().lower())
    return value.strip("_")


def _same_object(a: str, b: str) -> bool:
    return " ".join(a.lower().split()).rstrip(".") == " ".join(b.lower().split()).rstrip(".")


def memory_text(subject: str, predicate: str, obj: str) -> str:
    return f"{subject} {predicate.replace('_', ' ')} {obj}"


def validate(candidate: CandidateMemory) -> CandidateMemory | None:
    kind = candidate.kind.strip().lower()
    subject = normalize_subject(candidate.subject)
    predicate = normalize_predicate(candidate.predicate)
    obj = " ".join(candidate.object.strip().split())
    if kind not in KINDS or not subject or not predicate or not obj:
        return None
    confidence = min(1.0, max(0.0, float(candidate.confidence)))
    return CandidateMemory(kind, subject[:200], predicate[:100], obj[:1000], confidence)


async def store_memory(
    session: AsyncSession,
    candidate: CandidateMemory,
    embedding,
    source_msg_id: int | None,
) -> StoreResult:
    """Must run inside a transaction. Serialised per (subject, predicate) with an advisory
    lock so two workers can't both insert a 'current' value for the same key."""
    await session.execute(
        text("SELECT pg_advisory_xact_lock(hashtext(:k))"),
        {"k": f"{candidate.subject.lower()}|{candidate.predicate}"},
    )
    active = (
        await session.scalars(
            select(Memory)
            .where(
                func.lower(Memory.subject) == candidate.subject.lower(),
                Memory.predicate == candidate.predicate,
                Memory.superseded_by.is_(None),
                Memory.valid_until.is_(None),
            )
            .with_for_update()
        )
    ).all()

    for row in active:
        if _same_object(row.object, candidate.object):
            # Restating a known fact isn't a conflict; it's evidence. Keep the stronger belief.
            if candidate.confidence > row.confidence:
                row.confidence = candidate.confidence
            return StoreResult("reinforced", row.id, [])

    new = Memory(
        kind=candidate.kind,
        subject=candidate.subject,
        predicate=candidate.predicate,
        object=candidate.object,
        confidence=candidate.confidence,
        embedding=embedding,
        source_msg_id=source_msg_id,
    )
    session.add(new)
    await session.flush()
    superseded = [row.id for row in active]
    if superseded:
        await session.execute(
            update(Memory)
            .where(Memory.id.in_(superseded))
            .values(superseded_by=new.id, valid_until=datetime.now(UTC))
        )
    return StoreResult("superseded" if superseded else "inserted", new.id, superseded)


async def retract_memory(session: AsyncSession, memory_id: int) -> bool:
    """End a memory's validity without replacing it ("that's wrong, forget it")."""
    result = await session.execute(
        update(Memory)
        .where(Memory.id == memory_id, Memory.valid_until.is_(None), Memory.superseded_by.is_(None))
        .values(valid_until=datetime.now(UTC))
    )
    return bool(result.rowcount)


async def memory_history(session: AsyncSession, memory_id: int) -> list[Memory]:
    """Every belief the agent has held for this memory's (subject, predicate), oldest first."""
    anchor = await session.get(Memory, memory_id)
    if anchor is None:
        return []
    return list(
        (
            await session.scalars(
                select(Memory)
                .where(func.lower(Memory.subject) == anchor.subject.lower(), Memory.predicate == anchor.predicate)
                .order_by(Memory.valid_from, Memory.id)
            )
        ).all()
    )
