"""Scored memory lookup (§5):

    score = w_sim · cosine(query, memory)
          + w_rec · recency_decay(valid_from)          # exponential, configurable half-life
          + w_acc · log1p(access_count) / log1p(max)   # frequently-useful memories surface

Weights are a hypothesis to tune against the eval set, so they live in config.
"""

import math
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from pka.core.config import Settings
from pka.db.models import Memory

CANDIDATE_POOL = 50


@dataclass
class ScoredMemory:
    id: int
    kind: str
    subject: str
    predicate: str
    object: str
    confidence: float
    similarity: float
    score: float
    valid_from: datetime


def recency_decay(valid_from: datetime, now: datetime, half_life_days: float) -> float:
    age_days = max(0.0, (now - valid_from).total_seconds() / 86400)
    return math.exp(-math.log(2) * age_days / half_life_days)


def access_score(access_count: int, max_access: int) -> float:
    return math.log1p(access_count) / math.log1p(max(max_access, 1))


def score_memory(similarity: float, valid_from: datetime, access_count: int, max_access: int,
                 settings: Settings, now: datetime | None = None) -> float:  # fmt: skip
    now = now or datetime.now(UTC)
    return (
        settings.memory_w_similarity * similarity
        + settings.memory_w_recency * recency_decay(valid_from, now, settings.memory_half_life_days)
        + settings.memory_w_access * access_score(access_count, max_access)
    )


async def nearest_active(session: AsyncSession, query_vector, limit: int) -> list[Memory]:
    """Active memories closest to `query_vector`, no threshold, no access bookkeeping."""
    rows = await session.scalars(
        select(Memory)
        .where(Memory.superseded_by.is_(None), Memory.valid_until.is_(None), Memory.embedding.is_not(None))
        .order_by(Memory.embedding.cosine_distance(query_vector))
        .limit(limit)
    )
    return list(rows.all())


async def retrieve_memories(
    session: AsyncSession, query_vector, settings: Settings, *, touch: bool = True
) -> list[ScoredMemory]:
    distance = Memory.embedding.cosine_distance(query_vector)
    active = (
        Memory.superseded_by.is_(None),
        Memory.valid_until.is_(None),
        Memory.embedding.is_not(None),
    )
    rows = (
        await session.execute(
            select(Memory, (1 - distance).label("similarity"))
            .where(*active)
            .order_by(distance)
            .limit(CANDIDATE_POOL)
        )
    ).all()
    if not rows:
        return []
    max_access = await session.scalar(select(func.max(Memory.access_count)).where(*active)) or 0
    now = datetime.now(UTC)
    scored = [
        ScoredMemory(
            id=m.id, kind=m.kind, subject=m.subject, predicate=m.predicate, object=m.object,
            confidence=m.confidence, similarity=float(sim),
            score=score_memory(float(sim), m.valid_from, m.access_count, max_access, settings, now),
            valid_from=m.valid_from,
        )  # fmt: skip
        for m, sim in rows
        if float(sim) >= settings.memory_min_similarity
    ]
    scored.sort(key=lambda s: s.score, reverse=True)
    top = scored[: settings.memory_top_k]
    if touch and top:
        await session.execute(
            update(Memory)
            .where(Memory.id.in_([m.id for m in top]))
            .values(access_count=Memory.access_count + 1, last_accessed=now)
        )
        await session.commit()
    return top
