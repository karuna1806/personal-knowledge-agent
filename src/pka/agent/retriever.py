"""Hybrid retrieval (§3.9): pgvector HNSW top-k ∪ Postgres full-text top-k, merged with
Reciprocal Rank Fusion. RRF needs only ranks, so cosine and ts_rank — which are not on
comparable scales — never have to be normalised against each other."""

import time
from dataclasses import dataclass, field

from sqlalchemy import Text, cast, func, select, text
from sqlalchemy.dialects.postgresql import TSQUERY
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pka.core.config import Settings
from pka.core.metrics import STAGE_LATENCY
from pka.core.telemetry import tracer
from pka.db.models import Chunk, Document
from pka.ingest.embed import Embedder


@dataclass
class SearchFilters:
    source_types: list[str] | None = None
    language: str | None = None
    source_prefix: str | None = None

    def is_empty(self) -> bool:
        return not (self.source_types or self.language or self.source_prefix)


@dataclass
class RetrievedChunk:
    chunk_id: int
    document_id: int
    content: str
    metadata: dict
    source_uri: str
    source_type: str
    title: str | None
    vector_rank: int | None = None
    lexical_rank: int | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None

    @property
    def score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.rrf_score

    def location(self) -> str:
        """Human-readable citation anchor: file, section/symbol, lines/page."""
        m = self.metadata
        parts = [self.source_uri]
        if m.get("heading_path"):
            parts.append(" > ".join(m["heading_path"][-2:]))
        if m.get("symbol"):
            parts.append(f"{m['parent']}.{m['symbol']}" if m.get("parent") else m["symbol"])
        if m.get("page"):
            parts.append(f"p.{m['page']}")
        elif m.get("line_start"):
            parts.append(f"L{m['line_start']}-{m.get('line_end', m['line_start'])}")
        return " · ".join(parts)


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    vector_ids: list[int] = field(default_factory=list)
    lexical_ids: list[int] = field(default_factory=list)
    retrieval_ms: int = 0


def rrf_fuse(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """score(d) = Σ_r 1 / (k + rank_r(d)), ranks 1-based. Ties broken by first appearance."""
    scores: dict[int, float] = {}
    order: dict[int, int] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            order.setdefault(doc_id, len(order))
    return sorted(scores.items(), key=lambda item: (-item[1], order[item[0]]))


def _apply_filters(stmt, filters: SearchFilters | None):
    if not filters:
        return stmt
    if filters.source_types:
        stmt = stmt.where(Document.source_type.in_(filters.source_types))
    if filters.language:
        stmt = stmt.where(Chunk.metadata_["language"].astext == filters.language)
    if filters.source_prefix:
        escaped = filters.source_prefix.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")
        stmt = stmt.where(Document.source_uri.like(escaped + "%"))
    return stmt


class Retriever:
    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession], embedder: Embedder, settings: Settings):
        self._sessionmaker = sessionmaker
        self._embedder = embedder
        self._settings = settings

    async def vector_search(self, session: AsyncSession, query_vector, k: int,
                            filters: SearchFilters | None) -> list[int]:  # fmt: skip
        with tracer.start_as_current_span("retrieve.vector"), STAGE_LATENCY.labels("vector_search").time():
            await session.execute(text(f"SET LOCAL hnsw.ef_search = {int(self._settings.hnsw_ef_search)}"))
            if filters and not filters.is_empty():
                # Filtered HNSW scans can under-return; iterative scan keeps going until k match.
                await session.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
            distance = Chunk.embedding.cosine_distance(query_vector)
            stmt = select(Chunk.id, distance.label("d")).join(Document, Document.id == Chunk.document_id)
            stmt = _apply_filters(stmt, filters).order_by(distance).limit(k)
            rows = (await session.execute(stmt)).all()
            # relaxed_order may return slightly out of order; re-sort on the exact distance.
            return [r.id for r in sorted(rows, key=lambda r: r.d)]

    async def lexical_search(self, session: AsyncSession, query: str, k: int,
                             filters: SearchFilters | None) -> list[int]:  # fmt: skip
        """OR-semantics full-text match: plainto_tsquery ANDs every term, which returns
        nothing for most natural-language questions. ts_rank_cd still rewards chunks that
        match more (and closer) terms."""
        with tracer.start_as_current_span("retrieve.lexical"), STAGE_LATENCY.labels("lexical_search").time():
            or_query = func.replace(cast(func.plainto_tsquery("english", query), Text), " & ", " | ")
            tsq = cast(or_query, TSQUERY)
            rank = func.ts_rank_cd(Chunk.tsv, tsq, 1)
            stmt = (
                select(Chunk.id, rank.label("r"))
                .join(Document, Document.id == Chunk.document_id)
                .where(Chunk.tsv.op("@@")(tsq))
            )
            stmt = _apply_filters(stmt, filters).order_by(rank.desc(), Chunk.id).limit(k)
            return [r.id for r in (await session.execute(stmt)).all()]

    async def _hydrate(self, session: AsyncSession, ids: list[int]) -> dict[int, RetrievedChunk]:
        if not ids:
            return {}
        rows = (
            await session.execute(
                select(
                    Chunk.id, Chunk.document_id, Chunk.content, Chunk.metadata_,
                    Document.source_uri, Document.source_type, Document.title,
                )  # fmt: skip
                .join(Document, Document.id == Chunk.document_id)
                .where(Chunk.id.in_(ids))
            )
        ).all()
        return {
            r.id: RetrievedChunk(r.id, r.document_id, r.content, r.metadata_ or {}, r.source_uri,
                                 r.source_type, r.title)  # fmt: skip
            for r in rows
        }

    async def search(
        self,
        query: str,
        *,
        mode: str | None = None,
        filters: SearchFilters | None = None,
        k: int | None = None,
    ) -> RetrievalResult:
        mode = mode or self._settings.retrieval_mode
        if mode not in {"hybrid", "vector", "lexical"}:
            raise ValueError(f"unknown retrieval mode {mode!r}")
        fused_k = k or self._settings.fused_k
        started = time.perf_counter()
        with tracer.start_as_current_span("retrieve") as span:
            span.set_attribute("retrieve.mode", mode)
            query_vector = await self._embedder.embed_query(query) if mode != "lexical" else None
            async with self._sessionmaker() as session, session.begin():
                vector_ids = (
                    await self.vector_search(session, query_vector, self._settings.vector_k, filters)
                    if mode in {"hybrid", "vector"}
                    else []
                )
                lexical_ids = (
                    await self.lexical_search(session, query, self._settings.lexical_k, filters)
                    if mode in {"hybrid", "lexical"}
                    else []
                )
                fused = rrf_fuse([r for r in (vector_ids, lexical_ids) if r], self._settings.rrf_k)[:fused_k]
                hydrated = await self._hydrate(session, [doc_id for doc_id, _ in fused])
            vector_rank = {cid: i for i, cid in enumerate(vector_ids, start=1)}
            lexical_rank = {cid: i for i, cid in enumerate(lexical_ids, start=1)}
            chunks: list[RetrievedChunk] = []
            for chunk_id, score in fused:
                chunk = hydrated.get(chunk_id)
                if chunk is None:  # deleted between search and hydrate
                    continue
                chunk.rrf_score = score
                chunk.vector_rank = vector_rank.get(chunk_id)
                chunk.lexical_rank = lexical_rank.get(chunk_id)
                chunks.append(chunk)
            span.set_attribute("retrieve.results", len(chunks))
        return RetrievalResult(
            chunks=chunks,
            vector_ids=vector_ids,
            lexical_ids=lexical_ids,
            retrieval_ms=int((time.perf_counter() - started) * 1000),
        )
