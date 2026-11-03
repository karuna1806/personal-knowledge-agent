"""Run the golden set.

    python -m evals.run --name baseline --generate              # live DB, current config
    python -m evals.run --name vector-only --mode vector         # retrieval-only experiment
    python -m evals.run --name no-rerank --no-rerank
    python -m evals.run --name fixed-chunks --build-index --chunker fixed \
        --db postgresql+asyncpg://pka:pka@localhost:5433/pka_eval_fixed

Each run is stored in eval_runs/eval_results (of the target DB) and written up as
docs/evals/<name>.md. Retrieval metrics are always computed; generation metrics (refusal
accuracy, citation validity, optional LLM-judged faithfulness) with --generate.
"""

import argparse
import asyncio
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import yaml
from sqlalchemy import insert, select, text
from sqlalchemy.ext.asyncio import create_async_engine

from evals.metrics import is_refusal, mean, mrr, ndcg_at_k, recall_at_k
from pka.agent.citations import extract_citations
from pka.agent.prompt import Source, build_prompt
from pka.agent.reranker import CrossEncoderReranker, PassthroughReranker
from pka.agent.retriever import Retriever
from pka.core.config import get_settings
from pka.db.models import Chunk, Document, EvalResult, EvalRun
from pka.db.session import make_engine, make_sessionmaker
from pka.ingest.embed import Embedder, make_encoder
from pka.ingest.loaders import iter_corpus_files, load_path
from pka.ingest.pipeline import ingest_document
from pka.ingest.tokens import get_token_counter
from pka.llm.client import LLMClient, StreamStats

ROOT = Path(__file__).resolve().parents[1]

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"supported_claims": {"type": "integer"}, "unsupported_claims": {"type": "integer"}},
    "required": ["supported_claims", "unsupported_claims"],
    "additionalProperties": False,
}
JUDGE_SYSTEM = (
    "You grade an answer for faithfulness to its sources. Split the ANSWER into its factual "
    "claims. Count how many are fully supported by the SOURCES and how many are not "
    "(unsupported, contradicted, or added from outside knowledge). Ignore the citation "
    "markers themselves. A refusal has 0 claims."
)


@dataclass
class QuestionResult:
    id: str
    category: str
    question: str
    answerable: bool
    expected_ids: list[int]
    retrieved_ids: list[int]
    recall_at_5: float | None
    recall_at_20: float | None
    mrr: float | None
    ndcg_at_5: float | None
    answer: str | None = None
    refused: bool | None = None
    valid_citations: int | None = None
    invalid_citations: int | None = None
    faithfulness: float | None = None
    ttft_ms: int | None = None
    notes: list[str] = field(default_factory=list)


async def build_index(db_url: str, corpus: Path, settings, embedder) -> None:
    admin_url = db_url.rsplit("/", 1)[0] + "/postgres"
    db_name = db_url.rsplit("/", 1)[1]
    engine = create_async_engine(admin_url, isolation_level="AUTOCOMMIT")
    async with engine.connect() as conn:
        exists = await conn.scalar(text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": db_name})
        if not exists:
            await conn.execute(text(f'CREATE DATABASE "{db_name}"'))
    await engine.dispose()
    await asyncio.to_thread(
        subprocess.run, [sys.executable, "-m", "alembic", "-x", f"dburl={db_url}", "upgrade", "head"],
        cwd=ROOT, check=True, capture_output=True,
    )  # fmt: skip
    engine = make_engine(db_url)
    async with engine.begin() as conn:
        await conn.execute(text("TRUNCATE documents RESTART IDENTITY CASCADE"))
    sessionmaker = make_sessionmaker(engine)
    root = corpus.resolve()  # noqa: ASYNC240 — one-shot CLI, not a server
    for path in iter_corpus_files(root):
        doc = load_path(path, source_uri=str(path.relative_to(root)))
        outcome = await ingest_document(sessionmaker, embedder, doc, settings)
        print(f"  indexed {doc.source_uri}: {outcome.chunks} chunks")
    await engine.dispose()


async def resolve_locators(session, expected: list[dict]) -> list[set[int]]:
    hits: list[set[int]] = []
    for loc in expected:
        rows = await session.scalars(
            select(Chunk.id)
            .join(Document, Document.id == Chunk.document_id)
            .where(Document.source_uri.like(f"%{loc['source']}"), Chunk.content.contains(loc["contains"]))
        )
        hits.append(set(rows.all()))
    return hits


async def judge(llm: LLMClient, answer: str, sources: list[Source]) -> float | None:
    if not answer or is_refusal(answer):
        return None
    body = "\n\n".join(f"[{s.n}] {s.content}" for s in sources)
    raw = await llm.complete_json(
        [{"role": "system", "content": JUDGE_SYSTEM},
         {"role": "user", "content": f"SOURCES:\n{body}\n\nANSWER:\n{answer}"}],  # fmt: skip
        JUDGE_SCHEMA, name="judge", max_tokens=60,
    )  # fmt: skip
    s, u = int(raw.get("supported_claims", 0)), int(raw.get("unsupported_claims", 0))
    return s / (s + u) if (s + u) else None


def summarise(results: list[QuestionResult]) -> dict:
    answerable = [r for r in results if r.answerable]
    negatives = [r for r in results if not r.answerable]
    out: dict = {
        "n": len(results),
        "recall@5": mean([r.recall_at_5 for r in answerable]),
        "recall@20": mean([r.recall_at_20 for r in answerable]),
        "mrr": mean([r.mrr for r in answerable]),
        "ndcg@5": mean([r.ndcg_at_5 for r in answerable]),
    }
    generated = [r for r in results if r.answer is not None]
    if generated:
        out["refusal_accuracy"] = mean([1.0 if r.refused else 0.0 for r in negatives if r.answer is not None])
        out["false_refusal_rate"] = mean([1.0 if r.refused else 0.0 for r in answerable if r.answer is not None])
        cited = [r for r in answerable if r.answer is not None and not r.refused]
        out["answers_with_citation"] = mean([1.0 if r.valid_citations else 0.0 for r in cited])
        out["invalid_citation_rate"] = mean(
            [r.invalid_citations / max(1, r.invalid_citations + r.valid_citations) for r in cited]
        )
        out["faithfulness"] = mean([r.faithfulness for r in answerable])
        ttfts = sorted(r.ttft_ms for r in generated if r.ttft_ms is not None)
        if ttfts:
            out["ttft_p50_ms"] = ttfts[len(ttfts) // 2]
            out["ttft_p95_ms"] = ttfts[min(len(ttfts) - 1, int(len(ttfts) * 0.95))]
    return out


def by_category(results: list[QuestionResult]) -> dict[str, dict]:
    cats: dict[str, list[QuestionResult]] = {}
    for r in results:
        cats.setdefault(r.category, []).append(r)
    return {c: summarise(rs) for c, rs in cats.items()}


def fmt(v) -> str:
    if v is None:
        return "–"
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def write_report(path: Path, name: str, config: dict, results: list[QuestionResult]) -> None:
    overall = summarise(results)
    cats = by_category(results)
    keys = [k for k in ("recall@5", "recall@20", "mrr", "ndcg@5", "refusal_accuracy", "false_refusal_rate",
                        "answers_with_citation", "invalid_citation_rate", "faithfulness") if k in overall]  # fmt: skip
    lines = [
        f"# Eval run: {name}",
        "",
        f"_{datetime.now(UTC):%Y-%m-%d %H:%M UTC}_ · {overall['n']} questions",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(config, indent=2),
        "```",
        "",
        "## Overall",
        "",
        "| metric | value |",
        "|---|---|",
        *[f"| {k} | {fmt(overall[k])} |" for k in keys + [k for k in ("ttft_p50_ms", "ttft_p95_ms") if k in overall]],
        "",
        "## By category",
        "",
        "| category | n | " + " | ".join(keys) + " |",
        "|---|---|" + "---|" * len(keys),
        *[f"| {c} | {s['n']} | " + " | ".join(fmt(s.get(k)) for k in keys) + " |" for c, s in sorted(cats.items())],
        "",
        "## Per question",
        "",
        "| id | R@5 | MRR | refused | cites | notes |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.id} | {fmt(r.recall_at_5)} | {fmt(r.mrr)} | {fmt(r.refused)} | "
            f"{fmt(r.valid_citations)} | {'; '.join(r.notes)} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--golden", type=Path, default=ROOT / "evals" / "golden_set.yaml")
    parser.add_argument("--db", default=None, help="target database URL (default: DATABASE_URL)")
    parser.add_argument("--build-index", action="store_true", help="(re)build the target DB from --corpus first")
    parser.add_argument("--corpus", type=Path, default=ROOT / "corpus")
    parser.add_argument("--chunker", choices=["structure", "fixed"], default=None)
    parser.add_argument("--mode", choices=["hybrid", "vector", "lexical"], default=None)
    parser.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--k", type=int, default=None, help="chunks passed to the LLM (FINAL_K)")
    parser.add_argument("--rerank-max-length", type=int, default=None)
    parser.add_argument("--generate", action="store_true", help="also generate answers")
    parser.add_argument("--judge", action="store_true", help="LLM-judged faithfulness (implies --generate)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "docs" / "evals")
    args = parser.parse_args()

    overrides = {k: v for k, v in {"chunker": args.chunker, "retrieval_mode": args.mode,
                                    "rerank_enabled": args.rerank, "final_k": args.k,
                                    "rerank_max_length": args.rerank_max_length}.items() if v is not None}  # fmt: skip
    settings = get_settings().model_copy(update=overrides)
    db_url = args.db or settings.database_url
    generate = args.generate or args.judge

    encoder = make_encoder(settings.embed_model, settings.embed_dim, device=settings.embed_device,
                           query_prompt=settings.embed_query_prompt, batch_size=settings.embed_batch_size)  # fmt: skip
    embedder = Embedder(encoder, None, settings.embed_cache_ttl_s)  # no cache: measure the real thing
    if args.build_index:
        print(f"building index in {db_url} (chunker={settings.chunker})")
        await build_index(db_url, args.corpus, settings, embedder)

    engine = make_engine(db_url)
    sessionmaker = make_sessionmaker(engine)
    retriever = Retriever(sessionmaker, embedder, settings)
    reranker = (
        CrossEncoderReranker(settings.rerank_model, settings.embed_device, settings.rerank_max_length)
        if settings.rerank_enabled
        else PassthroughReranker()
    )
    llm = LLMClient(settings.llm_base_url, settings.llm_model, reasoning_effort=settings.llm_reasoning_effort,
                    temperature=0.0, max_tokens=settings.llm_max_tokens, timeout_s=settings.llm_timeout_s)  # fmt: skip
    count = get_token_counter(settings.embed_model)

    golden = yaml.safe_load(args.golden.read_text())["questions"][: args.limit]
    config = {
        "chunker": settings.chunker, "retrieval_mode": settings.retrieval_mode,
        "reranker": getattr(reranker, "model_name", None), "rerank_max_length": settings.rerank_max_length,
        "final_k": settings.final_k,
        "fused_k": settings.fused_k, "embed_model": settings.embed_model,
        "llm_model": settings.llm_model if generate else None, "chunk_max_tokens": settings.chunk_max_tokens,
        "golden_set": args.golden.name, "db": db_url.rsplit("/", 1)[-1],
    }  # fmt: skip

    results: list[QuestionResult] = []
    started = time.perf_counter()
    for item in golden:
        answerable = item.get("answerable", True)
        async with sessionmaker() as session:
            hits = await resolve_locators(session, item.get("expected", []))
        notes = [f"unresolved locator: {loc['contains'][:40]}" for loc, h in zip(item.get("expected", []), hits, strict=True) if not h]
        retrieval = await retriever.search(item["question"])
        reranked, _ = await reranker.rerank(item["question"], retrieval.chunks, max(settings.final_k, 20))
        ranked_ids = [c.chunk_id for c in reranked]
        result = QuestionResult(
            id=item["id"], category=item["category"], question=item["question"], answerable=answerable,
            expected_ids=sorted(set().union(*hits)) if hits else [], retrieved_ids=ranked_ids,
            recall_at_5=recall_at_k(ranked_ids, hits, 5) if answerable else None,
            recall_at_20=recall_at_k(ranked_ids, hits, 20) if answerable else None,
            mrr=mrr(ranked_ids, hits) if answerable else None,
            ndcg_at_5=ndcg_at_k(ranked_ids, hits, 5) if answerable else None,
            notes=notes,
        )  # fmt: skip
        if generate:
            top = reranked[: settings.final_k]
            candidates = [Source(n=i, kind="chunk", label=c.location(), content=c.content, chunk_id=c.chunk_id)
                          for i, c in enumerate(top, start=1)]  # fmt: skip
            built = build_prompt(route="retrieve", question=item["question"], candidates=candidates, memories=[],
                                 history=[], count=count, context_tokens=settings.llm_context_tokens,
                                 max_answer_tokens=settings.llm_max_tokens)  # fmt: skip
            stats = StreamStats()
            answer = "".join([t async for t in llm.stream(built.messages, stats)]).strip()
            parsed = extract_citations(answer, len(built.sources))
            result.answer = answer
            result.ttft_ms = stats.ttft_ms
            result.refused = is_refusal(answer)
            result.valid_citations = len({s.source for s in parsed.spans})
            result.invalid_citations = len(parsed.invalid)
            if args.judge and answerable:
                try:
                    result.faithfulness = await judge(llm, answer, built.sources)
                except Exception as exc:
                    result.notes.append(f"judge failed: {exc}")
            if answerable and result.refused:
                result.notes.append("false refusal")
            if not answerable and not result.refused:
                result.notes.append("confabulated")
        results.append(result)
        status = f"R@5={fmt(result.recall_at_5)}" if answerable else f"refused={result.refused}"
        print(f"  {result.id:<4} {status}  {'; '.join(result.notes)}")

    async with sessionmaker() as session, session.begin():
        run_id = await session.scalar(insert(EvalRun).values(name=args.name, config=config).returning(EvalRun.id))
        await session.execute(insert(EvalResult), [
            {"run_id": run_id, "question": r.question, "expected_chunk_ids": r.expected_ids,
             "retrieved_chunk_ids": r.retrieved_ids[:20], "answer": r.answer, "recall_at_k": r.recall_at_5,
             "mrr": r.mrr, "faithfulness": r.faithfulness,
             "notes": json.dumps({"id": r.id, "category": r.category, "refused": r.refused,
                                  "valid_citations": r.valid_citations, "notes": r.notes})}
            for r in results
        ])  # fmt: skip
    await engine.dispose()
    await llm.close()

    report = args.report_dir / f"{args.name}.md"
    write_report(report, args.name, config, results)
    overall = summarise(results)
    print(f"\n{args.name}: " + "  ".join(f"{k}={fmt(v)}" for k, v in overall.items()))
    print(f"run #{run_id} stored; report → {report.relative_to(ROOT)}  ({time.perf_counter() - started:.0f}s)")


if __name__ == "__main__":
    asyncio.run(main())
