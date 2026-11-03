import pytest

from pka.agent.citations import extract_citations
from pka.agent.prompt import REFUSAL, Source, build_prompt
from pka.agent.retriever import RetrievedChunk, rrf_fuse
from pka.ingest.tokens import estimate_tokens


def test_rrf_rewards_agreement_between_rankings():
    vector = [1, 2, 3, 4]
    lexical = [3, 9, 1]
    fused = dict(rrf_fuse([vector, lexical], k=60))
    assert fused[1] == pytest.approx(1 / 61 + 1 / 63)
    assert fused[3] == pytest.approx(1 / 63 + 1 / 61)
    ranked = [d for d, _ in rrf_fuse([vector, lexical], k=60)]
    assert set(ranked[:2]) == {1, 3}  # in both lists → top
    assert ranked.index(9) < ranked.index(4)  # rank 2 in one list beats rank 4


def test_rrf_single_ranking_preserves_order():
    assert [d for d, _ in rrf_fuse([[5, 4, 3]])] == [5, 4, 3]


def test_rrf_empty():
    assert rrf_fuse([]) == []


def test_citation_spans_attach_to_claims():
    answer = "RabbitMQ gives DLQs [1]. It also supports prefetch [2][3]. Bogus [7]."
    parsed = extract_citations(answer, n_sources=3)
    spans = {(s.source, answer[s.span_start : s.span_end]) for s in parsed.spans}
    assert spans == {(1, "RabbitMQ gives DLQs"), (2, "It also supports prefetch"), (3, "It also supports prefetch")}
    assert parsed.invalid == [7]


def test_citation_after_full_stop_and_comma_lists():
    answer = "Use HNSW.[1] IVFFlat degrades over time [1, 2]"
    parsed = extract_citations(answer, n_sources=2)
    got = [(s.source, answer[s.span_start : s.span_end]) for s in parsed.spans]
    assert got == [(1, "Use HNSW."), (1, "IVFFlat degrades over time"), (2, "IVFFlat degrades over time")]


def test_location_label():
    chunk = RetrievedChunk(1, 1, "x", {"heading_path": ["A", "B", "C"], "line_start": 3, "line_end": 9},
                           "notes/a.md", "note", "T")  # fmt: skip
    assert chunk.location() == "notes/a.md · B > C · L3-9"


def _src(n: int, size: int) -> Source:
    return Source(n=n, kind="chunk", label=f"doc{n}", content="word " * size, chunk_id=n)


def test_prompt_budget_drops_sources_and_renumbers_densely():
    sources = [_src(1, 200), _src(2, 5000), _src(3, 200)]
    built = build_prompt(route="retrieve", question="q?", candidates=sources, memories=[], history=[],
                         count=estimate_tokens, context_tokens=2048, max_answer_tokens=256)  # fmt: skip
    assert [s.chunk_id for s in built.sources] == [1, 3]
    assert [s.n for s in built.sources] == [1, 2]
    assert built.dropped_sources == 1
    user = built.messages[-1]["content"]
    assert "[1] doc1" in user and "[2] doc3" in user and "doc2" not in user


def test_prompt_keeps_recent_history_and_starts_with_user_turn():
    history = [{"role": "user", "content": "old " * 2000}, {"role": "assistant", "content": "a1"},
               {"role": "user", "content": "u2"}, {"role": "assistant", "content": "a2"}]  # fmt: skip
    built = build_prompt(route="direct", question="hi", candidates=[], memories=[], history=history,
                         count=estimate_tokens, context_tokens=2048, max_answer_tokens=256)  # fmt: skip
    roles = [m["role"] for m in built.messages]
    assert roles == ["system", "user", "assistant", "user"]
    assert built.messages[1]["content"] == "u2"


def test_grounded_prompt_contains_refusal_instruction_and_separate_memory():
    class Mem:
        kind, subject, predicate, object = "preference", "user", "prefers_editor", "vim"
        from datetime import datetime

        valid_from = datetime(2026, 1, 1)

    built = build_prompt(route="retrieve", question="q", candidates=[_src(1, 10)], memories=[Mem()],
                         history=[], count=estimate_tokens, context_tokens=4096, max_answer_tokens=512)  # fmt: skip
    system = built.messages[0]["content"]
    assert REFUSAL in system
    assert "<user_memory" in system and "prefers editor: vim" in system
    assert "<user_memory" not in built.messages[-1]["content"]
