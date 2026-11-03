from datetime import UTC, datetime, timedelta

import pytest

from pka.core.config import Settings
from pka.memory.extract import parse_candidates
from pka.memory.resolve import CandidateMemory, normalize_predicate, normalize_subject, validate
from pka.memory.retrieve import access_score, recency_decay, score_memory


def test_recency_decay_half_life():
    now = datetime(2026, 6, 1, tzinfo=UTC)
    assert recency_decay(now, now, 30) == pytest.approx(1.0)
    assert recency_decay(now - timedelta(days=30), now, 30) == pytest.approx(0.5)
    assert recency_decay(now - timedelta(days=60), now, 30) == pytest.approx(0.25)
    assert recency_decay(now + timedelta(days=1), now, 30) == pytest.approx(1.0)  # clock skew


def test_access_score_is_normalised():
    assert access_score(0, 10) == 0
    assert access_score(10, 10) == pytest.approx(1.0)
    assert access_score(0, 0) == 0  # no division by zero


def test_score_weights():
    s = Settings()
    now = datetime(2026, 6, 1, tzinfo=UTC)
    fresh_popular = score_memory(0.5, now, 10, 10, s, now)
    assert fresh_popular == pytest.approx(0.6 * 0.5 + 0.2 + 0.2)
    old_similar = score_memory(0.9, now - timedelta(days=365), 0, 10, s, now)
    assert old_similar == pytest.approx(0.6 * 0.9 + 0.2 * 0.5 ** (365 / 30))


def test_normalisation():
    assert normalize_predicate("Prefers Queue-Framework!") == "prefers_queue_framework"
    assert normalize_subject("  The User ") == "user"
    assert normalize_subject("Persona  Project") == "Persona Project"


def test_validate_rejects_bad_kinds_and_clamps_confidence():
    assert validate(CandidateMemory("opinion", "user", "x", "y")) is None
    assert validate(CandidateMemory("fact", "user", "  ", "y")) is None
    v = validate(CandidateMemory("FACT", "I", "Works At", " Acme ", 3.0))
    assert (v.kind, v.subject, v.predicate, v.object, v.confidence) == ("fact", "user", "works_at", "Acme", 1.0)


def test_extraction_prompt_shows_known_memories_for_key_reuse():
    from pka.memory.extract import build_messages

    with_known = build_messages("I switched to VS Code", "ok", ["user prefers_editor: Neovim"])[1]["content"]
    assert "ALREADY STORED" in with_known and "- user prefers_editor: Neovim" in with_known
    assert with_known.index("ALREADY STORED") < with_known.index("USER said")
    assert "ALREADY STORED" not in build_messages("hi", "hello", [])[1]["content"]


def test_parse_candidates_filters_low_confidence_and_garbage():
    raw = {"memories": [
        {"kind": "preference", "subject": "user", "predicate": "prefers_editor", "object": "vim", "confidence": 0.9},
        {"kind": "fact", "subject": "user", "predicate": "x", "object": "y", "confidence": 0.1},
        {"kind": "nonsense", "subject": "user", "predicate": "x", "object": "y", "confidence": 0.9},
        "not a dict",
        {"kind": "fact", "subject": "user", "predicate": "p", "object": "o", "confidence": "high"},
    ]}  # fmt: skip
    parsed = parse_candidates(raw)
    assert [(c.predicate, c.object) for c in parsed] == [("prefers_editor", "vim")]
