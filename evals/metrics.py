"""Retrieval and generation metrics. Retrieval is scored independently of generation so a
regression can be attributed to the stage that caused it (§6).

Ground truth is a list of *locators* per question; each locator is satisfied by any chunk
that matches it (a phrase may legitimately appear in more than one chunk, and chunk ids
change across re-ingests and chunking strategies)."""

import math

REFUSAL_MARKERS = ("i don't know", "i do not know", "not contain", "no information")


def _satisfied(retrieved: list[int], locator_hits: list[set[int]], k: int) -> list[int | None]:
    """For each locator, the 1-based rank of the first retrieved chunk satisfying it."""
    top = retrieved[:k]
    out: list[int | None] = []
    for hits in locator_hits:
        rank = next((i for i, cid in enumerate(top, start=1) if cid in hits), None)
        out.append(rank)
    return out


def recall_at_k(retrieved: list[int], locator_hits: list[set[int]], k: int) -> float | None:
    if not locator_hits:
        return None
    ranks = _satisfied(retrieved, locator_hits, k)
    return sum(r is not None for r in ranks) / len(locator_hits)


def mrr(retrieved: list[int], locator_hits: list[set[int]], k: int = 20) -> float | None:
    if not locator_hits:
        return None
    ranks = [r for r in _satisfied(retrieved, locator_hits, k) if r is not None]
    return 1.0 / min(ranks) if ranks else 0.0


def ndcg_at_k(retrieved: list[int], locator_hits: list[set[int]], k: int = 5) -> float | None:
    """Binary relevance; each locator contributes at most once (its best-ranked match)."""
    if not locator_hits:
        return None
    ranks = [r for r in _satisfied(retrieved, locator_hits, k) if r is not None]
    dcg = sum(1.0 / math.log2(r + 1) for r in ranks)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(locator_hits), k) + 1))
    return dcg / ideal if ideal else 0.0


def is_refusal(answer: str) -> bool:
    head = answer.strip().lower()[:200]
    return any(marker in head for marker in REFUSAL_MARKERS)


def mean(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None
