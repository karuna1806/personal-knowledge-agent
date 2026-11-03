"""Span-level citations (§4): each [n] marker in the answer is attached to the claim it
follows, as character offsets — so grounding is verifiable, not a decorative source list."""

import re
from dataclasses import dataclass

MARKER = re.compile(r"\[(\d{1,3}(?:\s*[,;]\s*\d{1,3})*)\]")
_BOUNDARY = re.compile(r"[.!?](?=\s)|\n")


@dataclass
class CitationSpan:
    source: int  # 1-based source number as it appears in the prompt
    span_start: int
    span_end: int


@dataclass
class ParsedCitations:
    spans: list[CitationSpan]
    invalid: list[int]  # numbers the model cited that don't correspond to any source


def extract_citations(answer: str, n_sources: int) -> ParsedCitations:
    spans: list[CitationSpan] = []
    invalid: list[int] = []
    prev_marker_end = 0
    prev_span: tuple[int, int] | None = None
    for match in MARKER.finditer(answer):
        numbers = [int(n) for n in re.split(r"\s*[,;]\s*", match.group(1))]
        between = answer[prev_marker_end : match.start()]
        if prev_span is not None and not between.strip(" \t"):
            # Adjacent markers ("...prefetch [2][3].") support the same claim.
            start, end = prev_span
        else:
            end = match.start()
            while end > prev_marker_end and answer[end - 1] in " \t":
                end -= 1
            # The claim starts after the last sentence boundary before it (ignoring a full stop
            # that sits right before the marker, as in "...queues.[1]").
            search_upto = end - 1 if end > 0 and answer[end - 1] in ".!?" else end
            start = prev_marker_end
            for boundary in _BOUNDARY.finditer(answer, prev_marker_end, max(prev_marker_end, search_upto)):
                start = boundary.end()
            while start < end and answer[start] in " \t\n":
                start += 1
            # Strip leading punctuation left over from a preceding marker, e.g. ". It also..."
            while start < end and answer[start] in ".,;:":
                start += 1
                while start < end and answer[start] in " \t\n":
                    start += 1
        for n in numbers:
            if 1 <= n <= n_sources:
                spans.append(CitationSpan(n, start, end))
            elif n not in invalid:
                invalid.append(n)
        prev_span = (start, end)
        prev_marker_end = match.end()
    return ParsedCitations(spans, invalid)
