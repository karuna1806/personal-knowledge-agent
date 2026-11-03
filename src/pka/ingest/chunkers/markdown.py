"""Split markdown on heading boundaries, keeping the heading path so citations can name
the section. Headings inside fenced code blocks are not headings."""

import re

from pka.ingest.chunkers.base import ChunkDraft, Unit, has_substance, pack_units
from pka.ingest.tokens import TokenCounter

_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_FENCE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")


def _sections(lines: list[str]) -> list[tuple[list[str], int, int]]:
    """Return (heading_path, start_idx, end_idx_exclusive) for each section."""
    sections: list[tuple[list[str], int, int]] = []
    stack: list[tuple[int, str]] = []
    start = 0
    path: list[str] = []
    fence: str | None = None
    for i, line in enumerate(lines):
        fence_match = _FENCE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            if fence is None:
                fence = marker[0] * 3
            elif marker.startswith(fence):
                fence = None
            continue
        if fence is not None:
            continue
        heading = _HEADING.match(line)
        if not heading:
            continue
        if i > start:
            sections.append((path, start, i))
        level = len(heading.group(1))
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, heading.group(2).strip()))
        path = [title for _, title in stack]
        start = i
    if start < len(lines):
        sections.append((path, start, len(lines)))
    return sections


def _blocks(lines: list[str], offset: int) -> list[Unit]:
    """Paragraph-level units (blank-line separated); a fenced block is always one unit."""
    units: list[Unit] = []
    buf: list[str] = []
    buf_start = 0
    fence: str | None = None
    for i, line in enumerate(lines):
        fence_match = _FENCE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            if fence is None:
                fence = marker[0] * 3
            elif marker.startswith(fence):
                fence = None
        if not line.strip() and fence is None:
            if buf:
                units.append(Unit("\n".join(buf), offset + buf_start + 1, offset + i))
                buf = []
            continue
        if not buf:
            buf_start = i
        buf.append(line)
    if buf:
        units.append(Unit("\n".join(buf), offset + buf_start + 1, offset + len(lines)))
    return units


def chunk_markdown(text: str, count: TokenCounter, max_tokens: int) -> list[ChunkDraft]:
    lines = text.split("\n")
    drafts: list[ChunkDraft] = []
    for path, start, end in _sections(lines):
        section_lines = lines[start:end]
        body = "\n".join(section_lines).strip()
        # A heading immediately followed by a sub-heading has no body of its own; its title
        # survives in the children's heading_path.
        has_heading = bool(path) and _HEADING.match(section_lines[0] or "") is not None
        body_only = "\n".join(section_lines[1:] if has_heading else section_lines)
        if not has_substance(body_only):
            continue
        meta_base = {"heading_path": path, "heading": path[-1] if path else None}
        tokens = count(body)
        if tokens <= max_tokens:
            first, last = _trim_range(section_lines, start)
            drafts.append(ChunkDraft(body, {**meta_base, "line_start": first, "line_end": last}, tokens))
            continue
        groups = pack_units(_blocks(section_lines, start), max_tokens, count)
        for part, group in enumerate(groups, start=1):
            content = "\n\n".join(u.text for u in group).strip()
            if not has_substance(content):
                continue
            drafts.append(
                ChunkDraft(
                    content,
                    {
                        **meta_base,
                        "line_start": group[0].line_start,
                        "line_end": group[-1].line_end,
                        "part": part,
                        "parts": len(groups),
                    },
                    count(content),
                )
            )
    return drafts


def _trim_range(section_lines: list[str], start: int) -> tuple[int, int]:
    first = 0
    while first < len(section_lines) and not section_lines[first].strip():
        first += 1
    last = len(section_lines) - 1
    while last > first and not section_lines[last].strip():
        last -= 1
    return start + first + 1, start + last + 1
