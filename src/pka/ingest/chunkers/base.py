from dataclasses import dataclass, field

from pka.ingest.tokens import TokenCounter


@dataclass
class ChunkDraft:
    content: str
    metadata: dict = field(default_factory=dict)
    token_count: int = 0


@dataclass
class Unit:
    """A contiguous run of lines that must not be split further unless it alone is over budget."""

    text: str
    line_start: int  # 1-based, inclusive
    line_end: int  # 1-based, inclusive
    tokens: int = 0


def split_oversized(unit: Unit, max_tokens: int, count: TokenCounter) -> list[Unit]:
    """Split a single over-budget unit on line boundaries; a single over-budget line is cut
    by characters. Only used as a last resort (e.g. a 600-line function)."""
    lines = unit.text.split("\n")
    out: list[Unit] = []
    buf: list[str] = []
    buf_start = unit.line_start
    buf_tokens = 0
    for offset, line in enumerate(lines):
        line_no = unit.line_start + offset
        line_tokens = count(line) + 1
        if line_tokens > max_tokens:
            if buf:
                out.append(Unit("\n".join(buf), buf_start, line_no - 1, buf_tokens))
                buf, buf_tokens = [], 0
            step = max(1, max_tokens * 3)  # ~3 chars/token: conservative so pieces fit
            for i in range(0, len(line), step):
                piece = line[i : i + step]
                out.append(Unit(piece, line_no, line_no, count(piece)))
            buf_start = line_no + 1
            continue
        if buf and buf_tokens + line_tokens > max_tokens:
            out.append(Unit("\n".join(buf), buf_start, line_no - 1, buf_tokens))
            buf, buf_tokens, buf_start = [], 0, line_no
        if not buf:
            buf_start = line_no
        buf.append(line)
        buf_tokens += line_tokens
    if buf:
        out.append(Unit("\n".join(buf), buf_start, unit.line_start + len(lines) - 1, buf_tokens))
    return out


def pack_units(units: list[Unit], max_tokens: int, count: TokenCounter) -> list[list[Unit]]:
    """Greedily pack consecutive units into groups that fit the budget. Units over budget are
    split first, so no group ever exceeds max_tokens (modulo tokenizer joins)."""
    groups: list[list[Unit]] = []
    current: list[Unit] = []
    current_tokens = 0
    for unit in units:
        if not unit.tokens:
            unit.tokens = count(unit.text)
        pieces = [unit] if unit.tokens <= max_tokens else split_oversized(unit, max_tokens, count)
        for piece in pieces:
            # +2 approximates the "\n\n" joiner between units.
            if current and current_tokens + piece.tokens + 2 > max_tokens:
                groups.append(current)
                current, current_tokens = [], 0
            current.append(piece)
            current_tokens += piece.tokens + 2
    if current:
        groups.append(current)
    return groups


def has_substance(text: str) -> bool:
    return any(ch.isalnum() for ch in text)
