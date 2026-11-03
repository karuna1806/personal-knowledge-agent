"""Structure-aware chunking per source type (§3.9), plus a fixed-size baseline that exists
only so the chunking experiment in §6 has something to be measured against."""

from pka.ingest.chunkers.base import ChunkDraft, Unit, has_substance, pack_units
from pka.ingest.chunkers.code import chunk_code
from pka.ingest.chunkers.markdown import chunk_markdown
from pka.ingest.loaders.base import LoadedDocument
from pka.ingest.tokens import TokenCounter


def chunk_pdf(pages: list[str], count: TokenCounter, max_tokens: int) -> list[ChunkDraft]:
    """Layout blocks within a page, packed to budget. Chunks never span pages, so every
    citation can name its page."""
    drafts: list[ChunkDraft] = []
    for page_no, page_text in enumerate(pages, start=1):
        if not has_substance(page_text):
            continue
        lines = page_text.split("\n")
        separator = "\n\n" if "\n\n" in page_text else "\n"
        blocks = [b for b in page_text.split(separator) if b.strip()]
        units, cursor = [], 0
        for block in blocks:
            n_lines = block.count("\n") + 1
            units.append(Unit(block.strip(), cursor + 1, cursor + n_lines))
            cursor += n_lines + (1 if separator == "\n\n" else 0)
        for group in pack_units(units, max_tokens, count):
            content = separator.join(u.text for u in group).strip()
            if has_substance(content):
                drafts.append(
                    ChunkDraft(
                        content,
                        {"page": page_no, "line_start": group[0].line_start,
                         "line_end": min(group[-1].line_end, len(lines))},
                        count(content),
                    )  # fmt: skip
                )
    return drafts


def chunk_fixed(text: str, count: TokenCounter, size_chars: int, overlap_chars: int) -> list[ChunkDraft]:
    """Fixed-size character windows with overlap — the naive default, kept as a baseline."""
    if overlap_chars >= size_chars:
        raise ValueError("overlap must be smaller than chunk size")
    drafts: list[ChunkDraft] = []
    step = size_chars - overlap_chars
    for start in range(0, max(len(text), 1), step):
        piece = text[start : start + size_chars]
        if has_substance(piece):
            line_start = text.count("\n", 0, start) + 1
            drafts.append(
                ChunkDraft(
                    piece.strip(),
                    {"char_start": start, "line_start": line_start,
                     "line_end": line_start + piece.count("\n")},
                    count(piece),
                )  # fmt: skip
            )
        if start + size_chars >= len(text):
            break
    return drafts


def chunk_document(
    doc: LoadedDocument,
    count: TokenCounter,
    *,
    strategy: str = "structure",
    max_tokens: int = 512,
    fixed_chars: int = 1500,
    fixed_overlap: int = 200,
) -> list[ChunkDraft]:
    if strategy == "fixed":
        drafts = chunk_fixed(doc.content, count, fixed_chars, fixed_overlap)
    elif strategy != "structure":
        raise ValueError(f"unknown chunking strategy: {strategy}")
    elif doc.source_type == "pdf" and doc.pages is not None:
        drafts = chunk_pdf(doc.pages, count, max_tokens)
    elif doc.source_type == "code":
        drafts = chunk_code(
            doc.content, doc.metadata.get("language", "text"),
            doc.metadata.get("path", doc.source_uri), count, max_tokens,
        )  # fmt: skip
    else:  # note, web
        drafts = chunk_markdown(doc.content, count, max_tokens)
    for draft in drafts:
        draft.metadata.setdefault("chunker", strategy)
    return drafts


def embedding_text(content: str, metadata: dict, title: str | None) -> str:
    """What actually gets embedded: the chunk plus its location (doc title, heading path,
    symbol). A bare paragraph under '§3.5 Message queue' embeds poorly without its heading.
    Only the chunk content is stored and shown; this prefix is for the vector alone."""
    parts: list[str] = []
    if title:
        parts.append(title)
    if metadata.get("heading_path"):
        parts.append(" > ".join(metadata["heading_path"]))
    if metadata.get("symbol"):
        parent = metadata.get("parent")
        parts.append(f"{parent}.{metadata['symbol']}" if parent else metadata["symbol"])
    if metadata.get("page"):
        parts.append(f"page {metadata['page']}")
    prefix = " | ".join(p for p in parts if p)
    return f"{prefix}\n\n{content}" if prefix else content


__all__ = [
    "ChunkDraft",
    "chunk_code",
    "chunk_document",
    "chunk_fixed",
    "chunk_markdown",
    "chunk_pdf",
    "embedding_text",
]
