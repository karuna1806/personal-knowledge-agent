"""Split source code on function/class boundaries via tree-sitter — never mid-function unless a
single function alone exceeds the token budget. Each chunk carries path, language, symbol and
line range so a citation can point at `file.py:L40-L72`."""

import logging
import re

from pka.ingest.chunkers.base import ChunkDraft, Unit, has_substance, pack_units
from pka.ingest.tokens import TokenCounter

log = logging.getLogger(__name__)

# Node types that are a "definition" across the grammars we ship. Matched by substring so
# e.g. function_definition, function_declaration, method_definition, class_declaration all hit.
_DEF_HINTS = (
    "function", "method", "class", "interface", "struct", "enum", "trait", "impl",
    "module", "mod_item", "namespace", "object_declaration", "type_declaration", "type_alias",
    "decorated_definition", "constructor",
)  # fmt: skip
_NOT_DEF = ("call", "arguments", "parameters", "_type", "expression", "identifier")
_WRAPPERS = ("export_statement",)  # `export function f() {}` / `export class A {}`
_COMMENT = re.compile(r"comment", re.I)


def _is_definition(node) -> bool:
    t = node.type
    if t in _WRAPPERS:
        return any(_is_definition(c) for c in node.children)
    if any(t.endswith(n) for n in _NOT_DEF) and not t.endswith("_definition"):
        return False
    return any(h in t for h in _DEF_HINTS)


def _symbol(node, source: bytes) -> str | None:
    target = node
    if node.type in _WRAPPERS:
        target = next((c for c in node.children if _is_definition(c)), node)
    if target.type == "decorated_definition":
        target = target.child_by_field_name("definition") or target
    name = target.child_by_field_name("name")
    if name is None:
        for child in target.children:
            if child.type in ("identifier", "type_identifier", "name", "constant"):
                name = child
                break
    if name is None:
        return None
    return source[name.start_byte : name.end_byte].decode("utf-8", errors="replace")


def _child_definitions(node) -> list:
    """Definitions directly inside `node` or one level down (class bodies, impl blocks)."""
    if node.type in _WRAPPERS or node.type == "decorated_definition":
        inner = next((c for c in node.children if _is_definition(c)), None)
        return _child_definitions(inner) if inner is not None else []
    found = []
    for child in node.children:
        if _is_definition(child):
            found.append(child)
        elif child.named_child_count:
            found.extend(c for c in child.children if _is_definition(c))
    return found


class _CodeChunker:
    def __init__(self, text: str, language: str, path: str, count: TokenCounter, max_tokens: int):
        self.lines = text.split("\n")
        self.source = text.encode("utf-8")
        self.language = language
        self.path = path
        self.count = count
        self.max_tokens = max_tokens
        self.drafts: list[ChunkDraft] = []

    def text_for(self, start_row: int, end_row: int) -> str:
        return "\n".join(self.lines[start_row : end_row + 1])

    def emit(self, start_row: int, end_row: int, symbol: str | None, kind: str | None,
             parent: str | None = None) -> None:  # fmt: skip
        text = self.text_for(start_row, end_row).strip("\n")
        if not has_substance(text):
            return
        tokens = self.count(text)
        meta = {
            "language": self.language,
            "path": self.path,
            "symbol": symbol,
            "kind": kind,
            "parent": parent,
        }
        if tokens <= self.max_tokens:
            self.drafts.append(
                ChunkDraft(text, {**meta, "line_start": start_row + 1, "line_end": end_row + 1}, tokens)
            )
            return
        # A single definition over budget with no nested definitions: split by lines, and say so.
        groups = pack_units([Unit(text, start_row + 1, end_row + 1, tokens)], self.max_tokens, self.count)
        for part, group in enumerate(groups, start=1):
            content = "\n".join(u.text for u in group)
            self.drafts.append(
                ChunkDraft(
                    content,
                    {**meta, "line_start": group[0].line_start, "line_end": group[-1].line_end,
                     "part": part, "parts": len(groups)},
                    self.count(content),
                )  # fmt: skip
            )

    def flush_misc(self, misc: list[tuple[int, int]]) -> None:
        """Imports, constants, top-level statements: packed together up to the budget."""
        if not misc:
            return
        units = [Unit(self.text_for(s, e), s + 1, e + 1) for s, e in misc]
        for group in pack_units(units, self.max_tokens, self.count):
            # Slice the original source so formatting (blank lines, indentation) survives.
            content = self.text_for(group[0].line_start - 1, group[-1].line_end - 1).strip("\n")
            if has_substance(content):
                self.drafts.append(
                    ChunkDraft(
                        content,
                        {"language": self.language, "path": self.path, "symbol": None,
                         "kind": "module", "parent": None,
                         "line_start": group[0].line_start, "line_end": group[-1].line_end},
                        self.count(content),
                    )  # fmt: skip
                )

    def definition(self, node, start_row: int, parent: str | None = None) -> None:
        end_row = node.end_point.row
        symbol = _symbol(node, self.source)
        tokens = self.count(self.text_for(start_row, end_row))
        children = _child_definitions(node) if tokens > self.max_tokens else []
        if not children:
            self.emit(start_row, end_row, symbol, node.type, parent)
            return
        # Oversized class/impl: header (signature, docstring, fields) + each member separately.
        qualified = f"{parent}.{symbol}" if parent and symbol else symbol
        header_end = children[0].start_point.row - 1
        if header_end >= start_row:
            self.emit(start_row, header_end, symbol, node.type, parent)
        prev_end = header_end
        for child in children:
            child_start = child.start_point.row
            # Keep comments between members attached to the member that follows them.
            gap = self.text_for(prev_end + 1, child_start - 1) if child_start > prev_end + 1 else ""
            if gap.strip() and all(_is_comment_line(line) for line in gap.split("\n") if line.strip()):
                child_start = prev_end + 1
            self.definition(child, child_start, qualified)
            prev_end = child.end_point.row
        if prev_end < end_row:
            self.emit(prev_end + 1, end_row, symbol, node.type, parent)

    def run(self, root) -> list[ChunkDraft]:
        misc: list[tuple[int, int]] = []
        pending_comment_start: int | None = None
        for node in root.children:
            start, end = node.start_point.row, node.end_point.row
            if _COMMENT.search(node.type):
                # Comments directly above a definition belong to it; otherwise they're misc.
                if pending_comment_start is None:
                    pending_comment_start = start
                continue
            if _is_definition(node):
                if pending_comment_start is not None:
                    self.flush_misc(misc)
                    misc = []
                    self.definition(node, pending_comment_start)
                else:
                    self.flush_misc(misc)
                    misc = []
                    self.definition(node, start)
                pending_comment_start = None
                continue
            if pending_comment_start is not None:
                misc.append((pending_comment_start, start - 1))
                pending_comment_start = None
            misc.append((start, end))
        if pending_comment_start is not None:
            misc.append((pending_comment_start, root.end_point.row))
        self.flush_misc(misc)
        return self.drafts


def _is_comment_line(line: str) -> bool:
    s = line.strip()
    return s.startswith(("#", "//", "/*", "*", "--", ";", "'''", '"""')) or s.endswith("*/")


def chunk_lines(text: str, count: TokenCounter, max_tokens: int, meta: dict) -> list[ChunkDraft]:
    """Fallback for unparseable files: blank-line-separated blocks packed to budget."""
    lines = text.split("\n")
    units: list[Unit] = []
    buf: list[str] = []
    start = 0
    for i, line in enumerate(lines):
        if not line.strip():
            if buf:
                units.append(Unit("\n".join(buf), start + 1, i))
                buf = []
            continue
        if not buf:
            start = i
        buf.append(line)
    if buf:
        units.append(Unit("\n".join(buf), start + 1, len(lines)))
    drafts = []
    for group in pack_units(units, max_tokens, count):
        content = "\n".join(lines[group[0].line_start - 1 : group[-1].line_end])
        if has_substance(content):
            drafts.append(
                ChunkDraft(
                    content,
                    {**meta, "line_start": group[0].line_start, "line_end": group[-1].line_end},
                    count(content),
                )
            )
    return drafts


def chunk_code(text: str, language: str, path: str, count: TokenCounter, max_tokens: int) -> list[ChunkDraft]:
    try:
        from tree_sitter_language_pack import get_parser

        parser = get_parser(language)
        tree = parser.parse(text.encode("utf-8"))
    except Exception as exc:
        log.warning("tree-sitter unavailable for %s (%s); using line chunker", language, exc)
        return chunk_lines(text, count, max_tokens, {"language": language, "path": path, "symbol": None})
    drafts = _CodeChunker(text, language, path, count, max_tokens).run(tree.root_node)
    if not drafts and has_substance(text):
        return chunk_lines(text, count, max_tokens, {"language": language, "path": path, "symbol": None})
    return drafts
