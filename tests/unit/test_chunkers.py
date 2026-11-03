from pka.ingest.chunkers import (
    chunk_document,
    chunk_fixed,
    chunk_markdown,
    chunk_pdf,
    embedding_text,
)
from pka.ingest.chunkers.code import chunk_code
from pka.ingest.loaders.base import LoadedDocument
from pka.ingest.tokens import estimate_tokens as count

MD = """# Title

Intro paragraph.

## Section A

Alpha text.

```python
# not a heading inside a fence
def f():
    pass
```

### Sub A1

Deep text.

## Section B
## Section C

Gamma text.
"""


def test_markdown_splits_on_headings_with_paths():
    chunks = chunk_markdown(MD, count, max_tokens=512)
    paths = [c.metadata["heading_path"] for c in chunks]
    assert paths == [
        ["Title"],
        ["Title", "Section A"],
        ["Title", "Section A", "Sub A1"],
        ["Title", "Section C"],  # Section B has no body of its own → no chunk
    ]
    assert "# not a heading inside a fence" in chunks[1].content
    assert chunks[1].content.startswith("## Section A")


def test_markdown_line_ranges_point_at_source():
    lines = MD.split("\n")
    for chunk in chunk_markdown(MD, count, max_tokens=512):
        first = lines[chunk.metadata["line_start"] - 1]
        last = lines[chunk.metadata["line_end"] - 1]
        assert chunk.content.startswith(first)
        assert chunk.content.rstrip().endswith(last.rstrip())


def test_markdown_oversized_section_splits_on_paragraphs_within_budget():
    body = "\n\n".join(f"Paragraph {i} " + "word " * 60 for i in range(20))
    text = f"# Big\n\n{body}\n"
    chunks = chunk_markdown(text, count, max_tokens=200)
    assert len(chunks) > 1
    assert all(c.token_count <= 200 for c in chunks)
    assert all(c.metadata["heading_path"] == ["Big"] for c in chunks)
    assert {c.metadata["parts"] for c in chunks} == {len(chunks)}
    # no paragraph is cut in half
    for c in chunks:
        for para in c.content.split("\n\n"):
            assert para.startswith(("Paragraph", "# Big"))


def test_markdown_never_splits_a_fence_across_chunks_when_it_fits():
    fence = "```\n" + "\n".join(f"line {i}" for i in range(30)) + "\n```"
    text = "# H\n\n" + "intro " * 150 + "\n\n" + fence + "\n\n" + "outro " * 150
    chunks = chunk_markdown(text, count, max_tokens=250)
    holders = [c for c in chunks if "```" in c.content]
    assert len(holders) == 1 and holders[0].content.count("```") == 2


PY = '''"""Module docstring."""
import os
import sys

CONSTANT = 1


# helper comment
def helper(x):
    """Help."""
    return x + 1


@decorator
def decorated():
    return 2


class Big:
    """A class."""

    attr = 1

    def method_one(self):
        return 1

    # comment for two
    def method_two(self):
        return 2
'''


def test_python_functions_are_their_own_chunks():
    chunks = chunk_code(PY, "python", "m.py", count, max_tokens=512)
    by_symbol = {c.metadata["symbol"]: c for c in chunks}
    assert by_symbol["helper"].content.startswith("# helper comment")
    assert by_symbol["decorated"].content.startswith("@decorator")
    assert "class Big" in by_symbol["Big"].content and "method_two" in by_symbol["Big"].content
    module = [c for c in chunks if c.metadata["kind"] == "module"]
    assert len(module) == 1 and "import os\nimport sys" in module[0].content


def test_python_line_ranges_are_exact():
    lines = PY.split("\n")
    for c in chunk_code(PY, "python", "m.py", count, max_tokens=512):
        assert c.content == "\n".join(lines[c.metadata["line_start"] - 1 : c.metadata["line_end"]]).strip("\n")


def test_oversized_class_splits_into_methods_never_mid_function():
    chunks = chunk_code(PY, "python", "m.py", count, max_tokens=25)
    symbols = [(c.metadata["symbol"], c.metadata["parent"]) for c in chunks]
    assert ("method_one", "Big") in symbols and ("method_two", "Big") in symbols
    two = next(c for c in chunks if c.metadata["symbol"] == "method_two")
    assert two.content.lstrip().startswith("# comment for two")
    assert all("part" not in c.metadata for c in chunks if c.metadata["symbol"] in ("method_one", "method_two"))


def test_typescript_exported_functions_are_definitions():
    ts = "import x from 'y';\n\nexport function alpha(a: number): number {\n  return a;\n}\n\nexport class Beta {\n  go() { return 1; }\n}\n"
    chunks = chunk_code(ts, "typescript", "a.ts", count, max_tokens=512)
    symbols = {c.metadata["symbol"] for c in chunks}
    assert {"alpha", "Beta"} <= symbols


def test_unparseable_language_falls_back_to_line_chunks():
    chunks = chunk_code("a b c\n\nd e f\n", "no-such-language", "x.weird", count, max_tokens=512)
    assert len(chunks) == 1 and chunks[0].metadata["line_start"] == 1


def test_pdf_chunks_never_span_pages():
    pages = ["Page one para.\n\nSecond para.", "", "Page three."]
    chunks = chunk_pdf(pages, count, max_tokens=512)
    assert [c.metadata["page"] for c in chunks] == [1, 3]


def test_fixed_chunker_overlaps_and_covers_text():
    text = "".join(f"{i:04d} " for i in range(1000))
    chunks = chunk_fixed(text, count, size_chars=500, overlap_chars=100)
    starts = [c.metadata["char_start"] for c in chunks]
    assert starts[:3] == [0, 400, 800]
    assert chunks[-1].content.endswith("0999")


def test_dispatch_and_embedding_text():
    doc = LoadedDocument("note", "n.md", "Doc", MD)
    chunks = chunk_document(doc, count, strategy="structure", max_tokens=512)
    assert all(c.metadata["chunker"] == "structure" for c in chunks)
    text = embedding_text(chunks[1].content, chunks[1].metadata, doc.title)
    assert text.startswith("Doc | Title > Section A\n\n")
