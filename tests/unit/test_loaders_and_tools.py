from pathlib import Path

import pytest

from pka.ingest.loaders import UnsupportedSource, classify_path, iter_corpus_files, load_path
from pka.ingest.loaders.web import html_to_markdown
from pka.mcp.registry import load_registry
from pka.mcp.servers._common import AccessDenied, resolve_inside
from pka.mcp.servers.postgres import _ALLOWED


def test_classify():
    assert classify_path(Path("a/b.md")) == ("note", None)
    assert classify_path(Path("x.PDF")) == ("pdf", None)
    assert classify_path(Path("x.py")) == ("code", "python")
    assert classify_path(Path("x.ts")) == ("code", "typescript")
    with pytest.raises(UnsupportedSource):
        classify_path(Path("photo.jpg"))


def test_iter_corpus_skips_vcs_and_dependency_dirs(tmp_path):
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "a.md").write_text("# A\n\nx")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD.md").write_text("x")
    (tmp_path / "node_modules" / "pkg").mkdir(parents=True)
    (tmp_path / "node_modules" / "pkg" / "index.js").write_text("x")
    (tmp_path / "img.png").write_bytes(b"\x89PNG")
    found = [p.relative_to(tmp_path).as_posix() for p in iter_corpus_files(tmp_path)]
    assert found == ["notes/a.md"]


def test_binary_rejected(tmp_path):
    f = tmp_path / "x.py"
    f.write_bytes(b"abc\x00def")
    with pytest.raises(UnsupportedSource):
        load_path(f)


def test_pdf_loader_keeps_pages(tmp_path):
    from fpdf import FPDF

    pdf = FPDF()
    for text in ("First page text about queues.", "Second page about vectors."):
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.cell(0, 10, text)
    path = tmp_path / "doc.pdf"
    pdf.output(str(path))
    doc = load_path(path)
    assert doc.source_type == "pdf" and len(doc.pages) == 2
    assert "queues" in doc.pages[0] and "vectors" in doc.pages[1]


def test_html_to_markdown_keeps_structure_and_drops_chrome():
    html = """<html><head><title>T</title><script>evil()</script></head><body>
      <nav>menu</nav><main><h1>Main</h1><p>Para <b>bold</b> text.</p>
      <div>Loose div text</div><ul><li>one</li><li>two</li></ul>
      <pre>code()</pre></main><footer>foot</footer></body></html>"""
    title, text = html_to_markdown(html)
    assert title == "T"
    assert text.split("\n\n") == ["# Main", "Para bold text.", "Loose div text", "- one\n- two", "```\ncode()\n```"]


def test_resolve_inside_blocks_traversal_and_symlinks(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "ok.txt").write_text("x")
    outside = tmp_path / "secret.txt"
    outside.write_text("s")
    (root / "link").symlink_to(outside)
    assert resolve_inside("ok.txt", [root.resolve()]) == (root / "ok.txt").resolve()
    for bad in ("../secret.txt", str(outside), "link"):
        with pytest.raises(AccessDenied):
            resolve_inside(bad, [root.resolve()])


def test_sql_allowlist():
    assert _ALLOWED.match("SELECT 1")
    assert _ALLOWED.match("  with x as (select 1) select * from x")
    for bad in ("delete from documents", "update x set y=1", "drop table chunks", "insert into x values (1)"):
        assert not _ALLOWED.match(bad)


def test_registry_env_expansion(tmp_path, monkeypatch):
    monkeypatch.setenv("SOME_ROOT", "/data")
    monkeypatch.delenv("FLAG", raising=False)
    cfg = tmp_path / "mcp.yaml"
    cfg.write_text(
        "servers:\n"
        "  a:\n    command: ${PYTHON}\n    args: ['-m', 'x']\n    env: {ROOT: '${SOME_ROOT}'}\n"
        "  b:\n    command: echo\n    enabled: ${FLAG:-false}\n"
    )
    servers = {s.name: s for s in load_registry(cfg)}
    assert servers["a"].command.endswith("python") or "python" in servers["a"].command
    assert servers["a"].env == {"ROOT": "/data"}
    assert servers["b"].enabled is False
