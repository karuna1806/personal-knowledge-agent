"""Fetch a URL and convert HTML to markdown-ish text, so the markdown chunker can split it
on real heading boundaries."""

import re

import httpx
from bs4 import BeautifulSoup, NavigableString, Tag

from pka.ingest.loaders.base import LoadedDocument, UnsupportedSource

MAX_BYTES = 10 * 1024 * 1024
_DROP = ["script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg", "iframe"]
_BLOCK_TAGS = [
    "h1", "h2", "h3", "h4", "h5", "h6", "p", "pre", "ul", "ol", "table", "blockquote",
    "div", "section", "article", "main",
]  # fmt: skip


def html_to_markdown(html: str) -> tuple[str | None, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else None
    for tag in soup(_DROP):
        tag.decompose()
    root = soup.find("main") or soup.find("article") or soup.body or soup
    out: list[str] = []

    def walk(node: Tag) -> None:
        for child in node.children:
            if isinstance(child, NavigableString):
                continue
            if not isinstance(child, Tag):
                continue
            name = child.name
            if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
                text = child.get_text(" ", strip=True)
                if text:
                    out.append("#" * int(name[1]) + " " + text)
            elif name == "pre":
                out.append("```\n" + child.get_text().strip("\n") + "\n```")
            elif name in {"p", "blockquote"}:
                text = child.get_text(" ", strip=True)
                if text:
                    out.append(text)
            elif name in {"ul", "ol"}:
                items = [li.get_text(" ", strip=True) for li in child.find_all("li", recursive=False)]
                items = [i for i in items if i]
                if items:
                    out.append("\n".join(f"- {i}" for i in items))
            elif name == "table":
                rows = []
                for tr in child.find_all("tr"):
                    cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
                    if cells:
                        rows.append("| " + " | ".join(cells) + " |")
                if rows:
                    out.append("\n".join(rows))
            elif child.find(_BLOCK_TAGS) is None:
                # Leaf container (div/span/section with only inline content): one paragraph.
                text = child.get_text(" ", strip=True)
                if text:
                    out.append(text)
            else:
                walk(child)

    walk(root)
    text = "\n\n".join(out)
    if not text.strip():  # page without semantic markup — fall back to raw text
        text = re.sub(r"\n{3,}", "\n\n", root.get_text("\n", strip=True))
    return title, text


async def fetch_url(url: str, timeout_s: float = 20.0) -> tuple[str, bytes]:
    if not url.startswith(("http://", "https://")):
        raise UnsupportedSource("only http(s) URLs are supported")
    async with httpx.AsyncClient(
        follow_redirects=True, timeout=timeout_s, headers={"User-Agent": "pka/0.1"}
    ) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
            body = bytearray()
            async for part in response.aiter_bytes():
                body.extend(part)
                if len(body) > MAX_BYTES:
                    raise UnsupportedSource("response too large")
    return content_type, bytes(body)


async def load_url(url: str) -> LoadedDocument:
    content_type, body = await fetch_url(url)
    if content_type == "application/pdf" or url.lower().endswith(".pdf"):
        from pka.ingest.loaders.pdf import load_pdf_bytes

        doc = load_pdf_bytes(body, url, fallback_title=url)
        doc.metadata["url"] = url
        return doc
    text = body.decode("utf-8", errors="replace")
    if content_type in {"text/html", "application/xhtml+xml"} or "<html" in text[:2000].lower():
        title, text = html_to_markdown(text)
    elif content_type.startswith("text/") or content_type in {"application/json", ""}:
        title = None
    else:
        raise UnsupportedSource(f"unsupported content-type: {content_type}")
    if not text.strip():
        raise UnsupportedSource("page has no text content")
    return LoadedDocument("web", url, title or url, text, {"url": url, "content_type": content_type})
