import io

from pypdf import PdfReader

from pka.ingest.loaders.base import LoadedDocument, UnsupportedSource


def load_pdf_bytes(data: bytes, source_uri: str, fallback_title: str | None = None) -> LoadedDocument:
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
    except Exception as exc:
        raise UnsupportedSource(f"unreadable PDF: {exc}") from exc
    if not any(pages):
        raise UnsupportedSource("PDF has no extractable text (scanned? OCR is out of scope)")
    title = None
    try:
        title = (reader.metadata.title or "").strip() if reader.metadata else None
    except Exception:
        title = None
    return LoadedDocument(
        source_type="pdf",
        source_uri=source_uri,
        title=title or fallback_title,
        content="\n\n".join(pages),
        metadata={"page_count": len(pages)},
        pages=pages,
    )
