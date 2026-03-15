from pathlib import Path
from typing import List

try:
    import fitz
except Exception:
    fitz = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None


def parse_text_from_bytes(content: bytes) -> str:
    """Best-effort parser for plain text resumes/job descriptions."""
    return content.decode("utf-8", errors="ignore").strip()


def parse_text_from_pdf_pymupdf(file_path: str | Path) -> str:
    if fitz is None:
        return ""

    try:
        pages: list[str] = []
        with fitz.open(str(file_path)) as doc:
            for page in doc:
                text = page.get_text("text") or ""
                if text.strip():
                    pages.append(text.strip())
        return "\n\n".join(pages).strip()
    except Exception:
        return ""


def parse_text_from_pdf_pypdf(file_path: str | Path) -> str:
    if PdfReader is None:
        return ""

    try:
        reader = PdfReader(str(file_path))
        pages: list[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages.append(text.strip())
        return "\n\n".join(pages).strip()
    except Exception:
        return ""


def parse_text_from_pdf(file_path: str | Path) -> str:
    # PyMuPDF is usually more robust on Chinese resumes with complex layouts.
    text = parse_text_from_pdf_pymupdf(file_path)
    if text:
        return text
    return parse_text_from_pdf_pypdf(file_path)


def parse_text_from_file(file_path: str | Path) -> str:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        text = parse_text_from_pdf(path)
        if text:
            return text

    try:
        raw = path.read_bytes()
        return parse_text_from_bytes(raw)
    except Exception:
        return ""


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    if not text:
        return []

    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == length:
            break
        start = max(0, end - overlap)

    return chunks
