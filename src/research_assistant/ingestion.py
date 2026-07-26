from __future__ import annotations

import hashlib
import re
from io import BytesIO
from pathlib import Path

from .domain import DocumentChunk, IngestionReport

MAX_PDF_BYTES = 30 * 1024 * 1024
DEFAULT_CHUNK_SIZE = 1_200
DEFAULT_CHUNK_OVERLAP = 180


class IngestionError(ValueError):
    """Raised when a user-supplied document cannot be safely processed."""


def validate_pdf(file_bytes: bytes, filename: str) -> None:
    if Path(filename).suffix.lower() != ".pdf":
        raise IngestionError("Only PDF files are supported.")
    if not file_bytes:
        raise IngestionError("The file is empty.")
    if len(file_bytes) > MAX_PDF_BYTES:
        raise IngestionError("The file is larger than the 30 MB limit.")
    if not file_bytes.lstrip().startswith(b"%PDF"):
        raise IngestionError("The file does not appear to be a valid PDF.")


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split text with paragraph preference and a character overlap."""
    if chunk_size < 200:
        raise ValueError("chunk_size must be at least 200 characters.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")

    cleaned = _clean_text(text)
    if not cleaned:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        ideal_end = min(start + chunk_size, len(cleaned))
        end = ideal_end
        if ideal_end < len(cleaned):
            search_floor = start + int(chunk_size * 0.6)
            paragraph_break = cleaned.rfind("\n\n", search_floor, ideal_end)
            sentence_break = max(
                cleaned.rfind(". ", search_floor, ideal_end),
                cleaned.rfind("? ", search_floor, ideal_end),
                cleaned.rfind("! ", search_floor, ideal_end),
            )
            if paragraph_break >= search_floor:
                end = paragraph_break
            elif sentence_break >= search_floor:
                end = sentence_break + 1

        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(cleaned):
            break
        next_start = max(end - overlap, start + 1)
        while next_start < end and not cleaned[next_start].isspace():
            next_start += 1
        start = next_start
    return chunks


def _metadata_value(metadata: object, key: str) -> str:
    value = getattr(metadata, key, "") if metadata is not None else ""
    return str(value or "").strip()


def ingest_pdf_bytes(
    file_bytes: bytes,
    filename: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> tuple[list[DocumentChunk], IngestionReport]:
    validate_pdf(file_bytes, filename)
    try:
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(file_bytes))
    except Exception as exc:
        raise IngestionError(f"PDF reading failed: {exc}") from exc

    if getattr(reader, "is_encrypted", False):
        try:
            unlocked = reader.decrypt("")
        except Exception as exc:
            raise IngestionError("Password-protected PDFs are not supported.") from exc
        if not unlocked:
            raise IngestionError("Password-protected PDFs are not supported.")

    document_id = hashlib.sha256(file_bytes).hexdigest()
    metadata = getattr(reader, "metadata", None)
    title = _metadata_value(metadata, "title") or Path(filename).stem
    author = _metadata_value(metadata, "author")
    chunks: list[DocumentChunk] = []
    text_pages = 0

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            page_text = _clean_text(page.extract_text() or "")
        except Exception:
            page_text = ""
        if not page_text:
            continue
        text_pages += 1
        for chunk_index, chunk_text in enumerate(
            split_text(page_text, chunk_size=chunk_size, overlap=overlap)
        ):
            short_hash = hashlib.sha1(chunk_text.encode("utf-8")).hexdigest()[:12]
            chunk_id = f"{document_id[:20]}-p{page_number}-c{chunk_index}-{short_hash}"
            chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    text=chunk_text,
                    metadata={
                        "document_id": document_id,
                        "source": Path(filename).name,
                        "title": title,
                        "author": author,
                        "page": page_number,
                        "chunk_index": chunk_index,
                        "page_count": len(reader.pages),
                    },
                )
            )

    if not chunks:
        raise IngestionError(
            "No searchable text was found. If this is a scanned PDF, run OCR first."
        )

    report = IngestionReport(
        document_id=document_id,
        source=Path(filename).name,
        title=title,
        author=author,
        page_count=len(reader.pages),
        chunk_count=len(chunks),
    )
    return chunks, report
