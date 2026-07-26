import pytest

from research_assistant.ingestion import IngestionError, split_text, validate_pdf


def test_split_text_produces_bounded_overlapping_chunks() -> None:
    text = " ".join(f"Sentence {index} has useful research content." for index in range(120))
    chunks = split_text(text, chunk_size=300, overlap=50)

    assert len(chunks) > 2
    assert all(1 <= len(chunk) <= 300 for chunk in chunks)
    assert "research content" in chunks[0]


def test_split_text_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        split_text("text", chunk_size=300, overlap=300)


def test_pdf_validation_rejects_wrong_content() -> None:
    with pytest.raises(IngestionError, match="valid PDF"):
        validate_pdf(b"not a pdf", "paper.pdf")


def test_pdf_validation_rejects_wrong_extension() -> None:
    with pytest.raises(IngestionError, match="Only PDF"):
        validate_pdf(b"%PDF-1.7", "paper.txt")
