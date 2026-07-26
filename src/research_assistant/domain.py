from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class DocumentChunk:
    id: str
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    text: str
    source: str
    title: str
    page: int
    score: float
    document_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IngestionReport:
    document_id: str
    source: str
    title: str
    author: str
    page_count: int
    chunk_count: int


@dataclass(frozen=True)
class AssistantResult:
    answer: str
    route: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    trace: list[dict[str, str]] = field(default_factory=list)
