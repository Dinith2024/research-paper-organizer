from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .domain import DocumentChunk, RetrievedChunk


def _safe_collection_name(name: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]", "_", name).strip("._-")
    value = value[:120]
    if len(value) < 3:
        value = f"rag_{value or 'default'}"
    return value


class ChromaResearchStore:
    def __init__(
        self,
        persist_directory: str | Path,
        collection_name: str,
        *,
        client: Any | None = None,
        embedding_function: Any | None = None,
    ) -> None:
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = _safe_collection_name(collection_name)

        if client is None:
            import chromadb

            client = chromadb.PersistentClient(path=str(self.persist_directory))
        if embedding_function is None:
            from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

            embedding_function = ONNXMiniLM_L6_V2()
            embedding_function.DOWNLOAD_PATH = (
                self.persist_directory
                / ".embedding_cache"
                / embedding_function.MODEL_NAME
            )

        self.client = client
        self.embedding_function = embedding_function
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self) -> Any:
        return self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
        )

    def upsert_chunks(self, chunks: list[DocumentChunk], batch_size: int = 100) -> None:
        if not chunks:
            return
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            self.collection.upsert(
                ids=[chunk.id for chunk in batch],
                documents=[chunk.text for chunk in batch],
                metadatas=[chunk.metadata for chunk in batch],
            )

    def search(self, query: str, top_k: int = 6) -> list[RetrievedChunk]:
        count = self.count()
        if count == 0:
            return []
        n_results = max(1, min(top_k, count))
        result = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        retrieved: list[RetrievedChunk] = []

        for item_id, text, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=True
        ):
            metadata = metadata or {}
            distance_value = float(distance) if distance is not None else 1.0
            score = 1.0 / (1.0 + max(0.0, distance_value))
            if not math.isfinite(score):
                score = 0.0
            retrieved.append(
                RetrievedChunk(
                    id=str(item_id),
                    text=str(text or ""),
                    source=str(metadata.get("source", "Unknown source")),
                    title=str(metadata.get("title", metadata.get("source", "Untitled"))),
                    page=int(metadata.get("page", 0) or 0),
                    score=max(0.0, min(1.0, score)),
                    document_id=str(metadata.get("document_id", "")),
                )
            )
        return retrieved

    def count(self) -> int:
        return int(self.collection.count())

    def list_documents(self) -> list[dict[str, Any]]:
        if self.count() == 0:
            return []
        result = self.collection.get(include=["metadatas"])
        grouped: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "document_id": "",
                "source": "",
                "title": "",
                "author": "",
                "pages": 0,
                "chunks": 0,
            }
        )
        for metadata in result.get("metadatas") or []:
            metadata = metadata or {}
            document_id = str(metadata.get("document_id", "unknown"))
            item = grouped[document_id]
            item["document_id"] = document_id
            item["source"] = str(metadata.get("source", "Unknown source"))
            item["title"] = str(metadata.get("title", item["source"]))
            item["author"] = str(metadata.get("author", ""))
            item["pages"] = max(
                int(item["pages"]),
                int(metadata.get("page_count", metadata.get("page", 0)) or 0),
            )
            item["chunks"] = int(item["chunks"]) + 1
        return sorted(grouped.values(), key=lambda item: str(item["title"]).lower())

    def delete_document(self, document_id: str) -> None:
        self.collection.delete(where={"document_id": document_id})

    def clear(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self._get_or_create_collection()
