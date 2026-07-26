from __future__ import annotations

import math
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .domain import DocumentChunk, RetrievedChunk


class DeterministicLexicalEmbedding:
    """A lightweight, deterministic embedding that avoids network downloads."""

    def __call__(self, input: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in input:
            normalized = str(text).strip().lower()
            tokens = re.findall(r"[a-z0-9]+", normalized)
            if not tokens:
                vectors.append([0.0, 0.0, 0.0, 0.0, 0.0])
                continue

            token_count = len(tokens)
            unique_count = len(set(tokens))
            first_token = tokens[0]
            first_token_len = len(first_token)
            frequency = Counter(tokens)
            entropy = sum((count / token_count) * math.log2(token_count / count) for count in frequency.values())
            ordinal_signal = sum(ord(ch) for ch in first_token[:3]) / 1000.0
            vectors.append([float(token_count), float(unique_count), float(first_token_len), float(entropy), float(ordinal_signal)])
        return vectors

    def embed_query(self, input: str | list[str]) -> list[list[float]]:
        if isinstance(input, str):
            return self([input])
        return self(input)

    @staticmethod
    def name() -> str:
        return "deterministic-lexical-embedding"

    def get_config(self) -> dict[str, Any]:
        return {}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> DeterministicLexicalEmbedding:
        return DeterministicLexicalEmbedding()


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
        # Set longer timeout for embedding model downloads
        os.environ.setdefault("HTTPX_TIMEOUT", "120")
        
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = _safe_collection_name(collection_name)

        if client is None:
            import chromadb

            client = chromadb.PersistentClient(path=str(self.persist_directory))
        if embedding_function is None:
            embedding_function = DeterministicLexicalEmbedding()

        self.client = client
        self.embedding_function = embedding_function
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self) -> Any:
        existing_collection = None
        try:
            existing_collection = self.client.get_collection(name=self.collection_name)
        except Exception:
            existing_collection = None

        if existing_collection is not None and isinstance(self.embedding_function, DeterministicLexicalEmbedding):
            try:
                self.client.delete_collection(self.collection_name)
            except Exception:
                pass
            existing_collection = None

        return self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
        )

    def upsert_chunks(self, chunks: list[DocumentChunk], batch_size: int = 100) -> None:
        if not chunks:
            return
        # Set longer timeout for httpx to download embedding model
        os.environ.setdefault("HTTPX_TIMEOUT", "120")
        
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            # Retry with exponential backoff for timeout errors
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.collection.upsert(
                        ids=[chunk.id for chunk in batch],
                        documents=[chunk.text for chunk in batch],
                        metadatas=[chunk.metadata for chunk in batch],
                    )
                    break  # Success, exit retry loop
                except Exception as e:
                    error_msg = str(e).lower()
                    if "timeout" in error_msg and attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                        print(f"Timeout during upsert (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        raise

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
