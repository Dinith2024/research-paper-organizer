from research_assistant.domain import DocumentChunk
from research_assistant.vector_store import ChromaResearchStore


class TinyEmbedding:
    def __call__(self, input):
        return [
            [float(len(text)), float(text.lower().count("rag") + 1), 1.0]
            for text in input
        ]

    def embed_query(self, input):
        return self(input)

    @staticmethod
    def name():
        return "tiny-test-embedding"

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(config):
        return TinyEmbedding()


def test_default_embedding_function_is_local(tmp_path) -> None:
    store = ChromaResearchStore(tmp_path, "default_embedding_collection")

    assert store.embedding_function.name() == "deterministic-lexical-embedding"


def test_chroma_upsert_search_and_catalog(tmp_path) -> None:
    store = ChromaResearchStore(
        tmp_path,
        "integration_collection",
        embedding_function=TinyEmbedding(),
    )
    store.upsert_chunks(
        [
            DocumentChunk(
                id="chunk-1",
                text="The paper evaluates RAG with citation grounding.",
                metadata={
                    "document_id": "doc-1",
                    "source": "paper.pdf",
                    "title": "RAG Study",
                    "author": "Test Author",
                    "page": 2,
                    "chunk_index": 0,
                    "page_count": 5,
                },
            )
        ]
    )

    results = store.search("RAG citations", top_k=1)
    catalog = store.list_documents()

    assert store.count() == 1
    assert results[0].title == "RAG Study"
    assert catalog[0]["chunks"] == 1
    assert catalog[0]["pages"] == 5
