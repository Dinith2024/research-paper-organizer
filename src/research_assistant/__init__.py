"""ResearchFlow AI: a LangGraph and ChromaDB research-paper assistant."""

from .config import ProviderConfig, WorkflowConfig
from .domain import AssistantResult, DocumentChunk, IngestionReport, RetrievedChunk

__all__ = [
    "AssistantResult",
    "DocumentChunk",
    "IngestionReport",
    "ProviderConfig",
    "RetrievedChunk",
    "WorkflowConfig",
]

__version__ = "1.0.0"
