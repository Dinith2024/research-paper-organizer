from research_assistant.domain import RetrievedChunk
from research_assistant.workflow import _format_context, _heuristic_route


def test_heuristic_routes() -> None:
    assert _heuristic_route("Compare both methods") == "compare"
    assert _heuristic_route("Give me a summary") == "summarize"
    assert _heuristic_route("What dataset was used?") == "ask"


def test_context_has_stable_citation_numbers() -> None:
    sources = [
        RetrievedChunk("1", "Evidence one", "a.pdf", "Paper A", 2, 0.8, "a"),
        RetrievedChunk("2", "Evidence two", "b.pdf", "Paper B", 4, 0.7, "b"),
    ]
    context = _format_context(sources, 2_000)

    assert "[1] Title: Paper A" in context
    assert "[2] Title: Paper B" in context
    assert "Page: 4" in context
