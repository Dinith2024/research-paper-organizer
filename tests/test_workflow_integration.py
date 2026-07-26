from research_assistant.config import WorkflowConfig
from research_assistant.domain import RetrievedChunk
from research_assistant.workflow import ResearchWorkflow


class FakeStore:
    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        assert query
        assert top_k == 2
        return [
            RetrievedChunk(
                id="chunk-1",
                text="The study used a mixed-method design.",
                source="paper.pdf",
                title="Paper",
                page=3,
                score=0.9,
                document_id="doc-1",
            )
        ]


class FakeLLM:
    def complete_json(self, messages, *, task="router", max_tokens=300):
        content = messages[0]["content"]
        if "planner" in content:
            return {"route": "ask", "search_query": "study methodology"}
        return {
            "verdict": "pass",
            "reason": "The factual claim has a valid citation.",
            "revision_instructions": "",
        }

    def complete(self, messages, *, task="answer", max_tokens=None):
        return "The study used a mixed-method design [1]."


def test_complete_langgraph_workflow() -> None:
    workflow = ResearchWorkflow(
        store=FakeStore(),
        llm=FakeLLM(),
        config=WorkflowConfig(top_k=2),
    )

    result = workflow.run(
        question="What methodology was used?",
        requested_mode="auto",
    )

    assert result.route == "ask"
    assert result.answer.endswith("[1].")
    assert len(result.sources) == 1
    assert [event["agent"] for event in result.trace] == [
        "Planner",
        "Retriever",
        "Researcher",
        "Critic",
    ]
