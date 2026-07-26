from __future__ import annotations

import re
from typing import Any, TypedDict

from .config import WorkflowConfig
from .domain import AssistantResult, RetrievedChunk
from .llm import LLMError, LLMGateway
from .prompts import CRITIC_SYSTEM, PLANNER_SYSTEM, RESEARCH_SYSTEM, REVISION_SYSTEM


class ResearchState(TypedDict, total=False):
    question: str
    requested_mode: str
    route: str
    search_query: str
    chat_history: list[dict[str, str]]
    sources: list[RetrievedChunk]
    context: str
    answer: str
    critique: str
    needs_revision: bool
    trace: list[dict[str, str]]


def _trace(state: ResearchState, agent: str, message: str) -> list[dict[str, str]]:
    return [*state.get("trace", []), {"agent": agent, "message": message}]


def _heuristic_route(question: str) -> str:
    normalized = question.lower()
    if any(word in normalized for word in ("compare", "difference", "versus", " vs ")):
        return "compare"
    if any(
        word in normalized
        for word in ("summarize", "summary", "overview", "main findings", "key findings")
    ):
        return "summarize"
    return "ask"


def _format_context(
    sources: list[RetrievedChunk],
    max_characters: int,
) -> str:
    parts: list[str] = []
    used = 0
    for index, source in enumerate(sources, start=1):
        header = (
            f"[{index}] Title: {source.title}\n"
            f"Source: {source.source}\nPage: {source.page}\nExcerpt:\n"
        )
        available = max_characters - used - len(header)
        if available <= 100:
            break
        excerpt = source.text[:available]
        part = f"{header}{excerpt}"
        parts.append(part)
        used += len(part)
    return "\n\n---\n\n".join(parts)


def _history_text(history: list[dict[str, str]], limit: int) -> str:
    recent = history[-limit:]
    if not recent:
        return "(No previous conversation.)"
    return "\n".join(f"{item['role']}: {item['content']}" for item in recent)


class ResearchWorkflow:
    def __init__(
        self,
        *,
        store: Any,
        llm: LLMGateway,
        config: WorkflowConfig | None = None,
    ) -> None:
        self.store = store
        self.llm = llm
        self.config = config or WorkflowConfig()
        self.graph = self._build_graph()

    def _planner(self, state: ResearchState) -> ResearchState:
        requested = state.get("requested_mode", "auto").lower()
        if requested in {"ask", "summarize", "compare"}:
            route = requested
            search_query = state["question"]
        else:
            route = _heuristic_route(state["question"])
            search_query = state["question"]
            try:
                decision = self.llm.complete_json(
                    [
                        {"role": "system", "content": PLANNER_SYSTEM},
                        {"role": "user", "content": state["question"]},
                    ],
                    task="router",
                )
                proposed = str(decision.get("route", "")).lower()
                if proposed in {"ask", "summarize", "compare"}:
                    route = proposed
                search_query = str(decision.get("search_query") or search_query)
            except LLMError:
                pass
        return {
            "route": route,
            "search_query": search_query,
            "trace": _trace(state, "Planner", f"Selected the {route} route."),
        }

    def _retriever(self, state: ResearchState) -> ResearchState:
        query = state.get("search_query") or state["question"]
        if state.get("route") == "summarize":
            query = f"{query} objectives methods findings results limitations conclusions"
        sources = self.store.search(query, top_k=self.config.top_k)
        context = _format_context(sources, self.config.max_context_characters)
        return {
            "sources": sources,
            "context": context,
            "trace": _trace(
                state,
                "Retriever",
                f"Retrieved {len(sources)} passages from ChromaDB.",
            ),
        }

    def _researcher(self, state: ResearchState) -> ResearchState:
        sources = state.get("sources", [])
        if not sources:
            answer = (
                "I could not find relevant text in the indexed papers. Try a more specific "
                "question or add papers that cover this topic."
            )
            return {
                "answer": answer,
                "trace": _trace(state, "Researcher", "No evidence was available."),
            }

        route_instruction = {
            "ask": "Provide a direct, evidence-grounded answer.",
            "summarize": "Create a structured summary of the most relevant evidence.",
            "compare": "Compare the relevant papers or approaches and make differences explicit.",
        }.get(state.get("route", "ask"), "Answer the question.")
        user_prompt = f"""Task: {route_instruction}

Conversation:
{_history_text(state.get("chat_history", []), self.config.max_history_messages)}

Question:
{state["question"]}

Paper excerpts:
{state["context"]}"""
        answer = self.llm.complete(
            [
                {"role": "system", "content": RESEARCH_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            task="answer",
        )
        return {
            "answer": answer,
            "trace": _trace(state, "Researcher", "Synthesized a grounded draft."),
        }

    def _critic(self, state: ResearchState) -> ResearchState:
        sources = state.get("sources", [])
        answer = state.get("answer", "")
        if not sources:
            return {
                "needs_revision": False,
                "critique": "No evidence was retrieved.",
                "trace": _trace(state, "Critic", "Skipped: no evidence to audit."),
            }

        valid_numbers = {str(index) for index in range(1, len(sources) + 1)}
        cited_numbers = set(re.findall(r"\[(\d+)\]", answer))
        deterministic_revision = not bool(cited_numbers & valid_numbers)
        critique = "The answer needs at least one valid source citation."
        needs_revision = deterministic_revision

        try:
            audit = self.llm.complete_json(
                [
                    {"role": "system", "content": CRITIC_SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            f"Question:\n{state['question']}\n\n"
                            f"Answer:\n{answer}\n\n"
                            f"Evidence:\n{state.get('context', '')}"
                        ),
                    },
                ],
                task="router",
            )
            verdict = str(audit.get("verdict", "revise")).lower()
            needs_revision = deterministic_revision or verdict != "pass"
            critique = str(
                audit.get("revision_instructions")
                or audit.get("reason")
                or critique
            )
        except LLMError:
            pass

        message = "Requested one grounded revision." if needs_revision else "Grounding check passed."
        return {
            "needs_revision": needs_revision,
            "critique": critique,
            "trace": _trace(state, "Critic", message),
        }

    def _revise(self, state: ResearchState) -> ResearchState:
        prompt = f"""Question:
{state["question"]}

Draft answer:
{state.get("answer", "")}

Audit:
{state.get("critique", "")}

Paper excerpts:
{state.get("context", "")}"""
        answer = self.llm.complete(
            [
                {"role": "system", "content": REVISION_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            task="answer",
        )
        return {
            "answer": answer,
            "trace": _trace(state, "Researcher", "Revised the answer after critique."),
        }

    def _after_critic(self, state: ResearchState) -> str:
        if self.config.revise_once and state.get("needs_revision"):
            return "revise"
        return "finish"

    def _build_graph(self) -> Any:
        from langgraph.graph import END, START, StateGraph

        builder = StateGraph(ResearchState)
        builder.add_node("planner", self._planner)
        builder.add_node("retriever", self._retriever)
        builder.add_node("researcher", self._researcher)
        builder.add_node("critic", self._critic)
        builder.add_node("revise", self._revise)
        builder.add_edge(START, "planner")
        builder.add_edge("planner", "retriever")
        builder.add_edge("retriever", "researcher")
        builder.add_edge("researcher", "critic")
        builder.add_conditional_edges(
            "critic",
            self._after_critic,
            {"revise": "revise", "finish": END},
        )
        builder.add_edge("revise", END)
        return builder.compile()

    def run(
        self,
        *,
        question: str,
        requested_mode: str = "auto",
        chat_history: list[dict[str, str]] | None = None,
    ) -> AssistantResult:
        question = question.strip()
        if not question:
            raise ValueError("The research question cannot be empty.")
        final_state = self.graph.invoke(
            {
                "question": question,
                "requested_mode": requested_mode,
                "chat_history": chat_history or [],
                "trace": [],
            }
        )
        return AssistantResult(
            answer=str(final_state.get("answer", "")),
            route=str(final_state.get("route", requested_mode)),
            sources=list(final_state.get("sources", [])),
            trace=list(final_state.get("trace", [])),
        )
