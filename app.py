from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from research_assistant.config import (  # noqa: E402
    GROQ_DEFAULT_ANSWER_MODEL,
    GROQ_DEFAULT_ROUTER_MODEL,
    OPENROUTER_DEFAULT_ANSWER_MODEL,
    ProviderConfig,
    WorkflowConfig,
)
from research_assistant.ingestion import IngestionError, ingest_pdf_bytes  # noqa: E402
from research_assistant.llm import LLMGateway  # noqa: E402
from research_assistant.vector_store import ChromaResearchStore  # noqa: E402
from research_assistant.workflow import ResearchWorkflow  # noqa: E402

st.set_page_config(
    page_title="ResearchFlow AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _load_css() -> None:
    css_path = PROJECT_ROOT / "assets" / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _secret(name: str, default: str = "") -> str:
    value = os.getenv(name, "")
    if value:
        return value
    try:
        return str(st.secrets.get(name, default))
    except (FileNotFoundError, AttributeError, KeyError):
        return default


def _init_state() -> None:
    defaults = {
        "collection_name": f"papers_{uuid.uuid4().hex}",
        "messages": [],
        "last_trace": [],
        "last_sources": [],
        "active_mode": "Ask",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource(show_spinner=False)
def _get_store(collection_name: str) -> ChromaResearchStore:
    persist_dir = Path(_secret("CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "data" / "chroma")))
    return ChromaResearchStore(
        persist_directory=persist_dir,
        collection_name=collection_name,
    )


def _provider_config(
    provider: str,
    api_key: str,
    model: str,
    temperature: float,
) -> ProviderConfig:
    return ProviderConfig(
        provider=provider.lower(),
        api_key=api_key.strip(),
        model=model.strip(),
        temperature=temperature,
    )


def _render_sidebar(store: ChromaResearchStore) -> dict[str, object]:
    with st.sidebar:
        st.markdown("## ResearchFlow")
        st.caption("Grounded answers from your research papers")

        st.markdown("### Model")
        provider_label = st.selectbox("Answer provider", ("Groq", "OpenRouter"))
        provider = provider_label.lower()

        if provider == "groq":
            saved_key = _secret("GROQ_API_KEY")
            default_model = _secret("GROQ_ANSWER_MODEL", GROQ_DEFAULT_ANSWER_MODEL)
        else:
            saved_key = _secret("OPENROUTER_API_KEY")
            default_model = _secret(
                "OPENROUTER_ANSWER_MODEL",
                OPENROUTER_DEFAULT_ANSWER_MODEL,
            )

        api_key = saved_key
        if saved_key:
            st.success(f"{provider_label} is ready", icon="✅")
        else:
            st.warning(
                f"{provider_label} is not configured by the app owner.",
                icon="⚠",
            )

        answer_model = st.text_input("Answer model", value=default_model)

        with st.expander("Retrieval settings"):
            top_k = st.slider("Retrieved passages", 3, 10, 6)
            temperature = st.slider("Answer creativity", 0.0, 1.0, 0.15, 0.05)

        st.markdown("### Workspace")
        count = store.count()
        documents = store.list_documents()
        col_a, col_b = st.columns(2)
        col_a.metric("Papers", len(documents))
        col_b.metric("Chunks", count)

        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_trace = []
            st.session_state.last_sources = []
            st.rerun()

        if st.button("Reset paper library", use_container_width=True):
            store.clear()
            st.session_state.messages = []
            st.session_state.last_trace = []
            st.session_state.last_sources = []
            st.rerun()

        st.caption("PDF text is embedded locally with ChromaDB. API keys are never written to the vector database.")

    groq_key = _secret("GROQ_API_KEY")
    if groq_key:
        router_config = _provider_config(
            "groq",
            groq_key,
            _secret("GROQ_ROUTER_MODEL", GROQ_DEFAULT_ROUTER_MODEL),
            0.05,
        )
    else:
        router_config = _provider_config(
            provider,
            api_key,
            answer_model,
            0.05,
        )

    return {
        "answer_config": _provider_config(
            provider,
            api_key,
            answer_model,
            temperature,
        ),
        "router_config": router_config,
        "top_k": top_k,
        "has_api_key": bool(api_key),
        "provider_label": provider_label,
    }


def _render_header() -> None:
    st.markdown(
        """
        <section class="hero">
          <div class="hero-kicker">LANGGRAPH + CHROMADB</div>
          <h1>Turn papers into answers you can verify.</h1>
          <p>Upload research PDFs, explore their evidence, compare findings, and generate source-linked summaries.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_upload(store: ChromaResearchStore) -> None:
    with st.container(border=True):
        title_col, status_col = st.columns([3, 1])
        with title_col:
            st.markdown("#### Add research papers")
            st.caption("Upload one or more searchable PDFs. Duplicate files are updated, not copied.")
        with status_col:
            st.markdown('<div class="status-pill">● Local embeddings</div>', unsafe_allow_html=True)

        uploads = st.file_uploader(
            "Research papers",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            help="Up to 30 MB per PDF.",
        )
        process = st.button(
            "Process papers",
            type="primary",
            disabled=not uploads,
            use_container_width=False,
        )

        if process and uploads:
            progress = st.progress(0, text="Reading PDFs…")
            reports = []
            errors = []
            total = len(uploads)
            for index, upload in enumerate(uploads, start=1):
                try:
                    chunks, report = ingest_pdf_bytes(upload.getvalue(), upload.name)
                    store.upsert_chunks(chunks)
                    reports.append(report)
                except IngestionError as exc:
                    errors.append(f"{upload.name}: {exc}")
                progress.progress(index / total, text=f"Processed {index} of {total}")
            progress.empty()

            if reports:
                added_chunks = sum(report.chunk_count for report in reports)
                st.success(
                    f"Ready: {len(reports)} paper(s), {added_chunks} searchable chunks.",
                    icon="📚",
                )
            for error in errors:
                st.error(error)


def _render_sources(sources: list[dict[str, object]]) -> None:
    if not sources:
        return
    with st.expander(f"Evidence used · {len(sources)} passages"):
        for index, source in enumerate(sources, start=1):
            page = source.get("page", "?")
            score = float(source.get("score", 0.0))
            st.markdown(f"**[{index}] {source.get('title') or source.get('source')} · page {page}**")
            st.caption(f"Relevance {score:.0%}")
            excerpt = str(source.get("text", "")).strip()
            st.markdown(f"> {excerpt[:650]}{'…' if len(excerpt) > 650 else ''}")


def _render_chat(
    store: ChromaResearchStore,
    runtime: dict[str, object],
) -> None:
    mode_label = st.radio(
        "Research task",
        ("Ask", "Summarize", "Compare", "Auto"),
        horizontal=True,
        key="active_mode",
        help="Auto lets the planner choose the best workflow.",
    )

    if not st.session_state.messages:
        st.markdown(
            """
            <div class="empty-state">
              <h3>Start with a research question</h3>
              <p>Try “Summarize the main methodology,” “What limitations do the authors report?”,
              or “Compare the approaches in these papers.”</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                _render_sources(message.get("sources", []))

    disabled = store.count() == 0
    placeholder = "Process at least one PDF to begin" if disabled else "Ask about your papers…"
    query = st.chat_input(placeholder, disabled=disabled)
    if not query:
        return

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    if not runtime["has_api_key"]:
        answer = (
            f"{runtime['provider_label']} is not configured. The app owner must add its API key "
            "to the server's Streamlit secrets."
        )
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": []}
        )
        with st.chat_message("assistant"):
            st.warning(answer)
        return

    answer_config = runtime["answer_config"]
    router_config = runtime["router_config"]
    gateway = LLMGateway(
        answer_config=answer_config,
        router_config=router_config,
    )
    workflow = ResearchWorkflow(
        store=store,
        llm=gateway,
        config=WorkflowConfig(top_k=int(runtime["top_k"])),
    )
    history = [
        {"role": item["role"], "content": item["content"]}
        for item in st.session_state.messages[-8:-1]
    ]

    with st.chat_message("assistant"):
        with st.status("Running research workflow…", expanded=False) as status:
            try:
                result = workflow.run(
                    question=query,
                    requested_mode=mode_label.lower(),
                    chat_history=history,
                )
                status.update(label="Answer grounded in your papers", state="complete")
                st.markdown(result.answer)
                source_dicts = [source.to_dict() for source in result.sources]
                _render_sources(source_dicts)
            except Exception as exc:  # Streamlit boundary: show a friendly error.
                status.update(label="Workflow stopped", state="error")
                st.error(
                    "The workflow could not finish. Check the API key/model name and try again."
                )
                with st.expander("Technical details"):
                    st.code(str(exc))
                return

    st.session_state.last_trace = result.trace
    st.session_state.last_sources = source_dicts
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.answer,
            "sources": source_dicts,
        }
    )


def _render_library(store: ChromaResearchStore) -> None:
    documents = store.list_documents()
    if not documents:
        st.info("No papers have been processed in this workspace yet.")
        return

    st.markdown("#### Indexed papers")
    for document in documents:
        with st.container(border=True):
            col_a, col_b = st.columns([4, 1])
            with col_a:
                st.markdown(f"**{document['title']}**")
                details = f"{document['source']} · {document['pages']} page(s)"
                if document.get("author"):
                    details += f" · {document['author']}"
                st.caption(details)
            with col_b:
                st.metric("Chunks", document["chunks"])

    options = {f"{item['title']} · {item['source']}": item["document_id"] for item in documents}
    selected = st.selectbox("Remove one paper", options=tuple(options.keys()))
    if st.button("Remove selected paper"):
        store.delete_document(options[selected])
        st.rerun()


def _render_workflow() -> None:
    st.markdown("#### Agent workflow")
    stages = [
        ("01", "Planner", "Chooses ask, summarize, or compare."),
        ("02", "Retriever", "Finds relevant ChromaDB passages."),
        ("03", "Researcher", "Synthesizes a cited answer."),
        ("04", "Critic", "Checks grounding and revises once if needed."),
    ]
    columns = st.columns(4)
    for column, (number, title, detail) in zip(columns, stages, strict=True):
        with column:
            st.markdown(
                f"""
                <div class="workflow-card">
                  <span>{number}</span>
                  <h4>{title}</h4>
                  <p>{detail}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if st.session_state.last_trace:
        st.markdown("#### Last run")
        for event in st.session_state.last_trace:
            st.markdown(
                f"**{event.get('agent', 'Agent')}** — {event.get('message', '')}"
            )
    else:
        st.caption("The execution trace will appear after the first answer.")


def main() -> None:
    _load_css()
    _init_state()
    store = _get_store(st.session_state.collection_name)
    runtime = _render_sidebar(store)
    _render_header()
    _render_upload(store)

    chat_tab, library_tab, workflow_tab = st.tabs(
        ("Research chat", "Paper library", "Workflow")
    )
    with chat_tab:
        _render_chat(store, runtime)
    with library_tab:
        _render_library(store)
    with workflow_tab:
        _render_workflow()


if __name__ == "__main__":
    main()
