from __future__ import annotations

from dataclasses import dataclass

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

GROQ_DEFAULT_ROUTER_MODEL = "llama-3.1-8b-instant"
GROQ_DEFAULT_ANSWER_MODEL = "llama-3.3-70b-versatile"
OPENROUTER_DEFAULT_ANSWER_MODEL = "openai/gpt-oss-20b"


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    api_key: str
    model: str
    temperature: float = 0.15
    max_tokens: int = 1_600

    @property
    def base_url(self) -> str:
        normalized = self.provider.strip().lower()
        if normalized == "groq":
            return GROQ_BASE_URL
        if normalized == "openrouter":
            return OPENROUTER_BASE_URL
        raise ValueError(f"Unsupported provider: {self.provider}")

    @property
    def default_headers(self) -> dict[str, str]:
        if self.provider.strip().lower() != "openrouter":
            return {}
        return {
            "HTTP-Referer": "https://streamlit.io/",
            "X-OpenRouter-Title": "ResearchFlow AI",
        }

    def validate(self) -> None:
        if not self.api_key:
            raise ValueError(f"Missing API key for {self.provider}.")
        if not self.model:
            raise ValueError(f"Missing model name for {self.provider}.")


@dataclass(frozen=True)
class WorkflowConfig:
    top_k: int = 6
    max_context_characters: int = 18_000
    max_history_messages: int = 6
    revise_once: bool = True
