import pytest

from research_assistant.config import GROQ_BASE_URL, OPENROUTER_BASE_URL, ProviderConfig


def test_provider_urls_and_headers() -> None:
    groq = ProviderConfig("groq", "secret", "model")
    openrouter = ProviderConfig("openrouter", "secret", "model")

    assert groq.base_url == GROQ_BASE_URL
    assert groq.default_headers == {}
    assert openrouter.base_url == OPENROUTER_BASE_URL
    assert openrouter.default_headers["X-OpenRouter-Title"] == "ResearchFlow AI"


def test_unknown_provider_is_rejected() -> None:
    config = ProviderConfig("unknown", "secret", "model")
    with pytest.raises(ValueError):
        _ = config.base_url


def test_missing_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="Missing API key"):
        ProviderConfig("groq", "", "model").validate()
