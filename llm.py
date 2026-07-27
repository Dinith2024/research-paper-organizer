from __future__ import annotations

import json
import re
from typing import Any

from .config import ProviderConfig


class LLMError(RuntimeError):
    """A provider request failed or returned unusable output."""


class LLMGateway:
    def __init__(
        self,
        *,
        answer_config: ProviderConfig,
        router_config: ProviderConfig,
    ) -> None:
        answer_config.validate()
        router_config.validate()
        self.answer_config = answer_config
        self.router_config = router_config
        self._clients: dict[tuple[str, str], Any] = {}

    def _client(self, config: ProviderConfig) -> Any:
        cache_key = (config.provider, config.api_key)
        if cache_key not in self._clients:
            from openai import OpenAI

            self._clients[cache_key] = OpenAI(
                base_url=config.base_url,
                api_key=config.api_key,
                default_headers=config.default_headers or None,
                timeout=60.0,
                max_retries=2,
            )
        return self._clients[cache_key]

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        task: str = "answer",
        max_tokens: int | None = None,
    ) -> str:
        config = self.router_config if task == "router" else self.answer_config
        try:
            response = self._client(config).chat.completions.create(
                model=config.model,
                messages=messages,
                temperature=config.temperature,
                max_tokens=max_tokens or config.max_tokens,
            )
            content = response.choices[0].message.content
        except Exception as exc:
            raise LLMError(
                f"{config.provider} request failed for model '{config.model}': {exc}"
            ) from exc
        if not content or not str(content).strip():
            raise LLMError(f"{config.provider} returned an empty response.")
        return str(content).strip()

    def complete_json(
        self,
        messages: list[dict[str, str]],
        *,
        task: str = "router",
        max_tokens: int = 300,
    ) -> dict[str, Any]:
        raw = self.complete(messages, task=task, max_tokens=max_tokens)
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        candidate = fenced.group(1) if fenced else raw
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1:
            raise LLMError("The model did not return a JSON object.")
        try:
            parsed = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMError(f"The model returned invalid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise LLMError("The model response must be a JSON object.")
        return parsed
