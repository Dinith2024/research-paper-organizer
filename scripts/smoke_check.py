from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from research_assistant.config import ProviderConfig  # noqa: E402
from research_assistant.ingestion import split_text  # noqa: E402
from research_assistant.workflow import _heuristic_route  # noqa: E402


def main() -> None:
    chunks = split_text("Research evidence. " * 200)
    assert chunks
    assert _heuristic_route("Compare the methods") == "compare"
    assert ProviderConfig("groq", "test", "model").base_url.startswith("https://")
    print("Smoke check passed.")


if __name__ == "__main__":
    main()
