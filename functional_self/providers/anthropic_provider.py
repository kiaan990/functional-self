from __future__ import annotations

import os
from typing import Optional

from .base import BrainProvider


class AnthropicProvider(BrainProvider):
    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-4-5", api_key: Optional[str] = None,
                 max_tokens: int = 700, temperature: float = 0.2):
        import anthropic  # lazy import so the mock provider works without the SDK
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"), max_retries=6)
        self.model, self.max_tokens, self.temperature = model, max_tokens, temperature

    in_tokens = 0
    out_tokens = 0

    def complete(self, task: str, system: str, user: str, context: Optional[dict] = None) -> str:
        self.calls += 1
        msg = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, temperature=self.temperature,
            system=system, messages=[{"role": "user", "content": user}],
        )
        self.in_tokens += msg.usage.input_tokens
        self.out_tokens += msg.usage.output_tokens
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
